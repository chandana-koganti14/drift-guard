"""Drift Guard core: read a Claude Code transcript, measure how much stale
context is loaded, and ask a judge whether the new prompt starts a new topic.

The judge speaks Jev's System One interface (state + a typed Choice question).
Backends, picked automatically:
  jev        real Jev via typesafe-sdk          (TYPESAFE_API_KEY set)
  adapter    TypeSafe's system-one-adapter on an LLM (OPENAI_API_KEY set)
  heuristic  offline word-overlap baseline, stdlib only
"""

import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path(os.environ.get("DRIFT_GUARD_HOME", Path.home() / ".drift-guard"))

DEFAULTS = {
    "judge": "auto",  # auto | jev | adapter | heuristic
    "min_context_tokens": 50_000,  # stay quiet while the context is small
    "min_confidence": 0.8,  # Jev-style Choice confidence needed to speak up
    "min_prior_prompts": 3,  # need some history before "drift" means anything
    "cooldown_prompts": 3,  # never warn twice within this many prompts
    "adapter_provider": "openai",
    "adapter_model": "gpt-4.1-mini",
    "judge_budget_ms": 1500,  # hard latency budget for the live hook
    "teacher_model": "gpt-4.1",  # benchmark only: labels examples so no human has to
    "exclude_projects": "",  # comma-separated; matching projects never leave the machine
    # dialog: Claude asks you with its clickable question UI before answering
    # ask:    hold the prompt until you /clear or resend it (no tokens spent)
    # warn:   show a note and let the prompt through
    "mode": "dialog",
}


def load_config() -> dict:
    """Defaults <- ~/.drift-guard/config.env <- DRIFT_GUARD_* environment variables.

    Non DRIFT_GUARD_ keys in config.env (API keys) are exported to the
    environment for the SDKs, without overriding values already set.
    """
    cfg = dict(DEFAULTS)
    env_file = HOME / "config.env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = (s.strip().strip('"').strip("'") for s in line.split("=", 1))
            if key.startswith("DRIFT_GUARD_"):
                cfg[key[len("DRIFT_GUARD_"):].lower()] = value
            else:
                os.environ.setdefault(key, value)
    for key in DEFAULTS:
        if f"DRIFT_GUARD_{key.upper()}" in os.environ:
            cfg[key] = os.environ[f"DRIFT_GUARD_{key.upper()}"]
    for key, default in DEFAULTS.items():
        if isinstance(default, int):
            cfg[key] = int(cfg[key])
        elif isinstance(default, float):
            cfg[key] = float(cfg[key])
    return cfg


# ---------------------------------------------------------------- transcript

SECRET_RE = re.compile(
    r"(sk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}|apikey_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9\-]{10,})"
)
NOISE_TAG_RE = re.compile(
    r"<(system-reminder|ide_[a-z_]+|command-[a-z]+|local-command-[a-z]+)>.*?</\1>", re.S
)
PASTE_RE = re.compile(r'<pasted_content id="[^"]*">\s*(.*?)\s*</pasted_content id="[^"]*">', re.S)
PATH_KEYS = ("file_path", "path", "notebook_path")


def redact(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text)


def clean_prompt(text: str) -> str:
    text = NOISE_TAG_RE.sub("", text)
    text = PASTE_RE.sub(lambda m: f"[pasted text: {m.group(1)[:400]} ...]", text)
    return redact(text).strip()


def _user_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


@dataclass
class Turn:
    prompt: str
    context_tokens_before: int  # context size when this prompt was sent
    files: list[str] = field(default_factory=list)
    reply: str = ""
    ts: str = ""  # when the prompt was sent (ISO 8601)
    last_ts: str = ""  # last activity in this turn


def parse_transcript(path: str | Path) -> tuple[list[Turn], int]:
    """Return the main-thread user turns and the current context size in tokens."""
    turns: list[Turn] = []
    context_tokens = 0
    if not Path(path).exists():  # first prompt of a new session: not written yet
        return turns, context_tokens
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("isSidechain"):
                continue
            kind = d.get("type")
            if turns and d.get("timestamp"):
                turns[-1].last_ts = d["timestamp"]
            if kind == "system" and d.get("subtype") == "compact_boundary":
                turns = []  # compaction replaced the old context
                continue
            if kind == "user":
                if d.get("isMeta") or d.get("isCompactSummary") or "toolUseResult" in d:
                    continue
                text = _user_text(d.get("message", {}).get("content"))
                if not text or text.startswith("[Request interrupted"):
                    continue
                prompt = clean_prompt(text)
                if prompt and not prompt.startswith("/"):
                    ts = d.get("timestamp", "")
                    turns.append(Turn(prompt=prompt, context_tokens_before=context_tokens, ts=ts, last_ts=ts))
            elif kind == "assistant":
                msg = d.get("message", {})
                usage = msg.get("usage") or {}
                if usage:
                    context_tokens = (
                        usage.get("input_tokens", 0)
                        + usage.get("cache_creation_input_tokens", 0)
                        + usage.get("cache_read_input_tokens", 0)
                    )
                if not turns:
                    continue
                for block in msg.get("content") or []:
                    if block.get("type") == "tool_use":
                        for key in PATH_KEYS:
                            value = (block.get("input") or {}).get(key)
                            if isinstance(value, str) and value not in turns[-1].files:
                                turns[-1].files.append(value)
                    elif block.get("type") == "text" and len(turns[-1].reply) < 1500:
                        turns[-1].reply += block.get("text", "")
    return turns, context_tokens


def build_state(history: list[Turn], new_prompt: str) -> dict:
    """The block of state the judge evaluates (Jev 'state')."""
    recent = history[-4:]
    files: list[str] = []
    for t in reversed(history[-6:]):
        for p in t.files:
            short = "/".join(Path(p).parts[-2:])
            if short not in files:
                files.append(short)
    return {
        "recent_user_prompts": [t.prompt[:600] for t in recent],
        "files_touched_recently": files[:15],
        "last_assistant_reply_excerpt": redact(recent[-1].reply[:800]) if recent else "",
        "new_prompt": clean_prompt(new_prompt)[:1500],
    }


# ---------------------------------------------------------------- judges

LABELS = ("continuation", "related_subtask", "new_topic")
INSTRUCTIONS = (
    "A developer is in a long coding-assistant session. Compare new_prompt with the "
    "recent prompts, files and reply. How does new_prompt relate to the recent work?"
)
CRITERIA = {
    "continuation": "Follows up on, fixes, refines, or asks about the recent work.",
    "related_subtask": "A new step in the same project that still benefits from the recent context.",
    "new_topic": "Starts an unrelated task; the recent conversation would not help answer it.",
}


@dataclass
class Verdict:
    label: str
    confidence: float
    probabilities: dict[str, float]
    backend: str
    latency_ms: int = 0


def choice_confidence(probabilities: dict[str, float]) -> float:
    """TypeSafe's documented Choice confidence: (n * peak - 1) / (n - 1)."""
    n = len(probabilities)
    peak = max(probabilities.values())
    return max(0.0, min(1.0, (n * peak - 1) / (n - 1)))


def _from_choice_answer(answer, backend: str) -> Verdict:
    probs = {k: float(v) for k, v in (answer.probabilities or {}).items()}
    conf = answer.confidence if getattr(answer, "confidence", None) is not None else choice_confidence(probs)
    return Verdict(answer.choice, float(conf), probs, backend)


def judge_jev(state: dict, cfg: dict) -> Verdict:
    from typesafe_sdk import Choice, TypeSafeClient

    with TypeSafeClient() as client:
        response = client.system_one(
            state=state, questions={"relation": Choice(instructions=INSTRUCTIONS, criteria=CRITERIA)}
        )
    return _from_choice_answer(response.choices["relation"], "jev")


def judge_adapter(state: dict, cfg: dict) -> Verdict:
    from system_one_adapter import Choice, SystemOneAdapterClient

    with SystemOneAdapterClient(
        structured_outputs=True, llm_answer_mode="probabilities", normalize_probabilities=True
    ) as client:
        response = client.system_one(
            state=state,
            questions={"relation": Choice(instructions=INSTRUCTIONS, criteria=CRITERIA)},
            provider=cfg["adapter_provider"],
            model=cfg["adapter_model"],
        )
    return _from_choice_answer(response.choices["relation"], f"adapter:{cfg['adapter_model']}")


STOPWORDS = set(
    """the a an and or but if then else for to of in on at by with from as is are was were be been
    it its this that these those i you we they he she me my your our their can could would should will
    do does did done have has had not no yes so just also please make want need like get got let lets
    what how why when where which who there here about into out up down over more some any all one two
    now new use using used file files code thing things way ok okay really very much well sure""".split()
)
CONTINUE_RE = re.compile(
    r"^\s*(it|that|this|those|these|also|and|but|now|then|still|again|same|ok|okay|yes|no|nope|why|"
    r"what about|how about|instead|fix|try|retry|continue|go ahead|do it|looks|great|thanks|perfect)\b",
    re.I,
)
NEW_TOPIC_RE = re.compile(
    r"\b(new (idea|task|project|question|topic|feature)|different (thing|topic|question|project)|"
    r"unrelated|switch(ing)? (to|gears)|moving on|another (thing|question|project|idea)|separately)\b",
    re.I,
)


def _terms(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return [w for w in re.findall(r"[a-z][a-z0-9]{2,}", text.lower()) if w not in STOPWORDS]


def judge_heuristic(state: dict, cfg: dict) -> Verdict:
    """Offline baseline: TF-IDF overlap plus a few phrasing cues. Not calibrated."""
    prompt = state["new_prompt"]
    context_docs = state["recent_user_prompts"] + [state["last_assistant_reply_excerpt"]]
    docs = [Counter(_terms(d)) for d in context_docs] + [Counter(_terms(prompt))]
    df = Counter(t for d in docs for t in d)
    idf = {t: math.log((1 + len(docs)) / (1 + c)) + 1 for t, c in df.items()}
    recent = Counter()
    for d in docs[:-1]:
        recent.update(d)
    for f in state["files_touched_recently"]:
        recent.update(_terms(f))
    new = docs[-1]

    def vec(c):
        return {t: n * idf.get(t, 1.0) for t, n in c.items()}

    a, b = vec(new), vec(recent)
    dot = sum(a[t] * b.get(t, 0) for t in a)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    sim = dot / norm if norm else 0.0

    names = {Path(f).name.lower() for f in state["files_touched_recently"]}
    file_ref = any(n and n in prompt.lower() for n in names)
    words = len(prompt.split())
    z = 2.0 - 25 * sim + 2.5 * bool(NEW_TOPIC_RE.search(prompt)) - 2.5 * bool(CONTINUE_RE.search(prompt))
    # 1-3 word prompts are almost always answers to the assistant ("c", "yes, go")
    z -= 3.0 * file_ref + (4.0 if words <= 3 else 1.5 if words <= 6 else 0)
    p_new = 1 / (1 + math.exp(-z))
    probs = {"continuation": (1 - p_new) * 0.7, "related_subtask": (1 - p_new) * 0.3, "new_topic": p_new}
    label = max(probs, key=probs.get)
    return Verdict(label, choice_confidence(probs), probs, "heuristic")


JUDGES = {"jev": judge_jev, "adapter": judge_adapter, "heuristic": judge_heuristic}


def excluded(project: str, patterns: list[str] | str) -> bool:
    if isinstance(patterns, str):
        patterns = [p for p in patterns.split(",") if p.strip()]
    return any(p.strip().lower() in project.lower() for p in patterns)


def pick_backend(cfg: dict, cwd: str = "") -> str:
    if cwd and excluded(cwd, cfg["exclude_projects"]):
        return "heuristic"  # privacy: excluded projects are judged locally only
    if cfg["judge"] != "auto":
        return cfg["judge"]
    if os.environ.get("TYPESAFE_API_KEY"):
        return "jev"
    if os.environ.get("OPENAI_API_KEY"):
        return "adapter"
    return "heuristic"


BREAKER_FILE = HOME / "breaker.json"
BREAKER_TRIPS_AFTER = 3  # consecutive remote failures
BREAKER_COOLDOWN_S = 600


def _breaker() -> dict:
    try:
        return json.loads(BREAKER_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _record_remote(backend: str, ok: bool) -> None:
    state = _breaker()
    entry = state.get(backend, {"failures": 0, "open_until": 0})
    if ok:
        entry = {"failures": 0, "open_until": 0}
    else:
        entry["failures"] += 1
        if entry["failures"] >= BREAKER_TRIPS_AFTER:
            entry["open_until"] = time.time() + BREAKER_COOLDOWN_S
    state[backend] = entry
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        BREAKER_FILE.write_text(json.dumps(state))
    except OSError:
        pass


def run_judge(backend: str, state: dict, cfg: dict, budget_ms: int | None = None,
              use_breaker: bool = True) -> Verdict:
    """Run a judge inside a latency budget.

    Remote judges degrade to the local heuristic on timeout, error, missing SDK,
    or while their circuit breaker is open, so a slow or failing API can never
    hold up the user's prompt. budget_ms=None means no budget (offline tools).
    """
    start = time.monotonic()
    fallback_reason = ""
    if use_breaker and backend != "heuristic" and _breaker().get(backend, {}).get("open_until", 0) > time.time():
        fallback_reason = "breaker_open"
    elif backend != "heuristic":
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(JUDGES[backend], state, cfg)
        try:
            verdict = future.result(timeout=None if budget_ms is None else budget_ms / 1000)
            if use_breaker:
                _record_remote(backend, ok=True)
        except ImportError:
            fallback_reason = "sdk_missing"
        except FuturesTimeout:
            fallback_reason = "timeout"
            if use_breaker:
                _record_remote(backend, ok=False)
        except Exception as exc:  # network, auth, rate limit, schema
            fallback_reason = f"error:{type(exc).__name__}"
            if use_breaker:
                _record_remote(backend, ok=False)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
    if backend == "heuristic" or fallback_reason:
        verdict = judge_heuristic(state, cfg)
        if fallback_reason:
            verdict.backend = f"heuristic (fallback: {backend} {fallback_reason})"
    verdict.latency_ms = int((time.monotonic() - start) * 1000)
    return verdict


# ---------------------------------------------------------------- decision

def decide(history: list[Turn], context_tokens: int, prompt: str, cfg: dict, backend: str) -> dict:
    """Cheap gates in code first; the judge only runs when a warning could matter."""
    if prompt.strip().startswith("/"):
        return {"fire": False, "reason": "slash_command"}
    if len(history) < cfg["min_prior_prompts"]:
        return {"fire": False, "reason": "too_little_history"}
    if context_tokens < cfg["min_context_tokens"]:
        return {"fire": False, "reason": "context_small", "context_tokens": context_tokens}
    state = build_state(history, prompt)
    v = run_judge(backend, state, cfg, budget_ms=cfg["judge_budget_ms"])
    fire = v.label == "new_topic" and v.confidence >= cfg["min_confidence"]
    return {
        "fire": fire,
        "reason": "new_topic_confident" if fire else f"{v.label}@{v.confidence:.2f}",
        "context_tokens": context_tokens,
        "verdict": v.__dict__,
    }


def warning_text(result: dict, checkpoint: str, mode: str) -> str:
    v = result["verdict"]
    k = round(result["context_tokens"] / 1000)
    why = f"new topic ({v['backend']} confidence {v['confidence']:.2f}); this chat still holds ~{k}k tokens of the old task."
    if mode == "ask":
        return (f"🧭 Drift Guard paused this prompt: {why}\n"
                f"   → Fresh start: type /clear (or open a new chat), then send your prompt again.\n"
                f"   → Continue here anyway: send the same prompt again.\n"
                f"   Previous task saved: {checkpoint}")
    return f"🧭 Drift Guard: {why} Consider /clear, then send this prompt again. Previous task saved: {checkpoint}"


def judge_label(backend: str) -> str:
    """Human name for the judge shown in the dialog."""
    if backend == "jev":
        return "Jev"
    if backend.startswith("adapter:"):
        return backend.split(":", 1)[1]
    return "local check"


def dialog_instruction(result: dict, checkpoint: str) -> str:
    """Context for Claude: ask the user with the native AskUserQuestion UI before answering."""
    v = result["verdict"]
    k = round(result["context_tokens"] / 1000)
    who = judge_label(v["backend"])
    return (
        "Drift Guard, a hook the user installed, detected that this prompt starts a NEW TOPIC unrelated to "
        f"the earlier conversation ({who} confidence {v['confidence']:.2f}; ~{k}k tokens of earlier context; "
        f"previous task saved to {checkpoint}). Before doing anything else, call AskUserQuestion with exactly "
        "one question, using these texts verbatim:\n"
        f'- question: "🧭 New topic detected ({who} confidence {v["confidence"]:.2f}). Start fresh so ~{k}k tokens '
        'of old context don\'t carry over?"\n'
        '- header: "Drift Guard"\n'
        '- option 1, label "Start fresh (Recommended)", description "Type /clear or click New chat, then send '
        'your prompt again. Your previous task is saved."\n'
        '- option 2, label "Continue here", description "Answer in this chat, keeping the earlier context."\n'
        "If the user picks Start fresh: do not answer the prompt and use no other tools; reply in one short "
        "line telling them to type /clear (or click New chat) and send their prompt again, and give the saved "
        "checkpoint path. If they pick Continue here or write their own answer: answer the prompt normally. "
        "If AskUserQuestion is unavailable, mention the topic switch in one line, then answer."
    )
