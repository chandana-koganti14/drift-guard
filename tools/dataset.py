"""Build evaluation data from your Claude Code history, with no hand labelling.

Three sets:
  natural     every real prompt with 2+ earlier prompts in its session. Labelled
              later by a teacher model (silver labels).
  behavioral  real topic switches proven by your own behaviour: you started a
              fresh session in the same project minutes after the previous one.
              The old session's history + the new session's first prompt.
  synthetic   one session's history + the opening prompt of a session from a
              different project. Guaranteed switches, cleaner than real ones.

Secrets are redacted while transcripts are parsed (see drift_core.redact).
"""

import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugins/drift-guard/scripts"))
import drift_core as dg  # noqa: E402

MAX_GAP_MIN = 30  # a new session this soon after the last one = deliberate fresh start


def load_sessions(exclude: list[str]) -> list[dict]:
    sessions = []
    for path in sorted(Path.home().glob(".claude/projects/*/*.jsonl")):
        project = path.parent.name
        if dg.excluded(project, exclude):
            continue
        turns, final_context = dg.parse_transcript(path)
        if turns:
            sessions.append({"id": path.stem[:8], "project": project, "turns": turns,
                             "final_context": final_context})
    return sessions


def _example(eid: str, project: str, history, prompt: str, context_tokens: int, **extra) -> dict:
    return {"id": eid, "project": project, "context_tokens": context_tokens,
            "state": dg.build_state(history, prompt), **extra}


def natural(sessions: list[dict], limit: int, seed: int = 7) -> list[dict]:
    examples = [
        _example(f"nat-{s['id']}-{i}", s["project"], s["turns"][:i], s["turns"][i].prompt,
                 s["turns"][i].context_tokens_before)
        for s in sessions for i in range(2, len(s["turns"]))
    ]
    random.Random(seed).shuffle(examples)
    return examples[:limit]


def _time(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def behavioral(sessions: list[dict]) -> list[dict]:
    by_project: dict[str, list[dict]] = {}
    for s in sessions:
        if _time(s["turns"][0].ts):
            by_project.setdefault(s["project"], []).append(s)
    examples = []
    for project, group in by_project.items():
        group.sort(key=lambda s: _time(s["turns"][0].ts))
        for a, b in zip(group, group[1:]):
            end, start = _time(a["turns"][-1].last_ts), _time(b["turns"][0].ts)
            if not end or not start:
                continue
            gap_min = (start - end).total_seconds() / 60
            opener = b["turns"][0].prompt
            if (0 <= gap_min <= MAX_GAP_MIN and len(a["turns"]) >= 3 and len(opener.split()) >= 4
                    and opener not in {t.prompt for t in a["turns"]}):  # skip resumed copies
                examples.append(_example(f"beh-{a['id']}-{b['id']}", project, a["turns"], opener,
                                         a["final_context"], gap_min=round(gap_min, 1)))
    return examples


def synthetic(sessions: list[dict], n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    openers = [(s["id"], s["project"], s["turns"][0].prompt) for s in sessions
               if len(s["turns"][0].prompt.split()) >= 4]
    hosts = [(s, i) for s in sessions for i in range(3, len(s["turns"]) + 1)]
    rng.shuffle(hosts)
    examples = []
    for s, i in hosts:
        pool = [o for o in openers if o[1] != s["project"]]
        if not pool:
            continue
        oid, oproject, opener = rng.choice(pool)
        examples.append(_example(f"syn-{s['id']}-{i}-{oid}", f"{s['project']} <- {oproject}",
                                 s["turns"][:i], opener, s["turns"][i - 1].context_tokens_before))
        if len(examples) >= n:
            break
    return examples
