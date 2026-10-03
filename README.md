<p align="center"><img src="docs/icon.svg" width="128" alt="Drift Guard icon"></p>

# Drift Guard

**Tells you when to `/clear` in Claude Code, and saves your place when you do.**

Long sessions collect context: files, errors, decisions. Switch to an unrelated task
in the same session and all of it rides along: every message re-sends it, spending tokens
and pulling answers off course. Drift Guard notices the switch, checkpoints the old task,
and has Claude **ask you before answering**, using Claude Code's own question UI:

```
┌ New topic ─────────────────────────────────────────────────────────┐
│ This looks like a new topic. Start fresh so the old context        │
│ doesn't carry over?                                                │
│  ❯ Start fresh (Recommended)  type /clear or click New chat, then │
│                               resend. Previous task saved.         │
│    Continue here              answer in this chat                  │
└────────────────────────────────────────────────────────────────────┘
```

| Mode (`DRIFT_GUARD_MODE`) | What happens on a topic switch |
|---|---|
| `dialog` (default) | Claude asks you with clickable options before writing anything |
| `ask` | The prompt is held before Claude sees it (zero tokens spent); resend it to continue |
| `warn` | A one-line note; the prompt goes through |

Hooks can't run `/clear` themselves, so "Start fresh" still means typing `/clear` or
clicking New chat. Your previous task is saved, so nothing is lost.

## How it works

```
prompt ─► GATE ─────────► JUDGE ─────────────────► POLICY ──────► ACTION ─────────► EVENT LOG
          ~1 ms, local     Jev Choice question       new_topic      ask + save        append-only
          · ≥3 prompts     1.5 s latency budget      AND conf ≥0.8  a checkpoint      events.jsonl
          · ≥50k context   timeout → local judge     cooldown
          · cooldown       circuit breaker
                           excluded project → local
/clear ─► SessionStart hook ─► last warning marked accepted ─► acceptance rate = live precision
resend ─► same prompt again ─► let through, logged as override
```

The judge is one [Jev](https://docs.typesafe.ai) `Choice` question (`continuation`,
`related_subtask` or `new_topic`) that returns calibrated probabilities. Everything around
it is plain, testable code. Design rationale: [docs/DESIGN.md](docs/DESIGN.md).

| Judge | When | |
|---|---|---|
| `jev` | `TYPESAFE_API_KEY` set | Calibrated decision model, ~250 ms |
| `adapter` | `OPENAI_API_KEY` set | Same interface via TypeSafe's [system-one-adapter](https://github.com/typesafe-ai/system-one-adapter-python) |
| `heuristic` | always available | Local TF-IDF + phrasing cues. Fallback and baseline |

## Install

```bash
# 1. SDKs in a private venv
python3 -m venv ~/.drift-guard/venv
~/.drift-guard/venv/bin/pip install typesafe-sdk 'system-one-adapter[openai]'

# 2. Keys, outside any repo, readable only by you
touch ~/.drift-guard/config.env && chmod 600 ~/.drift-guard/config.env
echo 'TYPESAFE_API_KEY=...' >> ~/.drift-guard/config.env

# 3. Hooks: per project (recommended) or for every project
python3 tools/install.py ~/code/my-project
python3 tools/install.py --user
```

Works in the terminal CLI and in the IDE extensions. A plugin marketplace is also included
(`/plugin marketplace add <this repo>` inside Claude Code).

### Settings (`~/.drift-guard/config.env`)

| Setting | Default | |
|---|---|---|
| `DRIFT_GUARD_MODE` | `dialog` | `dialog`, `ask` or `warn` (see above) |
| `DRIFT_GUARD_JUDGE` | `auto` | `auto`, `jev`, `adapter`, `heuristic` |
| `DRIFT_GUARD_MIN_CONFIDENCE` | `0.8` | Confidence needed to warn |
| `DRIFT_GUARD_MIN_CONTEXT_TOKENS` | `50000` | Stay quiet below this much context |
| `DRIFT_GUARD_COOLDOWN_PROMPTS` | `3` | Never warn twice within this many prompts |
| `DRIFT_GUARD_JUDGE_BUDGET_MS` | `1500` | Latency budget before falling back to the local judge |
| `DRIFT_GUARD_EXCLUDE_PROJECTS` | – | Comma-separated: matching projects are judged locally only, never sent to an API |

## Measure it (no labelling)

```bash
~/.drift-guard/venv/bin/python tools/benchmark.py   # heuristic vs Jev vs GPT-4.1-mini on your history
python3 tools/stats.py                              # live: warnings, acceptance rate, latency
python3 tools/replay.py <session.jsonl> --all       # where it would have warned in one session
```

`benchmark.py` builds ground truth three ways, none of them manual:

- **Behavioural:** sessions you restarted within 30 min in the same project. You chose fresh context.
- **Synthetic:** one session's history plus another project's opening prompt.
- **Teacher:** a stronger model (GPT-4.1) labels real prompts.

It reports catch rates, precision, warnings per 100 prompts, agreement (κ), calibration
error, p50/p95 latency and cost, and writes a paste-ready `results/report.md`.

`stats.py` reports the number that matters most in real use: **of the warnings you saw,
how many you acted on.** Your `/clear` is the label.

## Privacy

- Transcripts are read locally. Only a small state (recent prompts, file names, a reply
  excerpt) is sent to the judge, and only after the local gates pass.
- Keys and tokens (`sk-…`, `apikey_…`, `AKIA…`, `ghp_…`) are redacted before anything is
  sent, logged or checkpointed.
- `DRIFT_GUARD_EXCLUDE_PROJECTS` keeps chosen projects fully on-device, live and in benchmarks.
- `data/` (judge caches built from your history) is gitignored.
