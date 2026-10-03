"""Append-only event log plus the small mutable state the hooks share.

events.jsonl is the source of truth for everything Drift Guard did and what
the user did next, so stats and accuracy are computed from facts, not labels:

  decision  every judged prompt (label, confidence, judge, latency, fallback)
  warning   a warning was shown (with its checkpoint)
  clear     the user ran /clear
  outcome   a warning resolved: accepted (user cleared) or ignored (kept going)
"""

import json
import time
import uuid
from pathlib import Path

from drift_core import HOME, redact

EVENTS = HOME / "events.jsonl"
STATE = HOME / "state.json"
SESSION_TTL_S = 7 * 24 * 3600


def emit(kind: str, **fields) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    with open(EVENTS, "a") as f:
        f.write(json.dumps({"ts": time.time(), "kind": kind, **fields}) + "\n")


def read_events() -> list[dict]:
    if not EVENTS.exists():
        return []
    events = []
    for line in EVENTS.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def load_state() -> dict:
    try:
        state = json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    state.setdefault("sessions", {})
    cutoff = time.time() - SESSION_TTL_S
    state["sessions"] = {k: v for k, v in state["sessions"].items() if v.get("seen", 0) > cutoff}
    return state


def save_state(state: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE)  # atomic: a crash never leaves half-written state


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def project_name(cwd: str) -> str:
    """Name of the project root (nearest folder with .git or .claude), not the subfolder."""
    path = Path(cwd) if cwd else Path.cwd()
    for folder in (path, *path.parents):
        if folder != Path.home() and ((folder / ".git").exists() or (folder / ".claude").is_dir()):
            return folder.name
    return path.name or "project"


def write_checkpoint(history, cwd: str, session_id: str) -> Path:
    """Save the old task so clearing never means losing your place."""
    project = project_name(cwd)
    folder = HOME / "checkpoints" / project
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}-{session_id[:8]}.md"
    files: list[str] = []
    for turn in history:
        for f in turn.files:
            if f not in files:
                files.append(f)
    lines = [
        f"# Checkpoint: {project}",
        f"Saved {time.strftime('%Y-%m-%d %H:%M')} from session `{session_id}` before a topic switch.",
        "",
        "## What you were working on",
        *[f"- {t.prompt[:300].replace(chr(10), ' ')}" for t in history[-8:]],
        "",
        "## Files touched",
        *([f"- `{f}`" for f in files[-25:]] or ["- (none)"]),
        "",
        "## Last reply (excerpt)",
        "",
        redact(history[-1].reply[:1500] if history else "") or "(none)",
        "",
        "## Resume later",
        "",
        f"In a fresh session, say: `continue the task described in {path}`",
    ]
    path.write_text("\n".join(lines) + "\n")
    return path
