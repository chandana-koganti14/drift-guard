"""Hook entry point for two Claude Code events.

UserPromptSubmit  gate -> judge -> policy -> (ask / pause / warn + checkpoint) -> event log.
                  mode=dialog: Claude asks you via its AskUserQuestion UI first.
                  mode=ask: the prompt is held (decision: block) until you /clear
                  or resend it. mode=warn: a systemMessage only.
SessionStart      on /clear or a New chat, resolve the last warning as accepted. That is
                  the label: your own behaviour, no manual review needed.

Fail-open everywhere: any error is logged and the prompt goes through untouched.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift_core as dg  # noqa: E402
import ledger  # noqa: E402

ACCEPT_WINDOW_S = 15 * 60  # a /clear this soon after a warning counts as accepting it


def on_prompt(event: dict, cfg: dict) -> dict | None:
    session_id = event.get("session_id", "unknown")
    cwd = event.get("cwd", "")
    prompt = event.get("prompt") or ""
    history, context_tokens = dg.parse_transcript(event["transcript_path"])
    turn = len(history)

    state = ledger.load_state()
    sess = state["sessions"].setdefault(session_id, {})
    sess["seen"] = time.time()

    # Resending a paused prompt means "continue here anyway": let it through.
    held = sess.pop("held_prompt", None)
    override = held is not None and held == prompt.strip()
    # Still typing in the same session after a warning means the warning was ignored.
    if pending := sess.pop("pending_warning", None):
        ledger.emit("outcome", warning_id=pending, accepted=False,
                    via="override" if override else "kept_prompting")

    if override:
        result = {"fire": False, "reason": "override"}
    elif turn - sess.get("warned_at_turn", -10**9) < cfg["cooldown_prompts"]:
        result = {"fire": False, "reason": "cooldown"}
    else:
        result = dg.decide(history, context_tokens, prompt, cfg, dg.pick_backend(cfg, cwd))
    ledger.emit("decision", session=session_id, cwd=cwd, turn=turn,
                prompt=dg.clean_prompt(prompt)[:160], **result)

    output = None
    if result["fire"]:
        warning_id = ledger.new_id()
        checkpoint = ledger.write_checkpoint(history, cwd, session_id)
        sess.update(warned_at_turn=turn, pending_warning=warning_id)
        state["last_warning"] = {"id": warning_id, "cwd": cwd, "ts": time.time(), "session": session_id}
        ledger.emit("warning", warning_id=warning_id, session=session_id, cwd=cwd,
                    checkpoint=str(checkpoint), confidence=result["verdict"]["confidence"],
                    judge=result["verdict"]["backend"])
        message = dg.warning_text(result, str(checkpoint), cfg["mode"])
        if cfg["mode"] == "dialog":
            # Claude shows its clickable question UI; the user decides before any answer is written.
            output = {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                             "additionalContext": dg.dialog_instruction(result, str(checkpoint))}}
        elif cfg["mode"] == "ask":
            # Blocked prompts never reach Claude; the user sees the reason and their prompt to copy.
            sess["held_prompt"] = prompt.strip()
            output = {"decision": "block", "reason": message}
        else:
            output = {"systemMessage": message}
    ledger.save_state(state)
    return output


def on_session_start(event: dict, cfg: dict) -> dict | None:
    # /clear in the terminal, or "New chat" in an IDE (which starts with source=startup)
    if event.get("source") not in ("clear", "startup"):
        return None
    cwd = event.get("cwd", "")
    ledger.emit("clear", session=event.get("session_id"), cwd=cwd, source=event.get("source"))
    state = ledger.load_state()
    last = state.get("last_warning")
    if (not last or last.get("resolved")
            or ledger.project_name(last.get("cwd", "")) != ledger.project_name(cwd)
            or time.time() - last["ts"] > ACCEPT_WINDOW_S):
        return None
    last["resolved"] = True
    state["sessions"].get(last["session"], {}).pop("pending_warning", None)
    ledger.save_state(state)
    ledger.emit("outcome", warning_id=last["id"], accepted=True, via=event.get("source"))
    return None


HANDLERS = {"UserPromptSubmit": on_prompt, "SessionStart": on_session_start}


def main() -> None:
    event = json.load(sys.stdin)
    handler = HANDLERS.get(event.get("hook_event_name", ""))
    if handler and (output := handler(event, dg.load_config())):
        print(json.dumps(output))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # a broken guard must never break the user's prompt
        try:
            ledger.emit("error", error=repr(exc))
        except OSError:
            pass
    sys.exit(0)
