"""Install Drift Guard's hooks into Claude Code settings. Works in the terminal
CLI and in the VS Code / JetBrains extensions (no /plugin command needed).

    python3 tools/install.py ~/code/my-project     # this project only (recommended)
    python3 tools/install.py --user                # every project
    python3 tools/install.py ~/code/my-project --uninstall

Merges into existing settings (other hooks are kept) and is idempotent:
running it twice changes nothing. Project installs go to
<project>/.claude/settings.local.json, which Claude Code treats as personal.
"""

import argparse
import json
import sys
from pathlib import Path

RUN_SH = Path(__file__).resolve().parents[1] / "plugins/drift-guard/scripts/run.sh"
COMMAND = f'sh "{RUN_SH}"'
MARK = "drift-guard"  # identifies our entries for updates and uninstall
EVENTS = {"UserPromptSubmit": None, "SessionStart": "clear|startup"}


def strip_ours(groups: list) -> list:
    kept = []
    for group in groups:
        hooks = [h for h in group.get("hooks", []) if MARK not in h.get("command", "")]
        if hooks:
            kept.append({**group, "hooks": hooks})
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project", nargs="?", help="project folder to install into")
    ap.add_argument("--user", action="store_true", help="install for every project (~/.claude/settings.json)")
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()
    if bool(args.project) == args.user:
        sys.exit("Give a project folder, or --user (not both).")

    target = (Path.home() / ".claude/settings.json" if args.user
              else Path(args.project).expanduser().resolve() / ".claude/settings.local.json")
    settings = json.loads(target.read_text()) if target.exists() else {}
    hooks = settings.setdefault("hooks", {})
    for event, matcher in EVENTS.items():
        groups = strip_ours(hooks.get(event, []))
        if not args.uninstall:
            group = {"hooks": [{"type": "command", "command": COMMAND, "timeout": 10}]}
            if matcher:
                group = {"matcher": matcher, **group}
            groups.append(group)
        if groups:
            hooks[event] = groups
        else:
            hooks.pop(event, None)
    if not hooks:
        settings.pop("hooks")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"{'Removed from' if args.uninstall else 'Installed into'} {target}")
    if not args.uninstall:
        print("Takes effect in new sessions. In an open session, run /hooks once to reload.")


if __name__ == "__main__":
    main()
