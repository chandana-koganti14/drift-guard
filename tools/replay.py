"""Replay past Claude Code sessions through Drift Guard.

    python3 tools/replay.py <transcript.jsonl> [--judge heuristic|adapter|jev] [--all]

Prints every prompt with the judge's verdict, and marks where a warning would
have fired. --all ignores the context-size gate so every prompt is judged.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/drift-guard/scripts"))
import drift_core as dg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--all", action="store_true", help="judge every prompt, ignoring the context gate")
    args = ap.parse_args()

    cfg = dg.load_config()
    if args.all:
        cfg["min_context_tokens"] = 0
        cfg["min_prior_prompts"] = 1
    backend = args.judge or dg.pick_backend(cfg)
    turns, _ = dg.parse_transcript(args.transcript)
    print(f"judge={backend}  prompts={len(turns)}\n")

    for i, turn in enumerate(turns):
        result = dg.decide(turns[:i], turn.context_tokens_before, turn.prompt, cfg, backend)
        v = result.get("verdict")
        mark = "🧭 WARN" if result["fire"] else "  quiet"
        detail = (f"{v['label']:<16} conf={v['confidence']:.2f}  new_topic_p={v['probabilities'].get('new_topic', 0):.2f}"
                  if v else result["reason"])
        ctx = f"{turn.context_tokens_before // 1000:>4}k"
        print(f"{mark} #{i:<3} ctx={ctx}  {detail}")
        print(f"         > {turn.prompt[:110].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
