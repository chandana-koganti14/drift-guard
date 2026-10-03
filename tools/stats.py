"""Live stats from the event log: how Drift Guard behaves in real use.

    python3 tools/stats.py [--days 7]

Acceptance rate is the real-world precision: of the warnings you saw, how many
you acted on with /clear. It needs no labelling; your behaviour is the label.
"""

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins/drift-guard/scripts"))
import ledger  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=7)
    args = ap.parse_args()
    since = time.time() - args.days * 86400
    events = [e for e in ledger.read_events() if e["ts"] >= since]
    if not events:
        sys.exit(f"No events in the last {args.days:g} days. Is the hook installed? (python3 tools/install.py)")

    decisions = [e for e in events if e["kind"] == "decision"]
    judged = [e for e in decisions if e.get("verdict")]
    warnings = [e for e in events if e["kind"] == "warning"]
    outcomes = {e["warning_id"]: e for e in events if e["kind"] == "outcome"}
    accepted = sum(1 for w in warnings if outcomes.get(w["warning_id"], {}).get("accepted") is True)
    ignored = sum(1 for w in warnings if outcomes.get(w["warning_id"], {}).get("accepted") is False)
    gates = Counter(e["reason"] for e in decisions if not e.get("verdict"))
    judges = Counter(e["verdict"]["backend"] for e in judged)
    latency = sorted(e["verdict"]["latency_ms"] for e in judged)
    errors = sum(1 for e in events if e["kind"] == "error")

    print(f"Last {args.days:g} days")
    print(f"  prompts seen        {len(decisions)}")
    print(f"  stopped at gates    {sum(gates.values())}  {dict(gates)}")
    print(f"  judged              {len(judged)}")
    print(f"  warnings shown      {len(warnings)}  ({100 * len(warnings) / max(len(decisions), 1):.1f} per 100 prompts)")
    resolved = accepted + ignored
    rate = f"{accepted / resolved:.0%}" if resolved else "–"
    print(f"  accepted (/clear)   {accepted}   ignored {ignored}   pending {len(warnings) - resolved}")
    print(f"  acceptance rate     {rate}   ← real-world precision; aim for 70%+")
    if latency:
        print(f"  judge latency       p50 {latency[len(latency) // 2]} ms   "
              f"p95 {latency[max(int(len(latency) * 0.95) - 1, 0)]} ms   mean {statistics.mean(latency):.0f} ms")
    for name, count in judges.most_common():
        print(f"  judge               {name}: {count}")
    if errors:
        print(f"  hook errors         {errors}  (see {ledger.EVENTS})")


if __name__ == "__main__":
    main()
