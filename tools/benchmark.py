"""Label-free benchmark: heuristic vs Jev vs an LLM, on your own sessions.

    ~/.drift-guard/venv/bin/python tools/benchmark.py
    ~/.drift-guard/venv/bin/python tools/benchmark.py --judges heuristic jev --limit 150

Ground truth without a human:
  behavioral  your own fresh starts: catch rate on real switches
  synthetic   spliced switches: catch rate on clean switches
  natural     real prompts, labelled by a stronger teacher model: precision,
              false-alarm rate, agreement (kappa) and calibration (ECE)
Writes results/report.md (paste-ready) and results/results.json.
Answers are cached in data/, so reruns only judge what is new.
"""

import argparse
import json
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plugins/drift-guard/scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset  # noqa: E402
import drift_core as dg  # noqa: E402

DATA, RESULTS = ROOT / "data", ROOT / "results"
JEV_USD_PER_MTOK = 0.042  # docs.typesafe.ai/models
SPECS = {  # name -> (backend, config overrides)
    "heuristic": ("heuristic", {}),
    "jev": ("jev", {}),
    "gpt-4.1-mini": ("adapter", {"adapter_model": "gpt-4.1-mini"}),
}


def judge_all(name: str, backend: str, overrides: dict, examples: list[dict], cfg: dict,
              workers: int = 8) -> dict[str, dict]:
    """Judge every example once (cached, parallel). Failures are retried on the next run."""
    DATA.mkdir(exist_ok=True)
    cache_path = DATA / f"cache_{name}.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {r["id"]: r for r in map(json.loads, cache_path.read_text().splitlines())}
    todo = [e for e in examples if e["id"] not in cache]
    jcfg = {**cfg, **overrides}
    failures: dict[str, int] = {}
    lock = threading.Lock()
    with open(cache_path, "a") as out, ThreadPoolExecutor(max_workers=1 if backend == "heuristic" else workers) as pool:
        futures = {pool.submit(dg.run_judge, backend, e["state"], jcfg, None, False): e for e in todo}
        for n, future in enumerate(as_completed(futures), 1):
            e, v = futures[future], future.result()
            if not v.backend.startswith(backend):  # fell back: don't cache, retry next run
                with lock:
                    failures[v.backend] = failures.get(v.backend, 0) + 1
                continue
            cache[e["id"]] = {"id": e["id"], **v.__dict__}
            out.write(json.dumps(cache[e["id"]]) + "\n")
            print(f"\r  {name}: {n}/{len(todo)}", end="", file=sys.stderr)
    if todo:
        print(file=sys.stderr)
    for reason, count in failures.items():
        print(f"  ⚠️  {name}: {count} failed ({reason}); rerun to retry", file=sys.stderr)
    return cache


def warns(v: dict, t: float) -> bool:
    return v["label"] == "new_topic" and v["confidence"] >= t


def catch_rate(examples: list[dict], verdicts: dict, t: float) -> float | None:
    scored = [warns(verdicts[e["id"]], t) for e in examples if e["id"] in verdicts]
    return sum(scored) / len(scored) if scored else None


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    labels = dg.LABELS
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum((sum(a == k for a, _ in pairs) / n) * (sum(b == k for _, b in pairs) / n) for k in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else None


def ece(points: list[tuple[float, bool]], bins: int = 5) -> float | None:
    """Expected calibration error of P(new_topic): 0 = perfectly calibrated."""
    if not points:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [(p, y) for p, y in points if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if members:
            gap = abs(statistics.mean(p for p, _ in members) - statistics.mean(y for _, y in members))
            total += gap * len(members) / len(points)
    return total


def silver_labels(natural: list[dict], teacher: dict, min_conf: float = 0.6) -> dict[str, str]:
    return {e["id"]: teacher[e["id"]]["label"] for e in natural
            if e["id"] in teacher and teacher[e["id"]]["confidence"] >= min_conf}


def score(name: str, verdicts: dict, sets: dict, silver: dict, t: float) -> dict:
    nat = [e for e in sets["natural"] if e["id"] in verdicts]
    labelled = [e for e in nat if e["id"] in silver]
    fired = [e for e in labelled if warns(verdicts[e["id"]], t)]
    switches = [e for e in labelled if silver[e["id"]] == "new_topic"]
    all_ex = [e for s in sets.values() for e in s if e["id"] in verdicts]
    latencies = sorted(verdicts[e["id"]]["latency_ms"] for e in all_ex)
    tokens = statistics.mean(len(json.dumps(e["state"])) / 4 for e in all_ex) if all_ex else 0
    return {
        "judge": name,
        "behavioral_catch": catch_rate(sets["behavioral"], verdicts, t),
        "synthetic_catch": catch_rate(sets["synthetic"], verdicts, t),
        "precision": sum(silver[e["id"]] == "new_topic" for e in fired) / len(fired) if fired else None,
        "recall": sum(warns(verdicts[e["id"]], t) for e in switches) / len(switches) if switches else None,
        "warn_per_100": 100 * sum(warns(verdicts[e["id"]], t) for e in nat) / len(nat) if nat else None,
        "kappa": kappa([(verdicts[e["id"]]["label"], silver[e["id"]]) for e in labelled]),
        "ece": ece([(verdicts[e["id"]]["probabilities"].get("new_topic", 0.0), silver[e["id"]] == "new_topic")
                    for e in labelled]),
        "p50_ms": latencies[len(latencies) // 2] if latencies else None,
        "p95_ms": latencies[int(len(latencies) * 0.95) - 1] if latencies else None,
        "usd_per_1k": tokens * 1000 * JEV_USD_PER_MTOK / 1e6 if name == "jev" else None,
    }


def fmt(value, kind: str) -> str:
    if value is None:
        return "–"
    return {"pct": f"{value:.0%}", "num": f"{value:.2f}", "ms": f"{value:,} ms",
            "usd": f"${value:.4f}", "rate": f"{value:.1f}"}[kind]


SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9)


def sweep_table(sweeps: dict[str, list[dict]]) -> list[str]:
    """Same metrics at each threshold, so the cut-off is chosen from data, not guessed."""
    lines = ["", "### Choosing the threshold", "",
             "| Judge | Threshold | Precision | Recall | Catch: synthetic | Warnings / 100 prompts |",
             "|---|---|---|---|---|---|"]
    for name, rows in sweeps.items():
        for t, r in zip(SWEEP, rows):
            lines.append(f"| {name} | {t} | {fmt(r['precision'], 'pct')} | {fmt(r['recall'], 'pct')} "
                         f"| {fmt(r['synthetic_catch'], 'pct')} | {fmt(r['warn_per_100'], 'rate')} |")
    return lines


def report(rows: list[dict], sets: dict, silver: dict, t: float, teacher: str,
           sweeps: dict[str, list[dict]] | None = None) -> str:
    positives = sum(1 for v in silver.values() if v == "new_topic")
    lines = [
        "# Drift Guard benchmark",
        "",
        f"Warn when the judge says `new_topic` with confidence ≥ {t}. "
        f"Data: {len(sets['natural'])} real prompts ({len(silver)} teacher-labelled, {positives} switches), "
        f"{len(sets['behavioral'])} behavioural switches, {len(sets['synthetic'])} synthetic switches.",
        "",
        "| Judge | Catch: real fresh starts | Catch: synthetic | Precision | Recall | Warnings / 100 prompts "
        "| Agreement (κ) | Calibration error | p50 | p95 | Cost / 1k |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| **{r['judge']}** | {fmt(r['behavioral_catch'], 'pct')} | {fmt(r['synthetic_catch'], 'pct')} "
            f"| {fmt(r['precision'], 'pct')} | {fmt(r['recall'], 'pct')} | {fmt(r['warn_per_100'], 'rate')} "
            f"| {fmt(r['kappa'], 'num')} | {fmt(r['ece'], 'num')} | {fmt(r['p50_ms'], 'ms')} "
            f"| {fmt(r['p95_ms'], 'ms')} | {fmt(r['usd_per_1k'], 'usd')} |")
    lines += [
        "",
        "**How to read it.** Precision, recall, κ and calibration error are measured against "
        f"silver labels from `{teacher}` (examples where it was ≥ 0.6 confident), so they show agreement "
        "with a stronger model, not absolute truth. The catch rates need no model: real fresh starts come "
        f"from sessions you started ≤ {dataset.MAX_GAP_MIN} min after the previous one in the same project "
        "(occasionally a restart of the same task), synthetic ones splice an unrelated project's opening prompt. "
        "Calibration error 0 = when it says 80%, it's right 80% of the time.",
    ]
    if sweeps:
        lines += sweep_table(sweeps)
    if positives < 15:
        lines += ["", f"⚠️ Only {positives} teacher-labelled switches: precision and recall are rough."]
    return "\n".join(lines) + "\n"


def main() -> None:
    cfg = dg.load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=list(SPECS), choices=list(SPECS))
    ap.add_argument("--teacher", default=cfg["teacher_model"])
    ap.add_argument("--limit", type=int, default=300, help="max real prompts to judge")
    ap.add_argument("--synthetic", type=int, default=60)
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="projects to leave out (default: DRIFT_GUARD_EXCLUDE_PROJECTS)")
    ap.add_argument("--threshold", type=float, default=cfg["min_confidence"])
    args = ap.parse_args()

    exclude = args.exclude if args.exclude is not None else cfg["exclude_projects"].split(",")
    sessions = dataset.load_sessions([p for p in exclude if p.strip()])
    sets = {"natural": dataset.natural(sessions, args.limit),
            "behavioral": dataset.behavioral(sessions),
            "synthetic": dataset.synthetic(sessions, args.synthetic)}
    print(f"{len(sessions)} sessions → {', '.join(f'{len(v)} {k}' for k, v in sets.items())}", file=sys.stderr)
    if not sets["natural"]:
        sys.exit("No usable sessions found in ~/.claude/projects.")

    teacher = judge_all(f"teacher-{args.teacher}", "adapter", {"adapter_model": args.teacher},
                        sets["natural"], cfg)
    silver = silver_labels(sets["natural"], teacher)
    examples = [e for s in sets.values() for e in s]
    verdicts = {name: judge_all(name, *SPECS[name], examples, cfg) for name in args.judges}
    rows = [score(name, v, sets, silver, args.threshold) for name, v in verdicts.items()]
    sweeps = {name: [score(name, v, sets, silver, t) for t in SWEEP] for name, v in verdicts.items()}

    RESULTS.mkdir(exist_ok=True)
    md = report(rows, sets, silver, args.threshold, args.teacher, sweeps)
    (RESULTS / "report.md").write_text(md)
    (RESULTS / "results.json").write_text(json.dumps(
        {"threshold": args.threshold, "teacher": args.teacher, "rows": rows, "sweeps": sweeps,
         "counts": {k: len(v) for k, v in sets.items()}, "silver_labelled": len(silver)}, indent=2))
    print("\n" + md)
    print(f"Saved {RESULTS / 'report.md'} and results.json", file=sys.stderr)


if __name__ == "__main__":
    main()
