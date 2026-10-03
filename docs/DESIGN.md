# Drift Guard: design notes

A hook that runs before every prompt in someone's editor has a strict contract: it must
never make the tool slower, flakier or leakier. Every decision below follows from that.

## 1. Narrow model, code in control

The model makes one typed decision: a `Choice` over `continuation | related_subtask |
new_topic`. Thresholds, cooldowns, wording, checkpoints and logging are ordinary code.
Changing policy never means re-prompting a model, and every behaviour is unit-testable.

Why Jev: the decision runs inline on a human's prompt, so latency and cost matter more
than eloquence. Jev answers in ~250 ms for about $0.004 per 1,000 decisions, and its
probabilities are trained to be calibrated, so a fixed threshold (0.8) means the same
thing across sessions.

## 2. Gates before the model (cheap → expensive)

```
slash command? → enough history? → enough context? → cooldown? → judge
```

Most prompts stop at a local check in about a millisecond. The judge only runs when a
warning could actually help, which keeps cost and data exposure proportional to value.

**Considered:** a heuristic-first cascade that skips Jev when the local judge is confident.
**Rejected for now:** Jev is cheap and fast enough that the extra recall risk isn't worth
the savings. `benchmark.py` will show if that changes.

## 3. Fail open, with a latency budget and a circuit breaker

- **Budget:** the remote judge runs in a worker with a 1.5 s deadline. On timeout the
  local heuristic answers, and the event log records `fallback: timeout`.
- **Circuit breaker:** after 3 consecutive remote failures the breaker opens for
  10 minutes, so a dead API costs 0 ms per prompt instead of a timeout every time.
  State lives in `breaker.json`.
- **Fail open:** any exception is logged and the hook exits 0. A broken guard must
  never break the user's prompt.

## 4. Behaviour is the label

Asking users to grade warnings doesn't scale, and asking the builder to doesn't either.
Ground truth comes from what people do:

- **Live:** a `SessionStart` hook with `source: "clear"` resolves the last warning as
  *accepted*. Typing on in the same session resolves it as *ignored*. Acceptance rate is
  real-world precision.
- **Offline:** sessions restarted within 30 minutes in the same project are real topic
  switches the user chose. Synthetic splices and a stronger teacher model fill in the rest.

Known biases, stated in every report: a fresh start is sometimes a restart of the same
task; synthetic switches are cleaner than real ones; teacher agreement is not truth.

## 5. Append-only event log as the source of truth

`events.jsonl` records `decision`, `warning`, `clear`, `outcome` and `error` events. Stats
are derived from it, never stored separately, so any metric can be recomputed or redefined
later. Mutable state (cooldowns, the pending warning) is small, pruned after 7 days, and
written atomically (temp file + rename) so a crash can't corrupt it.

## 6. Privacy by default

- Secrets are redacted at parse time, before state reaches a judge, a log or a checkpoint.
- Only a bounded state is sent: 4 recent prompts, up to 15 file names, an 800-character
  reply excerpt.
- Excluded projects never use a remote judge, live or in benchmarks.
- Keys live in `~/.drift-guard/config.env` (mode 600), never in the repo.

## 7. Make the recommended action cheap

A warning that asks you to throw away context gets ignored. So when it fires, Drift Guard
writes a checkpoint first (what you were doing, files touched, the last reply) and puts
the path in the warning. `/clear` stops being a loss.

## 8. Idempotent installs

`tools/install.py` merges into existing settings, tags its own entries, and can be re-run
or reversed without touching anyone else's hooks.
