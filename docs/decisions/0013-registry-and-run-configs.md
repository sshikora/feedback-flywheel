# 0013: Model registry and run configs

**Status:** Accepted, 2026-11-04

**Context.** M3 registers each trained model with its dataset hash, cost and a model card. The gate in PLAN.md section 8 needs thresholds fixed before training, and the loop in `0008` needs to know which version a new one must beat.

**Decision.**

- **A run config states every setting.** It has no defaults, including for the gate thresholds, so a run cannot quietly inherit a value that changed later. The judge threshold must be above 50%.
- **The registry stores the run config's hash.** A model card is refused if the file has changed since the model was registered, so a reader can trust that the card shows the settings the model was trained and gated with.
- **A run config names a built snapshot.** Registration is refused unless a snapshot with that hash exists, so every model traces back to exactly what it learned from.
- **Status moves one way.** A model starts as a candidate. It can become a challenger, then a champion, and any model can be retired. Nothing returns to an earlier status, so a retired version comes back only as a new run.
- **A failed gate is a hold.** The model stays a candidate, and the decision is written to the history with its evidence. Every promotion and every rejection is in the history.
- **One model is live.** Challenger and champion are the live statuses. When a new version passes the gate and goes live as challenger, the version it replaces is retired in the same transaction, with the reason recorded. `live()` returns the version the next candidate must beat, or nothing before the first one passes.
- **SQLite, in `data/registry/`.** Records and history are stored as JSON validated by the same pydantic models that produce the JSON Schema, so the future TypeScript client reads the same shape.

**Not built yet.** Training through Together, which will register models itself, the gate that calls `transition` and `hold`, and the public run history page generated from the registry.

**Consequences.** `flywheel registry add` registers a model trained outside the pipeline, which covers runs started by hand. Changing a run config after training is allowed, but that model's card then fails until the change is reverted, so a new setting means a new run with a new name.
