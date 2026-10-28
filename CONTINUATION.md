# Gameplay learning implementation — October 28, 2026

User objective: fix session leakage, mixed mouse/button training and the obsolete
evaluator, then implement learning from gameplay and human corrections.

Work is on `codex/gameplay-learning-loop`, based on `jose-main` commit `5dc1dd7`.
This checkout was created separately from the Asphalt workspace. No existing
gameplay budgets, stop guards or other projects were changed.

Implemented:
- Shared session-grouped legacy loader; temporal windows never cross chunks.
- Stable recording IDs and fixed action widths; preserve fractional gamepad
  inputs and discard failed mouse samples instead of changing label dimensions.
- Explicit binary/unit/signed/real action schema with mixed loss, decoding and
  separate metrics; checkpoint action width/schema survive save/load.
- Shared PyTorch offline evaluator replacing the mismatched Keras helper.
- Pickle-free hashed recording sessions for one game/task, human-correction
  provenance, replay of prior demonstrations, correction-weighted training,
  validation epoch selection and frozen test-set comparison with the incumbent.
- Persistent candidate/report history, atomic promotion, no-new-data no-op,
  exclusive training lock, cancellation/failure preservation and rejected models.
- CLI record/correct/shadow/play/cycle/status; optional one-cycle --update after
  a completed session. Live input is separate opt-in, finite and foreground/
  geometry/stop guarded. F8 transfers live control to human correction recording;
  F9/F12 stops; emergency F12 does not trigger automatic training.
- Legacy launcher training delegates to the corrected shared trainer. Bridge
  training now runs real work instead of reporting placeholder success.

Validation in isolated local Python 3.12 / CPU PyTorch environment:
- 250 tests passed across learning, mouse capture, quality, production readiness,
  Diablo diagnostics, training API and Tauri unit phases (140.93 seconds).
- Final focused rerun after collector cleanup: 70 passed (10.21 seconds).
- New modules/tests pass Ruff check; Python compilation and git diff --check pass.
- Synthetic training actually exercised promotion, correction replay, rejection,
  checkpoint loading and offline evaluation. No game was controlled in testing.
- Initial pytest runs hit a pre-existing Windows temp ACL problem, then a missing
  custom temp parent. A new project-local parent resolved these without changing
  permissions. Initial regression expectations for integer gamepad truncation and
  dropping failed mouse labels were updated to the corrected contracts.
- Existing legacy pickle warnings remain visible. General lint on older modified
  modules still reports pre-existing style debt; this is not a clean whole-repo
  lint/full-suite/package-build claim. Python 3.9–3.11 packaged builds were not run.

Usage: `docs/LEARNING_LOOP.md`. Dedicated correction workflow is currently CLI;
the launcher has no dedicated graphical correction buttons. Old 29/39-channel
recordings are not silently imported into the new profile-based action schema.
New data requires independent train/validation/test demonstrations before the
first cycle. Game performance, CUDA, and live capture/control remain unvalidated;
offline imitation gains are not win-rate claims or automatic reinforcement learning.

Next: use the documented recording/shadow workflow on the actual target game,
review mistakes, record corrections and compare bounded game trials. No live
run was authorized or executed as part of this implementation turn.
