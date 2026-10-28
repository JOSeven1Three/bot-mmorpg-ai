# Learning from your gameplay and corrections

The learning loop records your actual keyboard, skill and mouse actions for one
game/task, trains a small local temporal policy, and evaluates each update against
the current model. It retains the old model if the candidate fails the promotion
gate. This is supervised imitation with human corrections, not reinforcement
learning or proof of better win rates. No paid API is required.

## Start from this checkout

Use the project's Python environment with PyTorch, NumPy, OpenCV, PyYAML, pywin32,
pynput and mss installed. From the repository folder in PowerShell:

```powershell
$env:PYTHONPATH = Join-Path $PWD 'src'
$python = '.\.venv\Scripts\python.exe'
& $python -m bot_mmorpg.scripts.learning --help
```

The commands below use Diablo IV's existing bindings. Change `--game`, `--task`
and the exact visible window title for another configured game. Capture uses
only the selected foreground client window. A focus or geometry change stops the
session and saves completed human samples. It never tries to refocus the game.
Profiles must explicitly list `input.default_bindings`; unsupported keys fail.

## 1. Record three independent demonstrations

Play the same narrowly defined task in different attempts. Record train,
validation and test sessions separately, before the first training cycle:

```powershell
& $python -m bot_mmorpg.scripts.learning --root learning/diablo-combat record --game diablo_4 --task combat --window-title 'Diablo IV' --seconds 120 --split train
& $python -m bot_mmorpg.scripts.learning --root learning/diablo-combat record --game diablo_4 --task combat --window-title 'Diablo IV' --seconds 120 --split validation
& $python -m bot_mmorpg.scripts.learning --root learning/diablo-combat record --game diablo_4 --task combat --window-title 'Diablo IV' --seconds 120 --split test
```

There is a three-second focus countdown. **F9 or F12 stops recording.** Q and T
remain game actions, so drinking a potion or using a portal does not stop the
recorder. Default capture is 10 FPS at 160x90 RGB, bounded to the requested
duration (maximum one hour). Samples flush every 300 frames and on clean exit.
Record several diverse training attempts, including recovery and idle examples.
Two minutes is a starting example, not an assurance of sufficient training data.

## 2. Train and compare

```powershell
& $python -m bot_mmorpg.scripts.learning --root learning/diablo-combat cycle --epochs 5
& $python -m bot_mmorpg.scripts.learning --root learning/diablo-combat status
```

The first candidate is compared with its untrained initialization. Later cycles
warm-start from the promoted model. Earlier demonstrations remain in the corpus;
corrections receive 2x sampling weight. A cycle with no new data does nothing.
Each invocation is finite; it does not install a scheduler or train indefinitely.
Use `--device cuda` only with a working CUDA PyTorch installation.

Validation chooses the best epoch. A separate frozen test set compares that
candidate with the incumbent. Promotion requires a score gain of at least 0.005,
no decrease in button macro F1 or exact-match accuracy, and no increase in any
continuous channel's MAE. The score is button macro F1 minus mean continuous MAE.
These metrics describe imitation. Repeated use of a benchmark can overfit model
selection; periodically start a fresh corpus/benchmark for an independent audit.

## 3. Supply corrections and update

```powershell
& $python -m bot_mmorpg.scripts.learning --root learning/diablo-combat shadow --game diablo_4 --task combat --window-title 'Diablo IV' --seconds 120
& $python -m bot_mmorpg.scripts.learning --root learning/diablo-combat correct --game diablo_4 --task combat --window-title 'Diablo IV' --seconds 120
& $python -m bot_mmorpg.scripts.learning --root learning/diablo-combat cycle --epochs 5
```

`shadow` and `correct` make predictions but send no inputs. You play, showing the
right action in situations the model handles badly. Both save your actions as
corrections with the active model's SHA-256. They cannot be used as held-out data.
Only genuine human-controlled frames enter the training corpus.

Add `--update` to `correct`, `shadow`, or `play` to run one training/comparison
cycle automatically after a normally completed recording session. It requires
the three initial splits and uses the default four-frame model configuration.
Focus/geometry failures stop the command without starting training.

For a separately chosen bounded live test, `play --live` uses the promoted model.
It is opt-in; `play` without `--live` remains shadow mode. In live mode **F8 releases
model-held keys and switches to human correction recording**; F8 again returns
control. **F9/F12 stops and releases keys.** Input checks foreground, original
geometry and stop keys immediately before dispatch, and is bounded to 20 ticks/s.
Only the profile's explicit bindings can be emitted. This learned policy has no
semantic purchase/menu detector: validate in shadow mode, use a narrow combat
profile and a suitable controlled scene before opting into live operation.

```powershell
# Optional explicit live test; never run by training or evaluation:
& $python -m bot_mmorpg.scripts.learning --root learning/diablo-combat play --game diablo_4 --task combat --window-title 'Diablo IV' --seconds 30 --live
```

## Evidence and compatibility

- `sessions/<id>/samples.npz`: pickle-free RGB frames, typed actions, timestamps.
- `session.json`: immutable identity, game/task, role, split and content hash.
- `runs/<id>/`: request, candidate checkpoint and evaluation or failure report.
- `registry.json`: active checkpoint hash and all accepted/rejected history.
- `training.lock`: exclusive cycle lock; on a process crash, first verify no
  trainer is running before manually removing a stale lock.

Original sessions cannot be changed or removed after a cycle. Validation/test
membership is frozen. Use a new root to intentionally change the benchmark.
Temporal windows never cross session or chunk boundaries. Training, prediction
and evaluation share RGB normalization and the same typed action schema.

The traditional trainer and the desktop launcher's legacy training entry point
now share a session-grouped loader and mixed binary/continuous loss. New legacy
recordings have session sidecars. Existing trusted `.npy` chunks without sidecars
are conservatively grouped by parent directory: provide at least two genuinely
independent recording folders. Do not relabel chunks from one recording as separate
sessions just to pass this check. Those older recordings are not automatically
imported into the new profile-based correction corpus, because their action
layout differs. The new learning CLI is the supported correction workflow.

The bridge training command runs a real learning cycle for a learning-root path
and reports rejection/failure honestly; it no longer emits placeholder success.
The existing launcher does not yet provide dedicated graphical correction buttons.

Offline evaluation uses PyTorch checkpoints and the shared recording loader:

```powershell
& $python scripts/evaluate_local_model.py --model learning/diablo-combat/runs/ACTUAL_RUN_ID/candidate.pth --dataset-dir learning/diablo-combat
```

No gameplay performance claim follows from passing offline tests. Compare task
completion, deaths, time and interventions in separate controlled game trials.
