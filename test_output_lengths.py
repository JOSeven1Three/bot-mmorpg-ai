import numpy as np
from versions.old.getgamepad import gamepad_check
from versions.old.getkeys import key_check

# Test to see what lengths we get
for i in range(5):
    keys = key_check()
    gamepad = gamepad_check()
    print(f"Keys: {keys}, len={len(keys)}")
    print(f"Gamepad: {gamepad}, len={len(gamepad)}")
    print()
