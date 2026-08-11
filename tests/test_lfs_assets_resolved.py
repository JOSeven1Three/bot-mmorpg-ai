"""
Guard against unresolved Git LFS pointers for tracked binary assets.

.gitattributes marks *.gif/*.avi/*.exe as `filter=lfs`. If a checkout runs
without Git LFS installed, or `git lfs pull` was never run (missing
`lfs: true` on actions/checkout, a plain `git clone` on a dev machine, Git
LFS not installed at all), these paths on disk are ~130-byte pointer stub
text files ("version https://git-lfs.github.com/spec/v1 ...") instead of
the real binaries. That failure mode is silent everywhere else: `Test-Path`
in build_pipeline.ps1 and plain file-existence checks all see a file and
move on, so the shipped installer can end up bundling a non-functional
"driver" with no error until a user tries to use it on Windows.

These tests only assert something when the file exists on disk -- a
checkout that doesn't have the asset at all (e.g. a minimal sparse
checkout) is a different situation and not this test's job. What we're
guarding against is the specific case where the path exists but is a
stub, which is exactly what an unresolved LFS pointer looks like.
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"

# The handful of LFS-tracked assets that are load-bearing for a real
# Windows build/run: the driver installers bundled into the NSIS
# installer. (Demo gifs/avi under assets/ and frontend/ are also LFS
# but are cosmetic -- a pointer stub there doesn't break the app.)
CRITICAL_LFS_ASSETS = [
    "frontend/input_record/install-interception.exe",
    "src-tauri/drivers/interception/install-interception.exe",
    "versions/0.01/pyvjoy/vJoySetup.exe",
    "src-tauri/drivers/vjoy/vJoySetup.exe",
]


def _is_lfs_pointer_stub(path: pathlib.Path) -> bool:
    with path.open("rb") as fh:
        head = fh.read(len(LFS_POINTER_PREFIX))
    return head == LFS_POINTER_PREFIX


@pytest.mark.parametrize("relative_path", CRITICAL_LFS_ASSETS)
def test_critical_driver_asset_is_not_an_lfs_pointer_stub(relative_path: str) -> None:
    path = ROOT / relative_path
    if not path.exists():
        pytest.skip(f"{relative_path} not present in this checkout")
    assert not _is_lfs_pointer_stub(path), (
        f"{relative_path} is a Git LFS pointer stub, not the real binary. "
        "Run 'git lfs install' then 'git lfs pull' (or ensure CI checks out "
        "with lfs: true) before building the installer."
    )
