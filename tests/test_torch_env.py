"""Tests for torch_env helpers — drive real shipped functions."""

from __future__ import annotations

import sys
from pathlib import Path

from src.torch_env import (
    describe_torch,
    resolve_torch_python,
    torch_available_here,
)


def test_resolve_torch_python_importable():
    py = resolve_torch_python()
    assert Path(py).exists()
    # Real subprocess import, not reimplemented
    import subprocess

    r = subprocess.run([py, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()


def test_describe_torch_reports_current_process():
    d = describe_torch()
    assert d["sys_executable"] == sys.executable
    assert "torch_available" in d
    if d["torch_available"]:
        assert "torch_version" in d
        assert "device" in d
