"""Locate a Python / torch runtime for GPU-optional scripts.

Default ``python3`` may lack torch; this module prefers an importable torch
and reports device (mps/cuda/cpu). Used by builders that need MMS/router decode.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


KNOWN_PYTHONS = (
    os.environ.get("TORCH_PYTHON", ""),
    sys.executable,
    shutil.which("python3") or "",
    "/opt/anaconda3/bin/python3",
    "/opt/anaconda3/envs/geoai/bin/python",
    "/opt/anaconda3/envs/geoai/bin/python3",
)


def _has_torch(python: str) -> bool:
    if not python or not Path(python).exists():
        return False
    try:
        r = subprocess.run(
            [python, "-c", "import torch"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_torch_python() -> str:
    """Return path to a Python interpreter that can ``import torch``."""
    seen: set[str] = set()
    for py in KNOWN_PYTHONS:
        if not py or py in seen:
            continue
        seen.add(py)
        if _has_torch(py):
            return py
    raise RuntimeError(
        "No Python with torch found. Install: "
        "/opt/anaconda3/bin/python3 -m pip install torch torchaudio "
        "or set TORCH_PYTHON to a torch-enabled interpreter."
    )


def torch_available_here() -> bool:
    """True if the current process can import torch."""
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def pick_torch_device(prefer_mps: bool = True):
    """Return a torch.device for inference (mps > cuda > cpu)."""
    import torch

    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_torch() -> dict:
    """Small diagnostic dict for logs/meta."""
    out: dict = {
        "sys_executable": sys.executable,
        "torch_available": torch_available_here(),
    }
    if out["torch_available"]:
        import torch

        out["torch_version"] = torch.__version__
        out["torch_file"] = torch.__file__
        out["cuda"] = torch.cuda.is_available()
        out["mps"] = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        out["device"] = str(pick_torch_device())
    else:
        try:
            out["resolved_python"] = resolve_torch_python()
        except RuntimeError as e:
            out["resolve_error"] = str(e)
    return out
