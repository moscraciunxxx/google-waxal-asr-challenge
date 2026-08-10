#!/usr/bin/env bash
# Resolve a Python that has torch importable; prefer PATH, then known envs.
# Usage:
#   ./scripts/with_torch_python.sh -c "import torch; print(torch.__version__)"
#   ./scripts/with_torch_python.sh scripts/luo_acoustic_router.py --max-files 4
set -euo pipefail

CANDIDATES=(
  "${TORCH_PYTHON:-}"
  "$(command -v python3 || true)"
  /opt/anaconda3/bin/python3
  /opt/anaconda3/envs/geoai/bin/python
  /opt/anaconda3/envs/geoai/bin/python3
)

pick=""
for py in "${CANDIDATES[@]}"; do
  [[ -z "$py" || ! -x "$py" ]] && continue
  if "$py" -c "import torch" 2>/dev/null; then
    pick="$py"
    break
  fi
done

if [[ -z "$pick" ]]; then
  echo "ERROR: no Python with torch found. Install with:" >&2
  echo "  /opt/anaconda3/bin/python3 -m pip install torch torchaudio" >&2
  exit 1
fi

if [[ "${WITH_TORCH_VERBOSE:-0}" == "1" ]]; then
  echo "using $pick ($("$pick" -c 'import torch; print(torch.__version__)'))" >&2
fi

exec "$pick" "$@"
