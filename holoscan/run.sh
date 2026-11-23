#!/usr/bin/env bash
# Launch the sMRI → Holoscan graph.
# Author: Mohammad H. Abbasi — Stanford STAI Lab
#
# Usage:
#   ./run.sh --input /path/to/sub_t1w.nii.gz --subject SUB01 --modality T1w --outdir ./out
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Holoscan recommends a 32 MB stack to avoid segfaults.
ulimit -s 32768 || true

# Keep the TemplateFlow MNI cache local to the project.
export TEMPLATEFLOW_HOME="${TEMPLATEFLOW_HOME:-$HERE/.templateflow}"

# If your system CUDA runtime is older than 12.5, uncomment and adjust:
# export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}

PY="$HERE/.venv/bin/python"
[ -x "$PY" ] || PY="python3"

exec "$PY" "$HERE/smri_holoscan_app.py" "$@"
