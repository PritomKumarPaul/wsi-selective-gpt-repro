#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${CONCHZ_MODEL_DIR:-${HOME:?HOME is not set}/models/conch-z}"
case "$model_dir" in
  "$repo_root"|"$repo_root"/*)
    echo "Refusing to download gated model weights inside the Git repository: $model_dir" >&2
    exit 2
    ;;
esac
if ! command -v hf >/dev/null 2>&1; then
  echo "Install huggingface_hub and run 'hf auth login' first." >&2
  exit 2
fi
mkdir -p "$model_dir"
hf download MahmoodLab/CONCH-Z \
  --revision 60211fced9c25a014494e248eedb6a6be80109f7 \
  --local-dir "$model_dir"
echo "CONCH-Z weights saved outside the repository at: $model_dir"
