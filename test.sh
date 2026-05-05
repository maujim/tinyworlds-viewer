#!/usr/bin/env bash
set -euo pipefail

uv sync --dev
uv run ruff check .
uv run pytest
uv run python - <<'PY'
from tinyworlds_viewer import TinyWorldsDataset

ds = TinyWorldsDataset()
print(f"Found {len(ds.files)} TinyWorlds files at {ds.root}")
for f in ds.files:
    print(f"- {f.world}: {f.frame_count} frames, {f.height}x{f.width}x{f.channels}")
PY

echo "OK: scaffold validates. Run: uv run streamlit run app.py"
