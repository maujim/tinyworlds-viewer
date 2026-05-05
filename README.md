# TinyWorlds Viewer Scaffold

A minimal, copyable Streamlit + `uv` scaffold for building a visualizer for the local `AlmondGod/tinyworlds` dataset.

## Quickstart

```bash
cd scaffold
uv sync --dev
uv run streamlit run app.py
```

Validate everything with one command:

```bash
./test.sh
```

Create a fresh model attempt directory from the scaffold:

```bash
./attempt.sh /path/to/attempt-a
```

The destination is created with `mkdir -p`, must be empty, is copied without ignored files, and then runs `uv sync --dev` there.

## Dataset location

By default, the loader searches:

```text
~/.cache/huggingface/hub/datasets--AlmondGod--tinyworlds/snapshots/*/
```

Override it if needed:

```bash
export TINYWORLDS_DATASET_DIR=/path/to/snapshot
uv run streamlit run app.py
```

## Project layout

```text
app.py                         # intentionally light Streamlit placeholder
src/tinyworlds_viewer/dataset.py # HDF5 discovery + frame loading helpers
tests/                         # smoke tests
test.sh                        # uv sync, lint, tests, local dataset probe
CONTESTANT_TASK.md             # prompt/spec for models being evaluated
DATASET_NOTES.md               # discovered dataset schema notes
```

The scaffold intentionally provides boilerplate and a tiny working viewer, not a finished product. Contestants should focus on creative visualization, interaction, usability, and insight.
