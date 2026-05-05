# Contestant Task: Build a TinyWorlds Dataset Visualizer

You are given a working scaffold. Your goal is to turn it into a compelling visualizer for the TinyWorlds frame dataset.

## What is already provided

- `uv` Python project setup.
- Streamlit entrypoint: `app.py`.
- Dataset helpers in `src/tinyworlds_viewer/dataset.py`.
- Smoke tests and linting via `./test.sh`.

## Your objective

Create a viewer that helps a user understand and explore the TinyWorlds dataset. Prioritize creativity, vibes, and functionality. The dataset contains frame sequences from multiple tiny game worlds.

Possible directions:

- Better navigation through huge frame sequences.
- Contact sheets / grids / filmstrips.
- Animation playback controls.
- World comparison views.
- Color/statistics summaries.
- Interesting sampling strategies.
- Bookmarking/exporting frames.
- A polished visual design.

## Constraints

- Keep setup simple: `uv sync --dev`, then run the app.
- Do not require moving the dataset.
- Respect `TINYWORLDS_DATASET_DIR` for custom dataset locations.
- Keep `./test.sh` passing, and add tests if you add reusable logic.
- Avoid committing generated caches, videos, or large artifacts.

## Success criteria

A reviewer should be able to copy this directory, run `./test.sh`, run the app, and quickly see that your viewer provides a useful and enjoyable way to inspect TinyWorlds.
