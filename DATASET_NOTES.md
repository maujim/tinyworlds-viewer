# TinyWorlds Dataset Notes

Local cache inspected at:

```text
~/.cache/huggingface/hub/datasets--AlmondGod--tinyworlds/
```

Current snapshot:

```text
snapshots/aa62b3ca4349c8ed23894ccb33ec6ff45fdcb3b4/
```

## Files and shapes

Each file is HDF5 and contains a single dataset named `frames` with unsigned 8-bit RGB image data.

| File | Dataset | Shape | Dtype |
| --- | --- | ---: | --- |
| `picodoom_frames.h5` | `frames` | `(59785, 64, 64, 3)` | `uint8` |
| `pole_position_frames.h5` | `frames` | `(5385, 64, 64, 3)` | `uint8` |
| `pong_frames.h5` | `frames` | `(108034, 64, 64, 3)` | `uint8` |
| `sonic_frames.h5` | `frames` | `(41242, 64, 64, 3)` | `uint8` |
| `zelda_frames.h5` | `frames` | `(72410, 128, 128, 3)` | `uint8` |

## Loader behavior

`tinyworlds_viewer.dataset.TinyWorldsDataset` discovers the latest local Hugging Face snapshot containing `*_frames.h5` files. It exposes:

- `files`: metadata for each world.
- `read_frame(world, index)`: returns one RGB `numpy.uint8` frame.
- `read_frames(world, indices)`: convenience method for multiple frames.

Use `TINYWORLDS_DATASET_DIR` to point at another snapshot or copy.
