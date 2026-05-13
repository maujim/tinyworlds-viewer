from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from tinyworlds_viewer import TinyWorldsDataset
from tinyworlds_viewer.visuals import contact_sheet, frame_delta, frame_stats, sample_indices

EXPORT_DIR = Path("tinyworlds_contact_sheets").resolve()

st.set_page_config(page_title="TinyWorlds Explorer", page_icon="🕹️", layout="wide")
st.markdown(
    """
    <style>
    :root { color-scheme: light dark; }
    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(255, 203, 107, .28), transparent 24rem),
            radial-gradient(circle at 88% 8%, rgba(125, 211, 252, .24), transparent 26rem),
            var(--background-color);
    }
    h1, h2, h3 { letter-spacing: -0.035em; }
    [data-testid="stMetric"] {
        background: color-mix(in srgb, var(--background-color) 88%, #7dd3fc 12%);
        border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
        border-radius: 16px;
        padding: 12px;
    }
    .tw-hero {
        border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
        border-radius: 24px;
        padding: 1.1rem 1.25rem;
        background: color-mix(in srgb, var(--background-color) 84%, #fef3c7 16%);
    }
    .tw-path {
        font-family: monospace;
        overflow-wrap: anywhere;
        font-size: .9rem;
    }
    [class*="st-key-atlas_cell_"] {
        position: relative;
    }
    [class*="st-key-atlas_cell_"] [data-testid="stImage"] {
        margin-bottom: 0;
    }
    [class*="st-key-atlas_cell_"] [data-testid="stImage"] img {
        border-radius: 10px;
        display: block;
    }
    [class*="st-key-atlas_cell_"] .stButton {
        position: relative;
        z-index: 5;
        margin-top: -2.55rem;
        margin-left: .25rem;
        width: fit-content;
    }
    [class*="st-key-atlas_cell_"] .stButton button {
        min-height: 1.75rem;
        padding: .15rem .45rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.55);
        background: rgba(0,0,0,.58);
        color: white;
        backdrop-filter: blur(6px);
        font-size: .68rem;
        box-shadow: 0 2px 10px rgba(0,0,0,.35);
    }
    [class*="st-key-atlas_cell_"] .tw-atlas-label {
        position: relative;
        z-index: 4;
        margin-top: -4.85rem;
        margin-bottom: 2.45rem;
        padding: .25rem .38rem;
        width: fit-content;
        max-width: calc(100% - .5rem);
        border-radius: 8px;
        background: rgba(0,0,0,.54);
        color: white;
        font-size: .64rem;
        line-height: 1.15;
        backdrop-filter: blur(6px);
        pointer-events: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_dataset() -> TinyWorldsDataset:
    return TinyWorldsDataset()


@st.cache_data(show_spinner=False)
def cached_frame(world: str, index: int):
    return load_dataset().read_frame(world, index)


@st.cache_data(show_spinner=False)
def cached_frames(world: str, indices: tuple[int, ...]):
    return load_dataset().read_frames(world, list(indices))


@st.cache_data(show_spinner=False)
def motion_indices(
    world: str,
    start: int,
    stop: int,
    count: int,
    candidates: int = 160,
) -> list[int]:
    probes = sample_indices(load_dataset().get_file(world).frame_count, candidates, start, stop)
    if len(probes) <= count:
        return probes
    scored: list[tuple[float, int]] = []
    previous = cached_frame(world, probes[0])
    for index in probes[1:]:
        current = cached_frame(world, index)
        score = float(abs(current.astype("int16") - previous.astype("int16")).mean())
        scored.append((score, index))
        previous = current
    chosen = [start, stop] + [index for _, index in sorted(scored, reverse=True)[: count - 2]]
    return sorted({max(start, min(stop, index)) for index in chosen})


def label_for(index: int, max_index: int) -> str:
    percent = 100 * index / max(1, max_index)
    return f"#{index:,} · {percent:04.1f}%"


def atlas_cells(start: int, stop: int, k: int) -> list[dict[str, int]]:
    """Split an inclusive frame interval into a deterministic K×K VideoAtlas grid."""
    cells: list[dict[str, int]] = []
    total = max(1, stop - start + 1)
    cell_count = k * k
    for i in range(cell_count):
        cell_start = start + (i * total) // cell_count
        cell_stop = start + (((i + 1) * total) // cell_count) - 1
        cell_stop = max(cell_start, min(stop, cell_stop))
        midpoint = (cell_start + cell_stop) // 2
        cells.append({"cell": i, "start": cell_start, "stop": cell_stop, "midpoint": midpoint})
    return cells


def atlas_depth_for_path(frame_count: int, k: int) -> int:
    if frame_count <= 1:
        return 0
    return math.ceil(math.log(frame_count, k * k))


try:
    dataset = load_dataset()
except Exception as exc:  # noqa: BLE001 - Streamlit should explain dataset setup problems
    st.error(f"Could not load TinyWorlds dataset: {exc}")
    st.info("Set TINYWORLDS_DATASET_DIR to a folder containing *_frames.h5 files.")
    st.stop()

worlds = [file.world for file in dataset.files]
summary = pd.DataFrame(
    [
        {
            "world": f.world,
            "frames": f.frame_count,
            "resolution": f"{f.width}×{f.height}",
            "pixels (M)": round(f.frame_count * f.width * f.height / 1_000_000, 1),
            "file": f.name,
        }
        for f in dataset.files
    ]
)
total_frames = int(summary["frames"].sum())
total_pixels = int(sum(f.frame_count * f.width * f.height for f in dataset.files))

st.title("🕹️ TinyWorlds Explorer")
st.caption("A readable, arcade-flavored dashboard for inspecting tiny game-world frame datasets.")

control_cols = st.columns([1, 2])
with control_cols[0]:
    world = st.selectbox("Selected world", worlds)
    file_info = dataset.get_file(world)
with control_cols[1]:
    st.markdown("**Dataset root**")
    st.markdown(f'<div class="tw-path">{dataset.root}</div>', unsafe_allow_html=True)

overview_tab, atlas_tab, sheet_tab, compare_tab, lab_tab = st.tabs(
    ["Overview", "VideoAtlas", "Contact Sheet", "Compare", "Frame Lab"]
)

with overview_tab:
    st.markdown(
        """
        <div class="tw-hero">
        <b>Landing overview</b><br>
        Start here: dataset shape, paths, representative thumbnails, and quick context before
        diving into timeline-level inspection.
        </div>
        """,
        unsafe_allow_html=True,
    )
    a, b, c = st.columns(3)
    a.metric("Worlds", len(worlds))
    b.metric("Total frames", f"{total_frames:,}")
    c.metric("Total pixels", f"{total_pixels / 1_000_000_000:.2f}B")
    st.info("Custom dataset location: set TINYWORLDS_DATASET_DIR to a folder of *_frames.h5 files.")
    st.dataframe(summary, hide_index=True, width="stretch")

    st.subheader("Representative thumbnails")
    thumb_cols = st.columns(min(5, len(worlds)))
    for i, each_world in enumerate(worlds):
        info = dataset.get_file(each_world)
        idx = info.frame_count // 2
        with thumb_cols[i % len(thumb_cols)]:
            st.image(
                cached_frame(each_world, idx),
                caption=f"{each_world}\nmidpoint #{idx:,}",
                clamp=True,
                width="stretch",
            )

with atlas_tab:
    st.subheader(f"VideoAtlas: {world}")
    st.caption(
        "Recursive K×K temporal grids. Each cell shows the midpoint frame for a contiguous "
        "interval; expand a cell to zoom in by K² with no preprocessing."
    )

    max_index = file_info.frame_count - 1
    atlas_key = f"video_atlas_path_{world}"
    if atlas_key not in st.session_state:
        st.session_state[atlas_key] = []

    controls = st.columns([1, 1, 1, 2])
    k = controls[0].slider(
        "K",
        2,
        8,
        8,
        help="Grid side length. K=8 yields the paper-style 64-cell atlas.",
    )
    fps = controls[1].number_input(
        "FPS",
        min_value=0.1,
        value=30.0,
        step=1.0,
        help="Used for time labels and Dmax. Frame intervals are exact regardless of FPS.",
    )
    scale_seconds = controls[2].toggle("Show seconds", value=True)
    if controls[3].button("Reset to root grid"):
        st.session_state[atlas_key] = []

    path: list[int] = list(st.session_state[atlas_key])
    interval_start, interval_stop = 0, max_index
    for selected_cell in path:
        selected = atlas_cells(interval_start, interval_stop, k)[selected_cell]
        interval_start, interval_stop = selected["start"], selected["stop"]

    depth = len(path)
    dmax = atlas_depth_for_path(file_info.frame_count, k)
    resolution_frames = max(1, math.ceil(file_info.frame_count / ((k * k) ** (depth + 1))))
    resolution_seconds = resolution_frames / fps

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Depth", f"{depth}/{dmax}")
    m2.metric("Current interval", f"#{interval_start:,}–#{interval_stop:,}")
    m3.metric("Grid resolution", f"~{resolution_frames:,} frames")
    m4.metric("Time resolution", f"~{resolution_seconds:.3f}s")

    breadcrumb = "root" + "".join(f" → c{cell}" for cell in path)
    st.markdown(f"**Address:** `{breadcrumb}`")
    if path and st.button("← Collapse one level"):
        st.session_state[atlas_key] = path[:-1]
        st.rerun()

    cells = atlas_cells(interval_start, interval_stop, k)
    midpoints = tuple(cell["midpoint"] for cell in cells)
    frames = cached_frames(world, midpoints)

    st.write(f"Expand any cell in this {k}×{k} grid:")
    for row in range(k):
        cols = st.columns(k)
        for col in range(k):
            cell = cells[row * k + col]
            frame = frames[row * k + col]
            with cols[col]:
                with st.container(key=f"atlas_cell_{world}_{depth}_{row}_{col}_{cell['start']}"):
                    start_label = cell["start"] / fps if scale_seconds else cell["start"]
                    stop_label = cell["stop"] / fps if scale_seconds else cell["stop"]
                    unit = "s" if scale_seconds else "f"
                    caption = (
                        f"c{cell['cell']} · mid #{cell['midpoint']:,}<br>"
                        f"[{start_label:.2f}, {stop_label:.2f}] {unit}"
                        if scale_seconds
                        else f"c{cell['cell']} · mid #{cell['midpoint']:,}<br>"
                        f"[#{cell['start']:,}, #{cell['stop']:,}]"
                    )
                    st.image(frame, clamp=True, width="stretch")
                    st.markdown(
                        f'<div class="tw-atlas-label">{caption}</div>',
                        unsafe_allow_html=True,
                    )
                    disabled = cell["start"] >= cell["stop"]
                    if st.button(
                        "EXPAND",
                        key=f"atlas_{world}_{depth}_{row}_{col}_{cell['start']}",
                        disabled=disabled,
                    ):
                        st.session_state[atlas_key] = path + [cell["cell"]]
                        st.rerun()

with sheet_tab:
    st.subheader(f"{world} contact sheet")
    max_index = file_info.frame_count - 1
    start, stop = st.slider("Range", 0, max_index, (0, min(max_index, 1200)))
    c1, c2, c3, c4 = st.columns(4)
    count = c1.slider("Samples", 4, 64, 24)
    columns = c2.slider("Columns", 2, 10, 6)
    scale = c3.slider("Scale", 1, 5, 3)
    strategy = c4.selectbox("Sampling", ["Even", "Random", "Biggest motion"])

    if strategy == "Random":
        seed = st.number_input("Random seed", value=7, step=1)
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(start, stop + 1), min(count, stop - start + 1)))
    elif strategy == "Biggest motion":
        indices = motion_indices(world, start, stop, count)
    else:
        indices = sample_indices(file_info.frame_count, count, start, stop)

    frames = cached_frames(world, tuple(indices))
    labels = [label_for(index, max_index) for index in indices]
    sheet = contact_sheet(frames, labels, columns=columns, scale=scale)
    st.image(sheet, caption=f"{len(indices)} {strategy.lower()} samples", width="stretch")

    EXPORT_DIR.mkdir(exist_ok=True)
    out_path = EXPORT_DIR / f"{world}_{strategy.lower().replace(' ', '_')}_contact_sheet.png"
    if st.button("Save contact sheet and show file location"):
        sheet.save(out_path)
        st.success(f"Saved: {out_path}")
        st.link_button("Open saved PNG in a new tab", out_path.as_uri())

with compare_tab:
    st.subheader("World comparison sampler")
    sample_at = st.slider("Relative position through each world", 0.0, 1.0, 0.5, 0.01)
    cols = st.columns(min(5, len(worlds)))
    rows = []
    for i, each_world in enumerate(worlds):
        info = dataset.get_file(each_world)
        idx = int(round(sample_at * (info.frame_count - 1)))
        frame = cached_frame(each_world, idx)
        stats = frame_stats(frame)
        with cols[i % len(cols)]:
            st.image(frame, caption=f"{each_world}\n#{idx:,}", clamp=True, width="stretch")
        rows.append(
            {
                "world": each_world,
                "frame": idx,
                "brightness": stats.brightness,
                "contrast": stats.contrast,
                "colors": stats.unique_colors,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with lab_tab:
    st.subheader(f"Frame Lab: {world}")
    max_index = file_info.frame_count - 1
    if "frame_lab_index" not in st.session_state:
        st.session_state.frame_lab_index = 0
    jumps = st.columns(6)
    if jumps[0].button("← Prev"):
        st.session_state.frame_lab_index = max(0, st.session_state.frame_lab_index - 1)
    if jumps[1].button("Next →"):
        st.session_state.frame_lab_index = min(max_index, st.session_state.frame_lab_index + 1)
    if jumps[2].button("Start"):
        st.session_state.frame_lab_index = 0
    if jumps[3].button("Middle"):
        st.session_state.frame_lab_index = max_index // 2
    if jumps[4].button("End"):
        st.session_state.frame_lab_index = max_index
    if jumps[5].button("Random"):
        st.session_state.frame_lab_index = random.randint(0, max_index)

    index = st.slider(
        "Frame timeline",
        0,
        max_index,
        int(st.session_state.frame_lab_index),
        help="Detailed inspector; not the default landing experience.",
    )
    st.session_state.frame_lab_index = index
    show_delta = st.toggle("Show motion/delta from previous frame", value=True)
    frame = cached_frame(world, index)
    prev = cached_frame(world, max(0, index - 1))
    stats = frame_stats(frame)

    left, right = st.columns([2, 1])
    with left:
        st.image(
            frame,
            caption=f"{world} · {label_for(index, max_index)}",
            clamp=True,
            width="content",
        )
        if show_delta:
            st.image(frame_delta(prev, frame), caption="Motion/delta", clamp=True, width="content")
    with right:
        a, b = st.columns(2)
        a.metric("Brightness", f"{stats.brightness:.1f}")
        b.metric("Contrast", f"{stats.contrast:.1f}")
        a.metric("Unique colors", f"{stats.unique_colors:,}")
        b.metric("Frame", f"{index:,}/{max_index:,}")
        rgb_df = pd.DataFrame(
            {"mean RGB": [stats.red, stats.green, stats.blue]},
            index=["R", "G", "B"],
        )
        st.bar_chart(rgb_df)

with st.expander("Raw selected file info"):
    st.json(
        {
            "world": file_info.world,
            "path": str(file_info.path),
            "frame_count": file_info.frame_count,
            "height": file_info.height,
            "width": file_info.width,
            "channels": file_info.channels,
        }
    )
