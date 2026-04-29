#!/usr/bin/env python3
"""Visualize an episode from a World Models H5 dataset and save as a GIF.

Each dataset stores flattened timesteps across all episodes. Episode
boundaries are encoded by two parallel arrays:
  - ep_offset[i]  : index of the first timestep in episode i
  - ep_len[i]     : number of timesteps in episode i

This script reads one episode, samples frames at a configurable stride,
and produces a multi-panel GIF showing:
  Panel 1 — pixel observations (image sequence)
  Panel 2 — action dimensions over time, with a moving cursor
  Panel 3 — dataset-specific state signals (e.g. effector pos, finger pos)

Usage:
    python visualize_dataset.py --dataset reacher --episode 0 --stride 5
    python visualize_dataset.py --dataset cube_single_expert --episode 3 \\
        --stride 3 --fps 15 --output cube_ep3.gif
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import hdf5plugin  # registers Blosc and other HDF5 filters at import time
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Dataset registry
# Maps short dataset name → relative path to the HDF5 file.
# ---------------------------------------------------------------------------
DATASETS: Dict[str, str] = {
    "cube_single_expert": ".stable_wm/datasets/cube_single_expert.h5",
    "pusht_expert_train": ".stable_wm/datasets/pusht_expert_train.h5",
    "reacher":            ".stable_wm/datasets/reacher.h5",
    "tworoom":            ".stable_wm/datasets/tworoom.h5",
}

# Human-readable label for each action dimension, indexed by dataset name.
ACTION_LABELS: Dict[str, List[str]] = {
    "cube_single_expert": ["x", "y", "z", "grip_open", "grip_close"],
    "pusht_expert_train": ["x", "y"],
    "reacher":            ["torque_0", "torque_1"],
    "tworoom":            ["dx", "dy"],
}

# Extra state signals to plot in the third panel.
# Structure: dataset → { hdf5_key → [per-column labels] }
# Only keys that exist in the file are loaded; missing keys are silently skipped.
EXTRA_SIGNALS: Dict[str, Dict[str, List[str]]] = {
    "cube_single_expert": {
        "proprio_effector_pos": ["eff_x", "eff_y", "eff_z"],
        "reward":               ["reward"],
    },
    "pusht_expert_train": {
        "proprio": ["p0", "p1", "p2", "p3"],
    },
    "reacher": {
        "finger_pos": ["finger_x", "finger_y"],
        "target_pos": ["target_x", "target_y"],
    },
    "tworoom": {
        "pos_agent": ["agent_x", "agent_y"],
        "pos_target": ["target_x", "target_y"],
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode(
    dataset: str,
    episode: int,
) -> Tuple[NDArray[np.uint8], NDArray[np.float32], Dict[str, NDArray[np.float32]]]:
    """Load pixels, actions, and extra state signals for one episode.

    Reads only the slice [ep_offset, ep_offset + ep_len) from each dataset
    array, so memory usage scales with episode length, not total dataset size.

    Args:
        dataset: Key into DATASETS (e.g. "reacher").
        episode: Zero-based episode index.

    Returns:
        pixels:  uint8 array of shape (T, 224, 224, 3) — RGB frames.
        actions: float32 array of shape (T, A) — action vector per step.
        extra:   dict mapping HDF5 key → float array of shape (T, D).
                 Only keys listed in EXTRA_SIGNALS and present in the file
                 are included. 1-D arrays are reshaped to (T, 1).
    """
    path = DATASETS[dataset]
    with h5py.File(path, "r") as f:
        offset: int = int(f["ep_offset"][episode])
        length: int = int(f["ep_len"][episode])
        sl = slice(offset, offset + length)

        pixels  = f["pixels"][sl]   # (T, 224, 224, 3)
        actions = f["action"][sl]   # (T, A)

        extra: Dict[str, NDArray[np.float32]] = {}
        for key in EXTRA_SIGNALS.get(dataset, {}):
            if key in f:
                arr = np.array(f[key][sl], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[:, None]  # ensure shape (T, D)
                extra[key] = arr

    return pixels, actions, extra


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _build_figure(
    n_rows: int,
) -> Tuple[plt.Figure, plt.Axes, plt.Axes, Optional[plt.Axes]]:
    """Create the matplotlib figure with 2 or 3 stacked panels.

    The image panel receives twice the vertical space of the data panels
    so frames are displayed at a comfortable size.

    Args:
        n_rows: 2 if no extra-signal panel is needed, 3 otherwise.

    Returns:
        fig:    The Figure object passed to FuncAnimation.
        ax_img: Axes for the pixel observation (top panel).
        ax_act: Axes for the action time series (middle panel).
        ax_ex:  Axes for extra state signals (bottom panel), or None.
    """
    height_ratios = [2] + [1] * (n_rows - 1)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(6, 4 * n_rows),
        gridspec_kw={"height_ratios": height_ratios},
    )
    ax_img = axes[0]
    ax_act = axes[1]
    ax_ex  = axes[2] if n_rows == 3 else None
    return fig, ax_img, ax_act, ax_ex


def _init_image_panel(
    ax: plt.Axes,
    first_frame: NDArray[np.uint8],
    dataset: str,
    episode: int,
    total_steps: int,
) -> Tuple[plt.Artist, plt.Text]:
    """Render the first frame and return handles for animation updates.

    Args:
        ax:          Target axes (image panel).
        first_frame: Initial RGB frame, shape (224, 224, 3).
        dataset:     Dataset name shown in the title.
        episode:     Episode index shown in the title.
        total_steps: Full episode length (unstrided) for the title.

    Returns:
        im:    imshow AxesImage — call im.set_data() to update the frame.
        title: Title Text artist — call title.set_text() to update the step.
    """
    ax.axis("off")
    im    = ax.imshow(first_frame)
    title = ax.set_title(
        f"{dataset}  |  episode {episode}  |  step 0/{total_steps}",
        fontsize=9,
    )
    return im, title


def _init_action_panel(
    ax: plt.Axes,
    t_full: NDArray[np.int64],
    t_idx: NDArray[np.int64],
    actions_full: NDArray[np.float32],
    actions_samp: NDArray[np.float32],
    labels: List[str],
    has_extra: bool,
) -> Tuple[List[plt.Line2D], plt.Line2D]:
    """Draw faded full-trajectory action traces and initialise animated lines.

    The faded traces give context about the full episode shape; the animated
    lines grow step-by-step as the GIF plays.

    Args:
        ax:           Target axes (action panel).
        t_full:       Timestep indices for the full (unstrided) episode.
        t_idx:        Timestep indices for the sampled (strided) frames.
        actions_full: Full action array, shape (T_full, A).
        actions_samp: Sampled action array, shape (T, A).
        labels:       Human-readable label per action dimension.
        has_extra:    If False, x-axis label is placed here instead of below.

    Returns:
        act_lines: One animated Line2D per action dimension.
        vline:     Vertical dashed cursor Line2D.
    """
    A = actions_full.shape[1]
    for a in range(A):
        ax.plot(t_full, actions_full[:, a], alpha=0.15, color=f"C{a}", linewidth=0.8)

    act_lines: List[plt.Line2D] = []
    for a in range(A):
        line, = ax.plot(t_idx[:1], actions_samp[:1, a],
                        color=f"C{a}", label=labels[a], linewidth=1.2)
        act_lines.append(line)

    vline, = ax.plot([t_idx[0], t_idx[0]],
                     [actions_full.min(), actions_full.max()],
                     color="k", linestyle="--", linewidth=0.7)

    ax.set_ylabel("action")
    ax.set_xlim(0, len(t_full))
    vmin = np.nanmin(actions_full[np.isfinite(actions_full)]) if np.any(np.isfinite(actions_full)) else -1.0
    vmax = np.nanmax(actions_full[np.isfinite(actions_full)]) if np.any(np.isfinite(actions_full)) else 1.0
    y_pad = max(0.05, (vmax - vmin) * 0.05)
    ax.set_ylim(vmin - y_pad, vmax + y_pad)
    ax.legend(loc="upper right", fontsize=7, ncol=A)
    ax.tick_params(labelsize=7)
    if not has_extra:
        ax.set_xlabel("step")

    return act_lines, vline


def _init_extra_panel(
    ax: plt.Axes,
    t_full: NDArray[np.int64],
    t_idx: NDArray[np.int64],
    extra_full: Dict[str, NDArray[np.float32]],
    extra_samp: Dict[str, NDArray[np.float32]],
    signal_cfg: Dict[str, List[str]],
) -> Tuple[List[Tuple[plt.Line2D, NDArray[np.float32]]], plt.Line2D]:
    """Draw faded full-trajectory state signals and initialise animated lines.

    Args:
        ax:          Target axes (extra-signal panel).
        t_full:      Timestep indices for the full episode.
        t_idx:       Timestep indices for the sampled frames.
        extra_full:  Full signal arrays keyed by HDF5 key, shape (T_full, D).
        extra_samp:  Sampled signal arrays keyed by HDF5 key, shape (T, D).
        signal_cfg:  Mapping of HDF5 key → list of per-column labels.

    Returns:
        ex_lines: List of (Line2D, sampled_column_array) pairs for animation.
        vline:    Vertical dashed cursor Line2D.
    """
    ex_lines: List[Tuple[plt.Line2D, NDArray[np.float32]]] = []
    color_idx = 0

    for key, sublabels in signal_cfg.items():
        if key not in extra_full:
            continue
        arr_full = extra_full[key]
        arr_samp = extra_samp[key]
        for d, lbl in enumerate(sublabels):
            ax.plot(t_full, arr_full[:, d], alpha=0.15,
                    color=f"C{color_idx}", linewidth=0.8)
            line, = ax.plot(t_idx[:1], arr_samp[:1, d],
                            color=f"C{color_idx}", label=lbl, linewidth=1.2)
            ex_lines.append((line, arr_samp[:, d]))
            color_idx += 1

    vline, = ax.plot([t_idx[0], t_idx[0]], ax.get_ylim(),
                     color="k", linestyle="--", linewidth=0.7)

    ax.set_xlabel("step")
    ax.set_ylabel("state")
    ax.set_xlim(0, len(t_full))
    ax.legend(loc="upper right", fontsize=6, ncol=4)
    ax.tick_params(labelsize=7)

    return ex_lines, vline


# ---------------------------------------------------------------------------
# GIF generation
# ---------------------------------------------------------------------------

def make_gif(
    dataset: str,
    episode: int,
    stride: int,
    output: str,
    duration: float = 10.0,
) -> None:
    """Load one episode, build the animated figure, and write a GIF to disk.

    The function samples every `stride`-th frame from the episode to keep GIF
    file size manageable while preserving the overall temporal shape.  The
    faded background traces in the data panels always show the full episode,
    so the viewer has context for the animated portion.

    Playback speed is derived automatically so that every GIF plays back in
    exactly `duration` seconds, regardless of how many frames were sampled.
    This makes episodes from different datasets directly comparable when
    viewed side-by-side.

    Args:
        dataset:  Key into DATASETS — must match one of the four dataset names.
        episode:  Zero-based episode index to visualise.
        stride:   Step between sampled frames. stride=1 includes every frame;
                  stride=5 (default) samples roughly 20 % of frames.
        output:   File path for the output GIF (e.g. "reacher_ep0.gif").
        duration: Target playback length in seconds (default: 10.0).
                  fps is computed as ceil(num_frames / duration).
    """
    plt.style.use("dark_background")

    print(f"Loading {dataset} episode {episode}...")
    pixels, actions, extra = load_episode(dataset, episode)

    T_full        = len(pixels)
    indices       = list(range(0, T_full, stride))
    fps           = max(1, round(len(indices) / duration))
    px            = pixels[indices]
    ac            = actions[indices].astype(np.float32)
    t_full        = np.arange(T_full, dtype=np.int64)
    t_idx         = np.array(indices, dtype=np.int64)
    T             = len(indices)

    extra_full: Dict[str, NDArray[np.float32]] = extra
    extra_samp: Dict[str, NDArray[np.float32]] = {k: v[indices] for k, v in extra.items()}
    signal_cfg    = EXTRA_SIGNALS.get(dataset, {})
    has_extra     = bool(extra_samp)

    n_rows        = 3 if has_extra else 2
    fig, ax_img, ax_act, ax_ex = _build_figure(n_rows)

    im, title = _init_image_panel(ax_img, px[0], dataset, episode, T_full)

    act_lines, vline_act = _init_action_panel(
        ax_act, t_full, t_idx, actions, ac, ACTION_LABELS[dataset], has_extra
    )

    ex_lines: List[Tuple[plt.Line2D, NDArray[np.float32]]] = []
    vline_ex: Optional[plt.Line2D] = None
    if ax_ex is not None:
        ex_lines, vline_ex = _init_extra_panel(
            ax_ex, t_full, t_idx, extra_full, extra_samp, signal_cfg
        )

    plt.tight_layout()

    def update(i: int) -> List[plt.Artist]:
        """Update all animated artists for frame i.

        Args:
            i: Frame index within the sampled sequence (0 … T-1).

        Returns:
            List of artists that changed, passed to blit for efficient redraw.
        """
        im.set_data(px[i])
        title.set_text(
            f"{dataset}  |  episode {episode}  |  step {t_idx[i]}/{T_full}"
        )
        vline_act.set_xdata([t_idx[i], t_idx[i]])
        for a, line in enumerate(act_lines):
            line.set_data(t_idx[: i + 1], ac[: i + 1, a])

        artists: List[plt.Artist] = [im, title, vline_act] + act_lines

        if vline_ex is not None:
            vline_ex.set_xdata([t_idx[i], t_idx[i]])
            artists.append(vline_ex)
            for line, arr in ex_lines:
                line.set_data(t_idx[: i + 1], arr[: i + 1])
                artists.append(line)

        return artists

    print(f"Rendering {T} frames (stride={stride}, fps={fps}, duration≈{T/fps:.1f}s)...")
    anim = animation.FuncAnimation(
        fig, update, frames=T, interval=1000 // fps, blit=True
    )
    anim.save(output, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved → {output}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse command-line arguments and run the visualisation."""
    parser = argparse.ArgumentParser(
        description="Visualize a World Models dataset episode as a GIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASETS),
        help="Dataset to visualise.",
    )
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Zero-based episode index.",
    )
    parser.add_argument(
        "--stride", type=int, default=5,
        help="Frame stride. 1 = every frame; 5 = every 5th frame.",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Target playback length in seconds. fps is derived automatically.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path. Defaults to <dataset>_ep<N>.gif.",
    )
    args = parser.parse_args()

    gif_dir = Path(".stable_wm/datasets/GIFs")
    gif_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or str(gif_dir / f"{args.dataset}_ep{args.episode}.gif")
    make_gif(args.dataset, args.episode, args.stride, output, args.duration)


if __name__ == "__main__":
    main()
