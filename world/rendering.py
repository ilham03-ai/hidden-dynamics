from __future__ import annotations

import os
from typing import Optional

from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle
import numpy as np

from .environment import SyntheticWorldEnv, WorldState


def _draw_grid(ax: Axes, grid_size: int) -> None:
    for x in range(grid_size + 1):
        ax.axvline(x, color="#d9d9d9", linewidth=0.8, zorder=0)
        ax.axhline(x, color="#d9d9d9", linewidth=0.8, zorder=0)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def _cell_xy(position: tuple[int, int]) -> tuple[float, float]:
    return float(position[0]), float(position[1])


def plot_state(ax: Axes, state: WorldState, grid_size: int, title: Optional[str] = None) -> None:
    _draw_grid(ax, grid_size)

    obstacle_x, obstacle_y = _cell_xy(state.obstacle_pos)
    ax.add_patch(Rectangle((obstacle_x, obstacle_y), 1, 1, facecolor="#6c757d", alpha=0.9))

    switch_x, switch_y = _cell_xy(state.switch_pos)
    ax.add_patch(Rectangle((switch_x, switch_y), 1, 1, facecolor="#7fc97f", alpha=0.55))
    ax.text(switch_x + 0.5, switch_y + 0.5, "S", ha="center", va="center", color="#1b4332", fontsize=12)

    beacon_x, beacon_y = _cell_xy(state.beacon_pos)
    beacon_color = "#f4d35e" if state.beacon_lit else "#f28e2b"
    ax.add_patch(Rectangle((beacon_x, beacon_y), 1, 1, facecolor=beacon_color, alpha=0.6))
    ax.text(beacon_x + 0.5, beacon_y + 0.5, "B", ha="center", va="center", color="#5f370e", fontsize=12)

    crate_x, crate_y = _cell_xy(state.crate_pos)
    ax.add_patch(Rectangle((crate_x + 0.12, crate_y + 0.12), 0.76, 0.76, facecolor="#8d6e63"))

    agent_x, agent_y = _cell_xy(state.agent_pos)
    ax.add_patch(Circle((agent_x + 0.5, agent_y + 0.5), 0.28, facecolor="#2b6cb0"))

    caption = f"armed={int(state.armed)} lit={int(state.beacon_lit)}"
    if title:
        ax.set_title(f"{title}\n{caption}", fontsize=10)
    else:
        ax.set_title(caption, fontsize=10)


def plot_observation(ax: Axes, observation: np.ndarray, grid_size: int, title: Optional[str] = None) -> None:
    env = SyntheticWorldEnv(grid_size=grid_size)
    decoded = env.decode_observation(observation)
    state = WorldState(
        agent_pos=decoded["agent_pos"],
        crate_pos=decoded["crate_pos"],
        switch_pos=decoded["switch_pos"],
        beacon_pos=decoded["beacon_pos"],
        obstacle_pos=decoded["obstacle_pos"],
        beacon_lit=bool(decoded["beacon_lit"]),
    )
    plot_state(ax, state, grid_size, title=title)


def save_observation_strip(
    observations: np.ndarray,
    grid_size: int,
    path: str,
    title_prefix: str,
    step_indices: list[int],
) -> None:
    fig, axes = plt.subplots(1, len(step_indices), figsize=(3 * len(step_indices), 3))
    if len(step_indices) == 1:
        axes = [axes]
    for axis, step_index in zip(axes, step_indices):
        plot_observation(axis, observations[step_index], grid_size, title=f"{title_prefix} t={step_index}")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
