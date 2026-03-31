from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


ACTIONS = {
    0: "up",
    1: "down",
    2: "left",
    3: "right",
}

ACTION_TO_DELTA = {
    0: (0, 1),
    1: (0, -1),
    2: (-1, 0),
    3: (1, 0),
}


Position = Tuple[int, int]


def add_position(a: Position, b: Position) -> Position:
    return a[0] + b[0], a[1] + b[1]


def in_bounds(position: Position, grid_size: int) -> bool:
    return 0 <= position[0] < grid_size and 0 <= position[1] < grid_size


def normalize_position(position: Position, grid_size: int) -> Tuple[float, float]:
    scale = max(grid_size - 1, 1)
    return position[0] / scale, position[1] / scale


def denormalize_position(coords: Tuple[float, float], grid_size: int) -> Position:
    scale = max(grid_size - 1, 1)
    x = int(np.clip(np.rint(coords[0] * scale), 0, grid_size - 1))
    y = int(np.clip(np.rint(coords[1] * scale), 0, grid_size - 1))
    return x, y


def manhattan_distance(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def greedy_action_toward(start: Position, target: Position, rng: np.random.Generator) -> int:
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    choices = []
    if dx > 0:
        choices.append(3)
    elif dx < 0:
        choices.append(2)
    if dy > 0:
        choices.append(0)
    elif dy < 0:
        choices.append(1)
    if not choices:
        return int(rng.integers(0, len(ACTIONS)))
    return int(choices[rng.integers(0, len(choices))])


def sample_layout(grid_size: int, rng: np.random.Generator) -> Dict[str, Position]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    indices = rng.choice(len(cells), size=5, replace=False)
    agent, crate, switch, beacon, obstacle = [cells[idx] for idx in indices]
    return {
        "agent_pos": agent,
        "crate_pos": crate,
        "switch_pos": switch,
        "beacon_pos": beacon,
        "obstacle_pos": obstacle,
    }
