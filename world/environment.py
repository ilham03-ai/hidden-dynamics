from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .rules import ACTION_TO_DELTA, add_position, denormalize_position, in_bounds, normalize_position, sample_layout


@dataclass
class WorldState:
    agent_pos: Tuple[int, int]
    crate_pos: Tuple[int, int]
    switch_pos: Tuple[int, int]
    beacon_pos: Tuple[int, int]
    obstacle_pos: Tuple[int, int]
    armed: bool = False
    beacon_lit: bool = False
    step_count: int = 0

    def copy(self) -> "WorldState":
        return replace(self)


class SyntheticWorldEnv:
    def __init__(
        self,
        grid_size: int = 6,
        max_steps: int = 18,
        action_noise: float = 0.0,
        observation_noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_noise = action_noise
        self.observation_noise = observation_noise
        self.rng = np.random.default_rng(seed)
        self.state: Optional[WorldState] = None

    @property
    def observation_dim(self) -> int:
        return 11

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def sample_state(self) -> WorldState:
        layout = sample_layout(self.grid_size, self.rng)
        return WorldState(**layout)

    def reset(
        self,
        seed: Optional[int] = None,
        state: Optional[WorldState] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if seed is not None:
            self.seed(seed)
        self.state = state.copy() if state is not None else self.sample_state()
        return self.get_observation(), self.get_info()

    def get_state(self) -> WorldState:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        return self.state

    def set_state(self, state: WorldState) -> None:
        self.state = state.copy()

    def get_info(self) -> Dict[str, float]:
        state = self.get_state()
        return {
            "armed": float(state.armed),
            "beacon_lit": float(state.beacon_lit),
            "step_count": float(state.step_count),
        }

    def get_observation(self, noise_std: Optional[float] = None) -> np.ndarray:
        state = self.get_state()
        obs = np.array(
            [
                *normalize_position(state.agent_pos, self.grid_size),
                *normalize_position(state.crate_pos, self.grid_size),
                *normalize_position(state.switch_pos, self.grid_size),
                *normalize_position(state.beacon_pos, self.grid_size),
                *normalize_position(state.obstacle_pos, self.grid_size),
                float(state.beacon_lit),
            ],
            dtype=np.float32,
        )
        std = self.observation_noise if noise_std is None else noise_std
        if std > 0.0:
            noise = self.rng.normal(0.0, std, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)
            obs[-1] = 1.0 if obs[-1] >= 0.5 else 0.0
        return obs

    def decode_observation(self, observation: np.ndarray) -> Dict[str, Tuple[int, int] | bool]:
        clipped = np.clip(observation.astype(np.float32), 0.0, 1.0)
        return {
            "agent_pos": denormalize_position((clipped[0], clipped[1]), self.grid_size),
            "crate_pos": denormalize_position((clipped[2], clipped[3]), self.grid_size),
            "switch_pos": denormalize_position((clipped[4], clipped[5]), self.grid_size),
            "beacon_pos": denormalize_position((clipped[6], clipped[7]), self.grid_size),
            "obstacle_pos": denormalize_position((clipped[8], clipped[9]), self.grid_size),
            "beacon_lit": bool(clipped[10] >= 0.5),
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        state = self.get_state()
        if self.action_noise > 0.0 and self.rng.random() < self.action_noise:
            action = int(self.rng.integers(0, len(ACTION_TO_DELTA)))

        delta = ACTION_TO_DELTA[int(action)]
        next_agent = add_position(state.agent_pos, delta)
        next_crate = state.crate_pos

        if in_bounds(next_agent, self.grid_size) and next_agent != state.obstacle_pos:
            if next_agent == state.crate_pos:
                pushed_crate = add_position(state.crate_pos, delta)
                if in_bounds(pushed_crate, self.grid_size) and pushed_crate != state.obstacle_pos:
                    next_agent = next_agent
                    next_crate = pushed_crate
                else:
                    next_agent = state.agent_pos
            else:
                next_agent = next_agent
        else:
            next_agent = state.agent_pos

        state.agent_pos = next_agent
        state.crate_pos = next_crate
        state.step_count += 1

        if state.agent_pos == state.switch_pos:
            state.armed = True
        if state.agent_pos == state.beacon_pos and state.armed:
            state.beacon_lit = True

        reward = 1.0 if state.beacon_lit else 0.0
        done = state.step_count >= self.max_steps
        return self.get_observation(), reward, done, self.get_info()

    def rollout(
        self,
        action_sequence: List[int],
        initial_state: Optional[WorldState] = None,
        observation_noise: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        self.reset(state=initial_state)
        observations = [self.get_observation(noise_std=observation_noise)]
        armed = [float(self.get_state().armed)]
        beacon_lit = [float(self.get_state().beacon_lit)]
        for action in action_sequence:
            obs, _, _, info = self.step(int(action))
            if observation_noise > 0.0:
                obs = self.get_observation(noise_std=observation_noise)
            observations.append(obs)
            armed.append(info["armed"])
            beacon_lit.append(info["beacon_lit"])
        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(action_sequence, dtype=np.int64),
            "armed": np.asarray(armed, dtype=np.float32),
            "beacon_lit": np.asarray(beacon_lit, dtype=np.float32),
        }
