from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import DATA_DIR, get_default_config
from utils.data import save_json
from utils.seed import set_global_seed
from world.environment import SyntheticWorldEnv
from world.rules import greedy_action_toward, manhattan_distance


POLICY_NAMES = ["random", "switch_then_beacon", "beacon_then_switch", "crate_first"]
POLICY_PROBABILITIES = np.asarray([0.30, 0.32, 0.23, 0.15], dtype=np.float64)


def choose_policy(rng: np.random.Generator) -> str:
    return str(rng.choice(POLICY_NAMES, p=POLICY_PROBABILITIES))


def choose_action(env: SyntheticWorldEnv, policy_name: str, rng: np.random.Generator, memory: dict) -> int:
    state = env.get_state()
    if rng.random() < 0.15:
        return int(rng.integers(0, 4))

    if policy_name == "random":
        return int(rng.integers(0, 4))

    if policy_name == "switch_then_beacon":
        if not state.armed:
            target = state.switch_pos
        elif not state.beacon_lit:
            target = state.beacon_pos
        else:
            target = state.crate_pos
        return greedy_action_toward(state.agent_pos, target, rng)

    if policy_name == "beacon_then_switch":
        if not memory["first_beacon_visit"]:
            target = state.beacon_pos
        elif not state.armed:
            target = state.switch_pos
        elif not state.beacon_lit:
            target = state.beacon_pos
        else:
            target = state.crate_pos
        return greedy_action_toward(state.agent_pos, target, rng)

    if manhattan_distance(state.agent_pos, state.crate_pos) > 1:
        return greedy_action_toward(state.agent_pos, state.crate_pos, rng)
    if not state.armed:
        return greedy_action_toward(state.agent_pos, state.switch_pos, rng)
    if not state.beacon_lit:
        return greedy_action_toward(state.agent_pos, state.beacon_pos, rng)
    return int(rng.integers(0, 4))


def generate_split(
    split_name: str,
    num_episodes: int,
    horizon: int,
    env: SyntheticWorldEnv,
    split_seed: int,
    output_dir: Path,
) -> Path:
    rng = np.random.default_rng(split_seed)
    obs_dim = env.observation_dim
    num_transitions = num_episodes * horizon

    observations = np.zeros((num_transitions, obs_dim), dtype=np.float32)
    next_observations = np.zeros((num_transitions, obs_dim), dtype=np.float32)
    actions = np.zeros((num_transitions,), dtype=np.int64)
    hidden_armed = np.zeros((num_transitions,), dtype=np.float32)
    next_hidden_armed = np.zeros((num_transitions,), dtype=np.float32)
    episode_observations = np.zeros((num_episodes, horizon + 1, obs_dim), dtype=np.float32)
    episode_actions = np.zeros((num_episodes, horizon), dtype=np.int64)
    episode_hidden_armed = np.zeros((num_episodes, horizon + 1), dtype=np.float32)
    episode_beacon_lit = np.zeros((num_episodes, horizon + 1), dtype=np.float32)
    behavior_ids = np.zeros((num_episodes,), dtype=np.int64)
    initial_layouts = np.zeros((num_episodes, 10), dtype=np.int64)

    transition_index = 0
    for episode_index in range(num_episodes):
        episode_seed = split_seed + episode_index
        observation, info = env.reset(seed=episode_seed)
        policy_name = choose_policy(rng)
        memory = {"first_beacon_visit": False}

        behavior_ids[episode_index] = POLICY_NAMES.index(policy_name)
        state = env.get_state()
        initial_layouts[episode_index] = np.asarray(
            [
                *state.agent_pos,
                *state.crate_pos,
                *state.switch_pos,
                *state.beacon_pos,
                *state.obstacle_pos,
            ],
            dtype=np.int64,
        )

        episode_observations[episode_index, 0] = observation
        episode_hidden_armed[episode_index, 0] = info["armed"]
        episode_beacon_lit[episode_index, 0] = info["beacon_lit"]

        for step_index in range(horizon):
            action = choose_action(env, policy_name, rng, memory)
            next_observation, _, _, next_info = env.step(action)
            current_state = env.get_state()
            memory["first_beacon_visit"] = memory["first_beacon_visit"] or (current_state.agent_pos == current_state.beacon_pos)

            observations[transition_index] = observation
            actions[transition_index] = action
            next_observations[transition_index] = next_observation
            hidden_armed[transition_index] = info["armed"]
            next_hidden_armed[transition_index] = next_info["armed"]

            episode_actions[episode_index, step_index] = action
            episode_observations[episode_index, step_index + 1] = next_observation
            episode_hidden_armed[episode_index, step_index + 1] = next_info["armed"]
            episode_beacon_lit[episode_index, step_index + 1] = next_info["beacon_lit"]

            observation = next_observation
            info = next_info
            transition_index += 1

    split_path = output_dir / f"{split_name}.npz"
    np.savez(
        split_path,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        hidden_armed=hidden_armed,
        next_hidden_armed=next_hidden_armed,
        episode_observations=episode_observations,
        episode_actions=episode_actions,
        episode_hidden_armed=episode_hidden_armed,
        episode_beacon_lit=episode_beacon_lit,
        behavior_ids=behavior_ids,
        initial_layouts=initial_layouts,
    )
    return split_path


def run_generation() -> dict:
    config = get_default_config()
    env_config = config["environment"]
    data_config = config["data"]

    set_global_seed(data_config.seed)
    env = SyntheticWorldEnv(
        grid_size=env_config.grid_size,
        max_steps=env_config.max_steps,
        action_noise=env_config.action_noise,
        observation_noise=env_config.observation_noise,
        seed=data_config.seed,
    )

    manifest = {
        "train": str(
            generate_split(
                "train",
                data_config.train_episodes,
                data_config.horizon,
                env,
                split_seed=data_config.seed,
                output_dir=DATA_DIR,
            )
        ),
        "val": str(
            generate_split(
                "val",
                data_config.val_episodes,
                data_config.horizon,
                env,
                split_seed=data_config.seed + 10_000,
                output_dir=DATA_DIR,
            )
        ),
        "test": str(
            generate_split(
                "test",
                data_config.test_episodes,
                data_config.horizon,
                env,
                split_seed=data_config.seed + 20_000,
                output_dir=DATA_DIR,
            )
        ),
    }
    save_json(manifest, DATA_DIR / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible trajectory data for Hidden Dynamics.")
    parser.parse_args()
    manifest = run_generation()
    for split_name, split_path in manifest.items():
        print(f"{split_name}: {split_path}")


if __name__ == "__main__":
    main()
