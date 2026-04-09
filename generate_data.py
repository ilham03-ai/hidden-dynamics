from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import DATA_DIR, get_default_config
from utils.data_utils import save_json
from utils.seed import set_global_seed
from world.environment import HiddenModeWorldEnv, WorldState
from world.rules import shortest_path_actions


POLICY_NAMES = [
    "random",
    "correct_pad_then_terminal",
    "wrong_pad_then_terminal",
    "both_pads_then_terminal",
    "direct_terminal",
]
POLICY_PROBABILITIES = np.asarray([0.18, 0.32, 0.18, 0.17, 0.15], dtype=np.float64)


def choose_policy(rng: np.random.Generator) -> str:
    return str(rng.choice(POLICY_NAMES, p=POLICY_PROBABILITIES))


def correct_and_wrong_pads(state: WorldState) -> tuple[tuple[int, int], tuple[int, int]]:
    if state.mode == 0:
        return state.pad_alpha_pos, state.pad_beta_pos
    return state.pad_beta_pos, state.pad_alpha_pos


def build_action_plan(state: WorldState, env: HiddenModeWorldEnv, policy_name: str, rng: np.random.Generator) -> list[int]:
    blocked = env.blocked_positions(state)
    correct_pad, wrong_pad = correct_and_wrong_pads(state)

    def path(start: tuple[int, int], goal: tuple[int, int]) -> list[int]:
        return shortest_path_actions(start, goal, env.grid_size, blocked)

    if policy_name == "random":
        return []

    if policy_name == "correct_pad_then_terminal":
        return path(state.agent_pos, correct_pad) + [4] + path(correct_pad, state.terminal_pos) + [4]

    if policy_name == "wrong_pad_then_terminal":
        return path(state.agent_pos, wrong_pad) + [4] + path(wrong_pad, state.terminal_pos) + [4]

    if policy_name == "both_pads_then_terminal":
        return (
            path(state.agent_pos, state.pad_alpha_pos)
            + [4]
            + path(state.pad_alpha_pos, state.pad_beta_pos)
            + [4]
            + path(state.pad_beta_pos, state.terminal_pos)
            + [4]
        )

    return path(state.agent_pos, state.terminal_pos) + [4]


def extend_plan(plan: list[int], horizon: int, rng: np.random.Generator) -> list[int]:
    if len(plan) < horizon:
        plan = plan + rng.integers(0, 5, size=horizon - len(plan)).tolist()
    return plan[:horizon]


def generate_split(
    split_name: str,
    num_episodes: int,
    horizon: int,
    env: HiddenModeWorldEnv,
    split_seed: int,
    output_dir: Path,
) -> Path:
    rng = np.random.default_rng(split_seed)
    obs_dim = env.observation_dim
    num_transitions = num_episodes * horizon

    observations = np.zeros((num_transitions, obs_dim), dtype=np.float32)
    next_observations = np.zeros((num_transitions, obs_dim), dtype=np.float32)
    actions = np.zeros((num_transitions,), dtype=np.int64)
    hidden_mode = np.zeros((num_transitions,), dtype=np.float32)
    charge_active = np.zeros((num_transitions,), dtype=np.float32)
    next_charge_active = np.zeros((num_transitions,), dtype=np.float32)

    episode_observations = np.zeros((num_episodes, horizon + 1, obs_dim), dtype=np.float32)
    episode_actions = np.zeros((num_episodes, horizon), dtype=np.int64)
    episode_mode = np.zeros((num_episodes,), dtype=np.float32)
    episode_charge_active = np.zeros((num_episodes, horizon + 1), dtype=np.float32)
    episode_beacon_lit = np.zeros((num_episodes, horizon + 1), dtype=np.float32)
    policy_ids = np.zeros((num_episodes,), dtype=np.int64)
    initial_layouts = np.zeros((num_episodes, 12), dtype=np.int64)

    transition_index = 0
    for episode_index in range(num_episodes):
        episode_seed = split_seed + episode_index
        observation, info = env.reset(seed=episode_seed)
        state = env.get_state().copy()
        policy_name = choose_policy(rng)
        plan = extend_plan(build_action_plan(state, env, policy_name, rng), horizon, rng)

        episode_mode[episode_index] = info["mode"]
        episode_observations[episode_index, 0] = observation
        episode_charge_active[episode_index, 0] = info["charge_active"]
        episode_beacon_lit[episode_index, 0] = info["beacon_lit"]
        policy_ids[episode_index] = POLICY_NAMES.index(policy_name)
        initial_layouts[episode_index] = np.asarray(
            [
                *state.agent_pos,
                *state.pad_alpha_pos,
                *state.pad_beta_pos,
                *state.terminal_pos,
                *state.obstacle_1_pos,
                *state.obstacle_2_pos,
            ],
            dtype=np.int64,
        )

        current_observation = observation
        current_info = info
        for step_index, action in enumerate(plan):
            next_observation, _, _, next_info = env.step(action)
            observations[transition_index] = current_observation
            actions[transition_index] = action
            next_observations[transition_index] = next_observation
            hidden_mode[transition_index] = current_info["mode"]
            charge_active[transition_index] = current_info["charge_active"]
            next_charge_active[transition_index] = next_info["charge_active"]

            episode_actions[episode_index, step_index] = action
            episode_observations[episode_index, step_index + 1] = next_observation
            episode_charge_active[episode_index, step_index + 1] = next_info["charge_active"]
            episode_beacon_lit[episode_index, step_index + 1] = next_info["beacon_lit"]

            current_observation = next_observation
            current_info = next_info
            transition_index += 1

    path = output_dir / f"{split_name}.npz"
    np.savez(
        path,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        hidden_mode=hidden_mode,
        charge_active=charge_active,
        next_charge_active=next_charge_active,
        episode_observations=episode_observations,
        episode_actions=episode_actions,
        episode_mode=episode_mode,
        episode_charge_active=episode_charge_active,
        episode_beacon_lit=episode_beacon_lit,
        policy_ids=policy_ids,
        initial_layouts=initial_layouts,
    )
    return path


def add_noise_to_payload(payload: dict, noise_std: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def apply_noise(array: np.ndarray) -> np.ndarray:
        noisy = np.clip(array + rng.normal(0.0, noise_std, size=array.shape).astype(np.float32), 0.0, 1.0)
        noisy[..., -2:] = (noisy[..., -2:] >= 0.5).astype(np.float32)
        return noisy

    noisy_payload = dict(payload)
    noisy_payload["observations"] = apply_noise(payload["observations"])
    noisy_payload["next_observations"] = apply_noise(payload["next_observations"])
    noisy_payload["episode_observations"] = apply_noise(payload["episode_observations"])
    return noisy_payload


def save_noisy_variant(clean_path: Path, noisy_path: Path, noise_std: float, seed: int) -> None:
    clean_payload = {key: value for key, value in np.load(clean_path).items()}
    noisy_payload = add_noise_to_payload(clean_payload, noise_std=noise_std, seed=seed)
    np.savez(noisy_path, **noisy_payload)


def run_generation() -> dict:
    config = get_default_config()
    env_config = config["environment"]
    data_config = config["data"]

    set_global_seed(data_config.seed)
    env = HiddenModeWorldEnv(
        grid_size=env_config.grid_size,
        max_steps=env_config.max_steps,
        charge_duration=env_config.charge_duration,
        observation_noise=env_config.observation_noise,
        action_noise=env_config.action_noise,
        seed=data_config.seed,
    )

    manifest = {}
    split_specs = [
        ("train", data_config.train_episodes, data_config.seed),
        ("val", data_config.val_episodes, data_config.seed + 10_000),
        ("test", data_config.test_episodes, data_config.seed + 20_000),
    ]
    for split_name, episode_count, split_seed in split_specs:
        clean_path = generate_split(split_name, episode_count, data_config.horizon, env, split_seed, DATA_DIR)
        noisy_path = DATA_DIR / f"{split_name}_noisy.npz"
        save_noisy_variant(clean_path, noisy_path, data_config.noisy_observation_std, split_seed + 777)
        manifest[split_name] = str(clean_path)
        manifest[f"{split_name}_noisy"] = str(noisy_path)

    save_json(manifest, DATA_DIR / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate world-model-lab-v2 datasets.")
    parser.parse_args()
    manifest = run_generation()
    for split_name, split_path in manifest.items():
        print(f"{split_name}: {split_path}")


if __name__ == "__main__":
    main()
