from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import CHECKPOINT_DIR, FIGURE_DIR, ensure_directories
from models.world_model import WorldModel
from utils.plotting import plot_counterfactual
from world.environment import SyntheticWorldEnv, WorldState
from world.rules import greedy_action_toward, manhattan_distance


def build_counterfactual_scenario() -> tuple[WorldState, dict[str, list[int]]]:
    initial_state = WorldState(
        agent_pos=(2, 3),
        crate_pos=(0, 4),
        switch_pos=(0, 0),
        beacon_pos=(2, 4),
        obstacle_pos=(3, 0),
    )
    action_sequences = {
        "Direct to beacon": [0],
        "Switch then beacon": [2, 2, 1, 1, 1, 3, 3, 0, 0, 0, 0],
    }
    return initial_state, action_sequences


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[WorldModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_config = config["model"]
    model = WorldModel(
        obs_dim=model_config["obs_dim"],
        num_actions=model_config["num_actions"],
        latent_dim=model_config["latent_dim"],
        hidden_dim=model_config["hidden_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def reconstruct_state(
    env: SyntheticWorldEnv,
    observation: np.ndarray,
    hidden_armed: float,
    beacon_lit: float,
    step_count: int,
) -> WorldState:
    decoded = env.decode_observation(observation)
    return WorldState(
        agent_pos=decoded["agent_pos"],
        crate_pos=decoded["crate_pos"],
        switch_pos=decoded["switch_pos"],
        beacon_pos=decoded["beacon_pos"],
        obstacle_pos=decoded["obstacle_pos"],
        armed=bool(hidden_armed >= 0.5),
        beacon_lit=bool(beacon_lit >= 0.5),
        step_count=step_count,
    )


def _greedy_actions_to_target(
    initial_state: WorldState,
    target: tuple[int, int],
    grid_size: int,
    max_steps: int,
    seed: int,
) -> tuple[list[int], WorldState] | None:
    env = SyntheticWorldEnv(grid_size=grid_size, max_steps=max_steps, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset(state=initial_state)
    actions = []
    best_distance = manhattan_distance(env.get_state().agent_pos, target)
    stalled_steps = 0

    while env.get_state().agent_pos != target and len(actions) < max_steps:
        action = greedy_action_toward(env.get_state().agent_pos, target, rng)
        actions.append(action)
        env.step(action)
        distance = manhattan_distance(env.get_state().agent_pos, target)
        if distance < best_distance:
            best_distance = distance
            stalled_steps = 0
        else:
            stalled_steps += 1
        if stalled_steps > max_steps:
            return None

    if env.get_state().agent_pos != target:
        return None
    return actions, env.get_state().copy()


def build_counterfactual_paths(state: WorldState, grid_size: int, max_steps: int) -> dict[str, list[int]] | None:
    direct_result = _greedy_actions_to_target(state, state.beacon_pos, grid_size, max_steps=max_steps, seed=11)
    switch_result = _greedy_actions_to_target(state, state.switch_pos, grid_size, max_steps=max_steps, seed=17)
    if direct_result is None or switch_result is None:
        return None
    to_switch, switch_state = switch_result
    beacon_result = _greedy_actions_to_target(switch_state, switch_state.beacon_pos, grid_size, max_steps=max_steps, seed=23)
    if beacon_result is None:
        return None
    switch_to_beacon, _ = beacon_result
    return {
        "Direct to beacon": direct_result[0],
        "Switch then beacon": to_switch + switch_to_beacon,
    }


def _evaluate_scenario_from_observation(
    model: WorldModel,
    device: torch.device,
    rollout: dict[str, np.ndarray],
    action_sequence: list[int],
) -> np.ndarray:
    initial_observation = torch.from_numpy(rollout["observations"][0]).float().unsqueeze(0).to(device)
    action_tensor = torch.tensor(action_sequence, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_future = model.rollout(initial_observation, action_tensor).cpu().numpy()[0]
    return np.concatenate([rollout["observations"][:1], predicted_future], axis=0)


def _evaluate_scenario_from_latent(
    model: WorldModel,
    device: torch.device,
    latent: np.ndarray,
    initial_observation: np.ndarray,
    action_sequence: list[int],
) -> np.ndarray:
    latent_tensor = torch.from_numpy(latent).float().unsqueeze(0).to(device)
    action_tensor = torch.tensor(action_sequence, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_future = model.rollout_from_latent(latent_tensor, action_tensor).cpu().numpy()[0]
    return np.concatenate([initial_observation[None, :], predicted_future], axis=0)


def evaluate_counterfactual_benchmark(
    model: WorldModel,
    split: dict,
    grid_size: int,
    max_steps: int,
    benchmark_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    env = SyntheticWorldEnv(grid_size=grid_size, max_steps=max_steps)
    rng = np.random.default_rng(seed)
    episode_indices = np.arange(split["episode_observations"].shape[0])
    rng.shuffle(episode_indices)

    cases = []
    for episode_index in episode_indices:
        state = reconstruct_state(
            env,
            split["episode_observations"][episode_index, 0],
            split["episode_hidden_armed"][episode_index, 0],
            split["episode_beacon_lit"][episode_index, 0],
            step_count=0,
        )
        action_sequences = build_counterfactual_paths(state, grid_size, max_steps=max_steps)
        if action_sequences is None:
            continue

        direct_actions = action_sequences["Direct to beacon"]
        via_switch_actions = action_sequences["Switch then beacon"]
        if len(direct_actions) == 0 or len(via_switch_actions) == 0:
            continue

        true_direct = env.rollout(direct_actions, initial_state=state)
        true_via_switch = env.rollout(via_switch_actions, initial_state=state)
        true_direct_final = float(true_direct["observations"][-1, -1] >= 0.5)
        true_via_switch_final = float(true_via_switch["observations"][-1, -1] >= 0.5)
        if true_direct_final == true_via_switch_final:
            continue

        pred_direct = _evaluate_scenario_from_observation(model, device, true_direct, direct_actions)
        pred_via_switch = _evaluate_scenario_from_observation(model, device, true_via_switch, via_switch_actions)
        pred_direct_final = float(pred_direct[-1, -1] >= 0.5)
        pred_via_switch_final = float(pred_via_switch[-1, -1] >= 0.5)

        cases.append(
            {
                "episode_index": int(episode_index),
                "step_index": 0,
                "direct_actions": direct_actions,
                "switch_then_beacon_actions": via_switch_actions,
                "true_direct_final_beacon": true_direct_final,
                "true_switch_then_beacon_final_beacon": true_via_switch_final,
                "pred_direct_final_beacon": pred_direct_final,
                "pred_switch_then_beacon_final_beacon": pred_via_switch_final,
                "paired_correct": bool(
                    (pred_direct_final == true_direct_final)
                    and (pred_via_switch_final == true_via_switch_final)
                ),
                "direct_correct": bool(pred_direct_final == true_direct_final),
                "switch_then_beacon_correct": bool(pred_via_switch_final == true_via_switch_final),
                "true_direct_sequence": true_direct["observations"],
                "pred_direct_sequence": pred_direct,
                "true_switch_sequence": true_via_switch["observations"],
                "pred_switch_sequence": pred_via_switch,
            }
        )
        if len(cases) >= benchmark_size:
            break

    if not cases:
        return {
            "num_cases": 0,
            "pair_accuracy": 0.0,
            "final_beacon_accuracy": 0.0,
            "divergence_accuracy": 0.0,
            "cases": [],
        }

    pair_accuracy = float(np.mean([case["paired_correct"] for case in cases]))
    final_beacon_accuracy = float(
        np.mean(
            [case["direct_correct"] for case in cases]
            + [case["switch_then_beacon_correct"] for case in cases]
        )
    )
    divergence_accuracy = float(
        np.mean(
            [
                (case["pred_switch_then_beacon_final_beacon"] - case["pred_direct_final_beacon"])
                == (case["true_switch_then_beacon_final_beacon"] - case["true_direct_final_beacon"])
                for case in cases
            ]
        )
    )

    return {
        "num_cases": len(cases),
        "pair_accuracy": pair_accuracy,
        "final_beacon_accuracy": final_beacon_accuracy,
        "divergence_accuracy": divergence_accuracy,
        "cases": cases,
    }


def evaluate_filtered_counterfactual_benchmark(
    model: WorldModel,
    split: dict,
    grid_size: int,
    max_steps: int,
    benchmark_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    env = SyntheticWorldEnv(grid_size=grid_size, max_steps=max_steps)
    observations = split["episode_observations"]
    actions = split["episode_actions"]
    armed = split["episode_hidden_armed"]
    beacon_lit = split["episode_beacon_lit"]

    with torch.no_grad():
        posterior_latents = model.infer_sequence(
            torch.from_numpy(observations).float().to(device),
            torch.from_numpy(actions).long().to(device),
        ).cpu().numpy()

    rng = np.random.default_rng(seed)
    candidates = [
        (episode_index, step_index)
        for episode_index in range(observations.shape[0])
        for step_index in range(1, observations.shape[1] - 1)
    ]
    rng.shuffle(candidates)

    cases = []
    for episode_index, step_index in candidates:
        if armed[episode_index, step_index] >= 0.5 or beacon_lit[episode_index, step_index] >= 0.5:
            continue

        state = reconstruct_state(
            env,
            observations[episode_index, step_index],
            armed[episode_index, step_index],
            beacon_lit[episode_index, step_index],
            step_index,
        )
        action_sequences = build_counterfactual_paths(state, grid_size, max_steps=max_steps)
        if action_sequences is None:
            continue

        direct_actions = action_sequences["Direct to beacon"]
        via_switch_actions = action_sequences["Switch then beacon"]
        if len(direct_actions) == 0 or len(via_switch_actions) == 0:
            continue

        true_direct = env.rollout(direct_actions, initial_state=state)
        true_via_switch = env.rollout(via_switch_actions, initial_state=state)
        true_direct_final = float(true_direct["observations"][-1, -1] >= 0.5)
        true_via_switch_final = float(true_via_switch["observations"][-1, -1] >= 0.5)
        if true_direct_final == true_via_switch_final:
            continue

        latent = posterior_latents[episode_index, step_index]
        pred_direct = _evaluate_scenario_from_latent(
            model,
            device,
            latent,
            observations[episode_index, step_index],
            direct_actions,
        )
        pred_via_switch = _evaluate_scenario_from_latent(
            model,
            device,
            latent,
            observations[episode_index, step_index],
            via_switch_actions,
        )
        pred_direct_final = float(pred_direct[-1, -1] >= 0.5)
        pred_via_switch_final = float(pred_via_switch[-1, -1] >= 0.5)

        cases.append(
            {
                "episode_index": int(episode_index),
                "step_index": int(step_index),
                "direct_actions": direct_actions,
                "switch_then_beacon_actions": via_switch_actions,
                "true_direct_final_beacon": true_direct_final,
                "true_switch_then_beacon_final_beacon": true_via_switch_final,
                "pred_direct_final_beacon": pred_direct_final,
                "pred_switch_then_beacon_final_beacon": pred_via_switch_final,
                "paired_correct": bool(
                    (pred_direct_final == true_direct_final)
                    and (pred_via_switch_final == true_via_switch_final)
                ),
                "direct_correct": bool(pred_direct_final == true_direct_final),
                "switch_then_beacon_correct": bool(pred_via_switch_final == true_via_switch_final),
                "true_direct_sequence": true_direct["observations"],
                "pred_direct_sequence": pred_direct,
                "true_switch_sequence": true_via_switch["observations"],
                "pred_switch_sequence": pred_via_switch,
            }
        )
        if len(cases) >= benchmark_size:
            break

    if not cases:
        return {
            "num_cases": 0,
            "pair_accuracy": 0.0,
            "final_beacon_accuracy": 0.0,
            "divergence_accuracy": 0.0,
            "cases": [],
        }

    pair_accuracy = float(np.mean([case["paired_correct"] for case in cases]))
    final_beacon_accuracy = float(
        np.mean(
            [
                case["direct_correct"]
                for case in cases
            ]
            + [
                case["switch_then_beacon_correct"]
                for case in cases
            ]
        )
    )
    divergence_accuracy = float(
        np.mean(
            [
                (case["pred_switch_then_beacon_final_beacon"] - case["pred_direct_final_beacon"])
                == (case["true_switch_then_beacon_final_beacon"] - case["true_direct_final_beacon"])
                for case in cases
            ]
        )
    )

    return {
        "num_cases": len(cases),
        "pair_accuracy": pair_accuracy,
        "final_beacon_accuracy": final_beacon_accuracy,
        "divergence_accuracy": divergence_accuracy,
        "cases": cases,
    }


def run_counterfactual_demo(checkpoint_path: str | Path) -> dict:
    ensure_directories()
    device = torch.device("cpu")
    model, checkpoint_config = load_model(checkpoint_path, device)
    env = SyntheticWorldEnv(
        grid_size=checkpoint_config["environment"]["grid_size"],
        max_steps=checkpoint_config["environment"]["max_steps"],
    )

    initial_state, action_sequences = build_counterfactual_scenario()
    scenario_outputs = {}
    summary = {}

    for name, actions in action_sequences.items():
        rollout = env.rollout(actions, initial_state=initial_state)
        predicted_sequence = _evaluate_scenario_from_observation(model, device, rollout, actions)
        metrics = {
            "true_final_beacon_lit": float(rollout["observations"][-1, -1] >= 0.5),
            "pred_final_beacon_lit": float(predicted_sequence[-1, -1] >= 0.5),
        }
        scenario_outputs[name] = {
            "true": rollout["observations"],
            "pred": predicted_sequence,
            "actions": actions,
            "summary": metrics,
        }
        summary[name] = metrics

    plot_counterfactual(
        scenario_outputs,
        grid_size=checkpoint_config["environment"]["grid_size"],
        output_path=FIGURE_DIR / "counterfactual_rollout.png",
        caption=(
            "Both rows start from the same visible state. The key question is whether the model preserves "
            "the hidden precondition needed for the beacon to activate under the longer action sequence."
        ),
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the counterfactual rollout demo.")
    parser.add_argument(
        "--checkpoint",
        default=str((CHECKPOINT_DIR / "best_model.pt").resolve()),
        help="Path to a trained checkpoint.",
    )
    args = parser.parse_args()
    summary = run_counterfactual_demo(args.checkpoint)
    for name, metrics in summary.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
