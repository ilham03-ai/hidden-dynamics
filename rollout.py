from __future__ import annotations

import argparse

import numpy as np
import torch

from config import FIGURE_DIR, get_default_config
from generate_data import build_action_plan
from models.world_model import WorldModel
from utils.plotting import plot_counterfactual
from world.environment import HiddenModeWorldEnv


def load_model(checkpoint_path: str, device: torch.device) -> tuple[WorldModel, dict]:
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


def find_counterfactual_scenario(model: WorldModel, config: dict, num_trials: int) -> dict:
    env = HiddenModeWorldEnv(
        grid_size=config["environment"]["grid_size"],
        max_steps=config["environment"]["max_steps"],
        charge_duration=config["environment"]["charge_duration"],
    )
    rng = np.random.default_rng(config["data"]["seed"] + 404)
    best = None

    for _ in range(num_trials):
        seed = int(rng.integers(0, 10_000_000))
        initial_observation, _ = env.reset(seed=seed)
        initial_state = env.get_state().copy()
        correct_sequence = build_action_plan(initial_state, env, "correct_pad_then_terminal", rng)
        wrong_sequence = build_action_plan(initial_state, env, "wrong_pad_then_terminal", rng)
        if len(correct_sequence) == 0 or len(wrong_sequence) == 0:
            continue

        true_correct = env.rollout(correct_sequence, initial_state=initial_state)["observations"]
        true_wrong = env.rollout(wrong_sequence, initial_state=initial_state)["observations"]
        if not (true_correct[-1, -1] >= 0.5 and true_wrong[-1, -1] < 0.5):
            continue

        init_tensor = torch.from_numpy(initial_observation).float().unsqueeze(0)
        with torch.no_grad():
            pred_correct = model.rollout(init_tensor, torch.tensor(correct_sequence, dtype=torch.long).unsqueeze(0)).cpu().numpy()[0]
            pred_wrong = model.rollout(init_tensor, torch.tensor(wrong_sequence, dtype=torch.long).unsqueeze(0)).cpu().numpy()[0]

        score = float(pred_correct[-1, -1] - pred_wrong[-1, -1])
        candidate = {
            "seed": seed,
            "initial_state": initial_state,
            "correct_sequence": correct_sequence,
            "wrong_sequence": wrong_sequence,
            "true_correct": true_correct,
            "true_wrong": true_wrong,
            "pred_correct": np.concatenate([true_correct[:1], pred_correct], axis=0),
            "pred_wrong": np.concatenate([true_wrong[:1], pred_wrong], axis=0),
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        raise RuntimeError("No usable counterfactual scenario found.")
    return best


def run_counterfactual_demo(checkpoint_path: str) -> dict:
    default_config = get_default_config()
    device = torch.device("cpu")
    model, config = load_model(checkpoint_path, device)
    scenario = find_counterfactual_scenario(model, config, num_trials=default_config["evaluation"].counterfactual_trials)

    scenario_outputs = {
        "Correct pad then terminal": {
            "true": scenario["true_correct"],
            "pred": scenario["pred_correct"],
        },
        "Wrong pad then terminal": {
            "true": scenario["true_wrong"],
            "pred": scenario["pred_wrong"],
        },
    }
    plot_counterfactual(
        scenario_outputs,
        grid_size=config["environment"]["grid_size"],
        output_path=FIGURE_DIR / "counterfactual_rollout.png",
    )

    return {
        "seed": scenario["seed"],
        "score": scenario["score"],
        "correct_path": {
            "true_final_beacon_lit": float(scenario["true_correct"][-1, -1] >= 0.5),
            "pred_final_beacon_score": float(scenario["pred_correct"][-1, -1]),
        },
        "wrong_path": {
            "true_final_beacon_lit": float(scenario["true_wrong"][-1, -1] >= 0.5),
            "pred_final_beacon_score": float(scenario["pred_wrong"][-1, -1]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate counterfactual rollouts.")
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    args = parser.parse_args()
    summary = run_counterfactual_demo(args.checkpoint)
    print(summary)


if __name__ == "__main__":
    main()
