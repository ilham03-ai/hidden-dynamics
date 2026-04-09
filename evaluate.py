from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import CHECKPOINT_DIR, FIGURE_DIR, get_default_config
from models.world_model import WorldModel
from rollout import run_counterfactual_demo
from utils.data_utils import load_split, save_json
from utils.metrics import mse, per_step_mse, uncertainty_error_correlation
from utils.plotting import plot_noise_bar, plot_rollout_frames, plot_uncertainty_curve


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


def teacher_forced_metrics(model: WorldModel, observations: np.ndarray, actions: np.ndarray, targets: np.ndarray, device: torch.device) -> dict:
    with torch.no_grad():
        outputs = model.forward_sequence(
            torch.from_numpy(observations).float().to(device),
            torch.from_numpy(actions).long().to(device),
        )
    predicted_observations = outputs["predicted_observations"].cpu().numpy()
    predicted_uncertainty = outputs["predicted_uncertainty"].cpu().numpy()
    actual_error = ((predicted_observations - targets) ** 2).mean(axis=-1)
    return {
        "predicted_observations": predicted_observations,
        "predicted_uncertainty": predicted_uncertainty,
        "one_step_mse": mse(predicted_observations, targets),
        "uncertainty_correlation": uncertainty_error_correlation(predicted_uncertainty, actual_error),
        "mean_step_uncertainty": predicted_uncertainty.mean(axis=0),
        "mean_step_error": actual_error.mean(axis=0),
    }


def rollout_metrics(model: WorldModel, split: dict, device: torch.device, rollout_horizon: int) -> dict:
    initial_observations = torch.from_numpy(split["episode_observations"][:, 0]).float().to(device)
    action_sequence = torch.from_numpy(split["episode_actions"][:, :rollout_horizon]).long().to(device)
    with torch.no_grad():
        imagined = model.imagine(initial_observations, action_sequence)
    predicted = imagined["predicted_observations"].cpu().numpy()
    target = split["episode_observations"][:, 1 : rollout_horizon + 1]
    return {
        "predicted_rollouts": predicted,
        "rollout_mse": mse(predicted, target),
        "per_step_mse": per_step_mse(predicted, target),
    }


def run_evaluation(checkpoint_path: str) -> dict:
    default_config = get_default_config()
    evaluation_config = default_config["evaluation"]
    device = torch.device("cpu")
    model, config = load_model(checkpoint_path, device)
    data_root = Path(__file__).resolve().parent / "data"

    test_split = load_split(data_root / "test.npz")
    noisy_split = load_split(data_root / "test_noisy.npz")

    clean_teacher = teacher_forced_metrics(
        model,
        observations=test_split["episode_observations"],
        actions=test_split["episode_actions"],
        targets=test_split["episode_observations"][:, 1:],
        device=device,
    )
    noisy_teacher = teacher_forced_metrics(
        model,
        observations=noisy_split["episode_observations"],
        actions=noisy_split["episode_actions"],
        targets=test_split["episode_observations"][:, 1:],
        device=device,
    )
    rollout = rollout_metrics(model, test_split, device, evaluation_config.rollout_horizon)

    interesting_episode = int(np.argmax(test_split["episode_beacon_lit"][:, -1]))
    true_sequence = test_split["episode_observations"][interesting_episode, : evaluation_config.rollout_horizon + 1]
    predicted_sequence = np.concatenate([true_sequence[:1], rollout["predicted_rollouts"][interesting_episode]], axis=0)

    plot_rollout_frames(
        true_sequence,
        predicted_sequence,
        config["environment"]["grid_size"],
        FIGURE_DIR / "rollout_example.png",
        title="Held-out open-loop rollout",
    )
    plot_uncertainty_curve(
        clean_teacher["mean_step_uncertainty"],
        clean_teacher["mean_step_error"],
        FIGURE_DIR / "uncertainty_over_time.png",
        title="Predicted uncertainty vs actual error",
    )
    plot_noise_bar(clean_teacher["one_step_mse"], noisy_teacher["one_step_mse"], FIGURE_DIR / "noise_robustness.png")

    counterfactual_summary = run_counterfactual_demo(checkpoint_path)
    metrics = {
        "clean_one_step_mse": clean_teacher["one_step_mse"],
        "noisy_one_step_mse": noisy_teacher["one_step_mse"],
        "rollout_mse": rollout["rollout_mse"],
        "rollout_mse_per_step": rollout["per_step_mse"].tolist(),
        "uncertainty_error_correlation": clean_teacher["uncertainty_correlation"],
        "counterfactual": counterfactual_summary,
    }
    save_json(metrics, CHECKPOINT_DIR / "evaluation_metrics.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate world-model-lab-v2.")
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    args = parser.parse_args()
    metrics = run_evaluation(args.checkpoint)
    print(metrics)


if __name__ == "__main__":
    main()
