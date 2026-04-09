from __future__ import annotations

import argparse

import numpy as np
import torch

from config import FIGURE_DIR, get_default_config
from models.world_model import WorldModel
from utils.plotting import plot_exploration_comparison
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


def select_uncertainty_action(
    model: WorldModel,
    observation: np.ndarray,
    planning_horizon: int,
    num_candidates: int,
    device: torch.device,
    rng: np.random.Generator,
) -> int:
    candidate_sequences = rng.integers(0, 5, size=(num_candidates, planning_horizon))
    repeated_observation = np.repeat(observation[None, :], num_candidates, axis=0)
    with torch.no_grad():
        imagined = model.imagine(
            torch.from_numpy(repeated_observation).float().to(device),
            torch.from_numpy(candidate_sequences).long().to(device),
        )
    scores = imagined["predicted_uncertainty"].sum(dim=1).cpu().numpy()
    return int(candidate_sequences[int(np.argmax(scores)), 0])


def run_policy(
    policy_name: str,
    model: WorldModel,
    config: dict,
    num_episodes: int,
    device: torch.device,
) -> dict:
    exploration_config = config["exploration"]
    env = HiddenModeWorldEnv(
        grid_size=config["environment"]["grid_size"],
        max_steps=config["environment"]["max_steps"],
        charge_duration=config["environment"]["charge_duration"],
    )
    rng = np.random.default_rng(exploration_config["seed"] + (0 if policy_name == "random" else 1_000))

    unique_positions = []
    informative_interactions = []
    predicted_uncertainty = []
    actual_error = []
    success_rate = []

    for episode_index in range(num_episodes):
        observation, _ = env.reset(seed=int(rng.integers(0, 10_000_000)))
        visited = {env.get_state().agent_pos}
        episode_uncertainty = []
        episode_error = []
        episode_informative_interactions = 0
        for _ in range(exploration_config["episode_horizon"]):
            if policy_name == "random":
                action = int(rng.integers(0, 5))
            else:
                action = select_uncertainty_action(
                    model,
                    observation,
                    planning_horizon=exploration_config["planning_horizon"],
                    num_candidates=exploration_config["num_candidates"],
                    device=device,
                    rng=rng,
                )

            with torch.no_grad():
                imagined = model.imagine(
                    torch.from_numpy(observation).float().unsqueeze(0).to(device),
                    torch.tensor([[action]], dtype=torch.long, device=device),
                )
            pred_next = imagined["predicted_observations"].cpu().numpy()[0, 0]
            pred_unc = float(imagined["predicted_uncertainty"].cpu().numpy()[0, 0])

            current_state = env.get_state()
            if action == 4 and current_state.agent_pos in {
                current_state.pad_alpha_pos,
                current_state.pad_beta_pos,
                current_state.terminal_pos,
            }:
                episode_informative_interactions += 1

            next_observation, _, _, info = env.step(action)
            episode_uncertainty.append(pred_unc)
            episode_error.append(float(np.mean((pred_next - next_observation) ** 2)))
            observation = next_observation
            visited.add(env.get_state().agent_pos)

        unique_positions.append(len(visited))
        informative_interactions.append(episode_informative_interactions)
        predicted_uncertainty.append(np.mean(episode_uncertainty))
        actual_error.append(np.mean(episode_error))
        success_rate.append(float(env.get_state().beacon_lit))

    return {
        "mean_unique_positions": float(np.mean(unique_positions)),
        "mean_informative_interactions": float(np.mean(informative_interactions)),
        "mean_predicted_uncertainty": float(np.mean(predicted_uncertainty)),
        "mean_actual_error": float(np.mean(actual_error)),
        "success_rate": float(np.mean(success_rate)),
    }


def run_exploration(checkpoint_path: str) -> dict:
    device = torch.device("cpu")
    model, config = load_model(checkpoint_path, device)
    num_episodes = config["exploration"]["num_episodes"]

    summary = {
        "random": run_policy("random", model, config, num_episodes, device),
        "uncertainty": run_policy("uncertainty", model, config, num_episodes, device),
    }
    plot_exploration_comparison(summary, FIGURE_DIR / "exploration_comparison.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uncertainty-aware exploration.")
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    args = parser.parse_args()
    summary = run_exploration(args.checkpoint)
    print(summary)


if __name__ == "__main__":
    main()
