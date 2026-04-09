from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from config import FIGURE_DIR, get_default_config
from models.world_model import WorldModel
from utils.data_utils import load_split
from utils.plotting import plot_latent_projection


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


def run_latent_analysis(checkpoint_path: str) -> dict:
    default_config = get_default_config()
    device = torch.device("cpu")
    model, config = load_model(checkpoint_path, device)
    split = load_split(Path(__file__).resolve().parent / "data" / "test.npz")

    with torch.no_grad():
        posterior_latents = model.infer_sequence(
            torch.from_numpy(split["episode_observations"]).float().to(device),
            torch.from_numpy(split["episode_actions"]).long().to(device),
        ).cpu().numpy()

    flattened_latents = posterior_latents.reshape(-1, posterior_latents.shape[-1])
    mode_labels = np.repeat(split["episode_mode"], split["episode_observations"].shape[1])
    charge_labels = split["episode_charge_active"].reshape(-1)

    limit = min(default_config["evaluation"].latent_sample_limit, len(flattened_latents))
    flattened_latents = flattened_latents[:limit]
    mode_labels = mode_labels[:limit]
    charge_labels = charge_labels[:limit]

    pca_points = PCA(n_components=2, random_state=default_config["data"].seed).fit_transform(flattened_latents)
    tsne_points = TSNE(
        n_components=2,
        init="random",
        learning_rate="auto",
        perplexity=30,
        random_state=default_config["data"].seed,
    ).fit_transform(flattened_latents)

    plot_latent_projection(pca_points, mode_labels, FIGURE_DIR / "latent_pca_mode.png", "PCA of latent states", "mode")
    plot_latent_projection(pca_points, charge_labels, FIGURE_DIR / "latent_pca_charge.png", "PCA colored by hidden charge", "charge")
    plot_latent_projection(tsne_points, mode_labels, FIGURE_DIR / "latent_tsne_mode.png", "t-SNE of latent states", "mode")
    plot_latent_projection(tsne_points, charge_labels, FIGURE_DIR / "latent_tsne_charge.png", "t-SNE colored by hidden charge", "charge")

    return {"num_points": int(limit)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latent states for world-model-lab-v2.")
    parser.add_argument("--checkpoint", default="results/checkpoints/best_model.pt")
    args = parser.parse_args()
    summary = run_latent_analysis(args.checkpoint)
    print(summary)


if __name__ == "__main__":
    main()
