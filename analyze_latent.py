from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import CHECKPOINT_DIR, DATA_DIR, FIGURE_DIR, get_default_config
from models.world_model import WorldModel
from utils.data import load_split, save_json
from utils.metrics import binary_classification_metrics_from_logits
from utils.plotting import plot_latent_embedding, plot_latent_pca
from utils.seed import set_global_seed


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


def _collect_latents(
    model: WorldModel,
    split: dict,
    device: torch.device,
    max_samples: int | None = None,
) -> dict:
    observations = torch.from_numpy(split["episode_observations"]).float().to(device)
    actions = torch.from_numpy(split["episode_actions"]).long().to(device)
    with torch.no_grad():
        latents = model.infer_sequence(observations, actions).cpu().numpy()

    time_steps = np.broadcast_to(
        np.arange(latents.shape[1], dtype=np.float32)[None, :],
        split["episode_hidden_armed"].shape,
    )
    payload = {
        "latents": latents.reshape(-1, latents.shape[-1]),
        "armed": split["episode_hidden_armed"].reshape(-1).astype(np.float32),
        "beacon_lit": split["episode_beacon_lit"].reshape(-1).astype(np.float32),
        "timestep": time_steps.reshape(-1),
    }
    if max_samples is not None and payload["latents"].shape[0] > max_samples:
        for key, value in payload.items():
            payload[key] = value[:max_samples]
    return payload


def _fit_linear_probe(
    train_latents: np.ndarray,
    train_targets: np.ndarray,
    test_latents: np.ndarray,
    test_targets: np.ndarray,
    epochs: int,
    learning_rate: float,
) -> dict:
    mean = train_latents.mean(axis=0, keepdims=True)
    std = train_latents.std(axis=0, keepdims=True) + 1e-6
    train_x = torch.from_numpy((train_latents - mean) / std).float()
    test_x = torch.from_numpy((test_latents - mean) / std).float()
    train_y = torch.from_numpy(train_targets.astype(np.float32)).float()

    probe = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)

    for _ in range(epochs):
        logits = probe(train_x).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = probe(test_x).squeeze(-1).cpu().numpy()
    metrics = binary_classification_metrics_from_logits(test_logits, test_targets)
    metrics["train_loss_final"] = float(loss.detach().cpu())
    return metrics


def _pca_projection(latents: np.ndarray) -> np.ndarray:
    centered = latents - latents.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _centroid_vector(embedding: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return np.zeros((embedding.shape[1],), dtype=np.float32)
    return embedding[labels].mean(axis=0) - embedding[~labels].mean(axis=0)


def _normed_gap(embedding: np.ndarray, labels: np.ndarray) -> float:
    vector = _centroid_vector(embedding, labels)
    scale = np.std(embedding, axis=0).mean() + 1e-6
    return float(np.linalg.norm(vector) / scale)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-6
    return float(np.dot(a, b) / denom)


def _build_summary(analysis: dict) -> dict:
    lines = []
    probe_accuracy = analysis["linear_probe"]["accuracy"]
    armed_gap = analysis["pca_separation"]["armed_gap"]
    beacon_gap = analysis["pca_separation"]["beacon_gap"]
    time_corr = analysis["pca_temporal_structure"]["max_abs_correlation"]
    axis_alignment = abs(analysis["pca_axis_alignment"]["armed_vs_beacon_cosine"])

    if probe_accuracy >= 0.85:
        lines.append("Posterior latents support strong linear recovery of the hidden armed state on held-out episodes.")
    elif probe_accuracy >= 0.7:
        lines.append("Posterior latents retain a usable but incomplete linear signal for the hidden armed state.")
    else:
        lines.append("Hidden armed state is only weakly linearly recoverable from the posterior latent.")

    if armed_gap >= 1.0:
        lines.append("The first two principal components show visible separation between armed and unarmed states.")
    elif armed_gap >= 0.5:
        lines.append("The PCA projection shows partial, but not clean, separation between armed and unarmed states.")
    else:
        lines.append("Hidden-state separation is weak in the first two PCA dimensions.")

    if time_corr >= 0.25:
        lines.append("Trajectory phase leaves a measurable gradient in the leading latent directions.")
    else:
        lines.append("Temporal structure is present but not dominant in the leading PCA directions.")

    if armed_gap > 0.5 and beacon_gap > 0.5 and axis_alignment < 0.85:
        lines.append("Hidden and visible outcome variables appear partially disentangled rather than collapsing onto the same PCA axis.")
    else:
        lines.append("Hidden and visible variables remain at least partly entangled in the low-dimensional projection.")

    return {
        "summary_lines": lines,
        "summary_text": " ".join(lines),
    }


def analyze_latent_space(
    model: WorldModel,
    train_split: dict,
    test_split: dict,
    device: torch.device,
    evaluation_config,
) -> dict:
    train_payload = _collect_latents(model, train_split, device)
    test_payload = _collect_latents(model, test_split, device, max_samples=evaluation_config.latent_samples)

    linear_probe = _fit_linear_probe(
        train_payload["latents"],
        train_payload["armed"],
        test_payload["latents"],
        test_payload["armed"],
        epochs=evaluation_config.latent_probe_epochs,
        learning_rate=evaluation_config.latent_probe_learning_rate,
    )

    plot_latent_pca(
        test_payload["latents"],
        test_payload["armed"],
        FIGURE_DIR / "latent_pca_armed.png",
        title="Posterior latent PCA colored by hidden armed state",
        label_name="armed",
        discrete=True,
    )
    plot_latent_pca(
        test_payload["latents"],
        test_payload["armed"],
        FIGURE_DIR / "latent_pca.png",
        title="Posterior latent PCA colored by hidden armed state",
        label_name="armed",
        discrete=True,
    )
    plot_latent_pca(
        test_payload["latents"],
        test_payload["beacon_lit"],
        FIGURE_DIR / "latent_pca_beacon_lit.png",
        title="Posterior latent PCA colored by beacon state",
        label_name="beacon_lit",
        discrete=True,
    )
    plot_latent_pca(
        test_payload["latents"],
        test_payload["timestep"],
        FIGURE_DIR / "latent_pca_timestep.png",
        title="Posterior latent PCA colored by trajectory phase",
        label_name="timestep",
        discrete=False,
    )

    embedding = _pca_projection(test_payload["latents"])
    armed_vector = _centroid_vector(embedding, test_payload["armed"])
    beacon_vector = _centroid_vector(embedding, test_payload["beacon_lit"])
    pc1_corr = float(np.corrcoef(embedding[:, 0], test_payload["timestep"])[0, 1])
    pc2_corr = float(np.corrcoef(embedding[:, 1], test_payload["timestep"])[0, 1])

    analysis = {
        "linear_probe": linear_probe,
        "pca_separation": {
            "armed_gap": _normed_gap(embedding, test_payload["armed"]),
            "beacon_gap": _normed_gap(embedding, test_payload["beacon_lit"]),
        },
        "pca_temporal_structure": {
            "pc1_timestep_correlation": pc1_corr,
            "pc2_timestep_correlation": pc2_corr,
            "max_abs_correlation": float(max(abs(pc1_corr), abs(pc2_corr))),
        },
        "pca_axis_alignment": {
            "armed_vs_beacon_cosine": _cosine_similarity(armed_vector, beacon_vector),
        },
        "tsne_generated": False,
    }

    if evaluation_config.enable_tsne:
        try:
            from sklearn.manifold import TSNE

            tsne_count = min(600, test_payload["latents"].shape[0])
            tsne = TSNE(
                n_components=2,
                init="pca",
                learning_rate="auto",
                perplexity=min(30, max(5, tsne_count // 20)),
                random_state=evaluation_config.seed,
            )
            embedding_tsne = tsne.fit_transform(test_payload["latents"][:tsne_count])
            plot_latent_embedding(
                embedding_tsne,
                test_payload["armed"][:tsne_count],
                FIGURE_DIR / "latent_tsne_armed.png",
                title="Optional t-SNE view of posterior latents",
                label_name="armed",
                discrete=True,
            )
            analysis["tsne_generated"] = True
        except Exception as exc:
            analysis["tsne_error"] = str(exc)

    analysis.update(_build_summary(analysis))
    save_json(analysis, CHECKPOINT_DIR / "latent_analysis_summary.json")
    (CHECKPOINT_DIR / "latent_analysis_summary.txt").write_text(
        analysis["summary_text"] + "\n",
        encoding="utf-8",
    )
    return analysis


def run_latent_analysis(checkpoint_path: str | Path) -> dict:
    config = get_default_config()
    evaluation_config = config["evaluation"]
    set_global_seed(evaluation_config.seed)
    device = torch.device("cpu")
    model, _ = load_model(checkpoint_path, device)
    train_split = load_split(DATA_DIR / "train.npz")
    test_split = load_split(DATA_DIR / "test.npz")
    return analyze_latent_space(model, train_split, test_split, device, evaluation_config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latent-state structure for a trained world model.")
    parser.add_argument(
        "--checkpoint",
        default=str((CHECKPOINT_DIR / "best_model.pt").resolve()),
        help="Path to a trained checkpoint.",
    )
    args = parser.parse_args()
    summary = run_latent_analysis(args.checkpoint)
    print(summary["summary_text"])


if __name__ == "__main__":
    main()
