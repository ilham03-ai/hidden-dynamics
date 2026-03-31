from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, DATA_DIR, FIGURE_DIR, get_default_config, config_to_dict
from models.world_model import WorldModel
from utils.data import EpisodeDataset, save_json
from utils.plotting import plot_loss_curves
from utils.seed import set_global_seed


def compute_losses(
    model: WorldModel,
    batch: dict,
    latent_loss_weight: float,
    rollout_loss_weight: float,
    armed_loss_weight: float,
    beacon_loss_weight: float,
) -> dict:
    observation_sequence = batch["observations"]
    action_sequence = batch["actions"]
    hidden_armed = batch["hidden_armed"][:, 1:]
    beacon_lit = batch["beacon_lit"][:, 1:]
    next_observations = observation_sequence[:, 1:]

    outputs = model.forward_sequence(observation_sequence, action_sequence)
    imagined_observations = model.rollout(observation_sequence[:, 0], action_sequence)
    reconstruction_loss = F.mse_loss(outputs["predicted_observations"], next_observations)
    rollout_loss = F.mse_loss(imagined_observations, next_observations)
    armed_loss = F.binary_cross_entropy_with_logits(
        outputs["prior_state_logits"][..., 0],
        hidden_armed,
    )
    beacon_loss = F.binary_cross_entropy_with_logits(
        outputs["prior_state_logits"][..., 1],
        beacon_lit,
    )
    latent_loss = F.mse_loss(outputs["prior_latents"], outputs["posterior_latents"].detach())
    total_loss = (
        reconstruction_loss
        + latent_loss_weight * latent_loss
        + rollout_loss_weight * rollout_loss
        + armed_loss_weight * armed_loss
        + beacon_loss_weight * beacon_loss
    )
    return {
        "total_loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "rollout_loss": rollout_loss,
        "latent_loss": latent_loss,
        "armed_loss": armed_loss,
        "beacon_loss": beacon_loss,
    }


def run_epoch(
    model: WorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    latent_loss_weight: float,
    rollout_loss_weight: float,
    armed_loss_weight: float,
    beacon_loss_weight: float,
    grad_clip_norm: float,
) -> dict:
    training = optimizer is not None
    model.train(training)
    running = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "rollout_loss": 0.0,
        "latent_loss": 0.0,
        "armed_loss": 0.0,
        "beacon_loss": 0.0,
    }

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        losses = compute_losses(
            model,
            batch,
            latent_loss_weight,
            rollout_loss_weight,
            armed_loss_weight,
            beacon_loss_weight,
        )
        if not torch.isfinite(losses["total_loss"]):
            raise RuntimeError("Encountered a non-finite loss during training.")
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        for key in running:
            running[key] += float(losses[key].detach().cpu())

    num_batches = max(len(loader), 1)
    return {key: value / num_batches for key, value in running.items()}


def run_training() -> dict:
    config = get_default_config()
    training_config = config["training"]
    model_config = config["model"]

    set_global_seed(training_config.seed)
    device = torch.device(training_config.device)

    train_dataset = EpisodeDataset(DATA_DIR / "train.npz")
    val_dataset = EpisodeDataset(DATA_DIR / "val.npz")
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False)

    model = WorldModel(
        obs_dim=model_config.obs_dim,
        num_actions=model_config.num_actions,
        latent_dim=model_config.latent_dim,
        hidden_dim=model_config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history = {
        "train_total_loss": [],
        "val_total_loss": [],
        "train_reconstruction_loss": [],
        "val_reconstruction_loss": [],
        "train_rollout_loss": [],
        "val_rollout_loss": [],
        "train_latent_loss": [],
        "val_latent_loss": [],
        "train_armed_loss": [],
        "val_armed_loss": [],
        "train_beacon_loss": [],
        "val_beacon_loss": [],
    }

    best_val_loss = float("inf")
    best_checkpoint_path = CHECKPOINT_DIR / "best_model.pt"

    for epoch in range(1, training_config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training_config.latent_loss_weight,
            training_config.rollout_loss_weight,
            training_config.armed_loss_weight,
            training_config.beacon_loss_weight,
            training_config.grad_clip_norm,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                latent_loss_weight=training_config.latent_loss_weight,
                rollout_loss_weight=training_config.rollout_loss_weight,
                armed_loss_weight=training_config.armed_loss_weight,
                beacon_loss_weight=training_config.beacon_loss_weight,
                grad_clip_norm=training_config.grad_clip_norm,
            )

        history["train_total_loss"].append(train_metrics["total_loss"])
        history["val_total_loss"].append(val_metrics["total_loss"])
        history["train_reconstruction_loss"].append(train_metrics["reconstruction_loss"])
        history["val_reconstruction_loss"].append(val_metrics["reconstruction_loss"])
        history["train_rollout_loss"].append(train_metrics["rollout_loss"])
        history["val_rollout_loss"].append(val_metrics["rollout_loss"])
        history["train_latent_loss"].append(train_metrics["latent_loss"])
        history["val_latent_loss"].append(val_metrics["latent_loss"])
        history["train_armed_loss"].append(train_metrics["armed_loss"])
        history["val_armed_loss"].append(val_metrics["armed_loss"])
        history["train_beacon_loss"].append(train_metrics["beacon_loss"])
        history["val_beacon_loss"].append(val_metrics["beacon_loss"])

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config_to_dict(config),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                best_checkpoint_path,
            )

        print(
            f"epoch={epoch:02d} "
            f"train_total={train_metrics['total_loss']:.4f} "
            f"val_total={val_metrics['total_loss']:.4f}"
        , flush=True)

    plot_loss_curves(history, FIGURE_DIR / "loss_curves.png")
    save_json(history, CHECKPOINT_DIR / "training_history.json")
    return {
        "best_checkpoint": str(best_checkpoint_path),
        "best_val_loss": best_val_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the latent world model.")
    parser.parse_args()
    outcome = run_training()
    print(f"best_checkpoint: {outcome['best_checkpoint']}")
    print(f"best_val_loss: {outcome['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
