from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import ObservationDecoder
from .encoder import ObservationEncoder
from .transition import LatentTransition
from .uncertainty_head import UncertaintyHead


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.encoder = ObservationEncoder(obs_dim, hidden_dim, latent_dim)
        self.posterior_update = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.transition = LatentTransition(latent_dim, num_actions, hidden_dim)
        self.decoder = ObservationDecoder(latent_dim, hidden_dim, obs_dim)
        self.uncertainty_head = UncertaintyHead(latent_dim, hidden_dim)

    def _one_hot(self, action: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action.long(), num_classes=self.num_actions).float()

    def observe(self, observation: torch.Tensor, prior_latent: torch.Tensor | None = None) -> torch.Tensor:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        encoded = self.encoder(observation)
        if prior_latent is None:
            return encoded
        correction = self.posterior_update(torch.cat([encoded, prior_latent], dim=-1))
        return encoded + correction

    def predict_next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition(latent, self._one_hot(action))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def predict_uncertainty(self, latent: torch.Tensor) -> torch.Tensor:
        return self.uncertainty_head(latent)

    def forward_sequence(self, observation_sequence: torch.Tensor, action_sequence: torch.Tensor) -> dict:
        latent = self.observe(observation_sequence[:, 0])
        predicted_observations = []
        predicted_uncertainty = []
        prior_latents = []
        posterior_latents = []

        for step in range(action_sequence.shape[1]):
            prior_next_latent = self.predict_next_latent(latent, action_sequence[:, step])
            posterior_next_latent = self.observe(observation_sequence[:, step + 1], prior_next_latent)
            predicted_observations.append(self.decode(prior_next_latent))
            predicted_uncertainty.append(self.predict_uncertainty(prior_next_latent))
            prior_latents.append(prior_next_latent)
            posterior_latents.append(posterior_next_latent)
            latent = posterior_next_latent

        return {
            "predicted_observations": torch.stack(predicted_observations, dim=1),
            "predicted_uncertainty": torch.stack(predicted_uncertainty, dim=1).squeeze(-1),
            "prior_latents": torch.stack(prior_latents, dim=1),
            "posterior_latents": torch.stack(posterior_latents, dim=1),
        }

    def infer_sequence(self, observation_sequence: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        latent = self.observe(observation_sequence[:, 0])
        latents = [latent]
        for step in range(action_sequence.shape[1]):
            prior_next_latent = self.predict_next_latent(latent, action_sequence[:, step])
            latent = self.observe(observation_sequence[:, step + 1], prior_next_latent)
            latents.append(latent)
        return torch.stack(latents, dim=1)

    def imagine(self, initial_observation: torch.Tensor, action_sequence: torch.Tensor) -> dict:
        if initial_observation.dim() == 1:
            initial_observation = initial_observation.unsqueeze(0)
        if action_sequence.dim() == 1:
            action_sequence = action_sequence.unsqueeze(0)

        latent = self.observe(initial_observation)
        predictions = []
        uncertainties = []
        for step in range(action_sequence.shape[1]):
            latent = self.predict_next_latent(latent, action_sequence[:, step])
            predictions.append(self.decode(latent))
            uncertainties.append(self.predict_uncertainty(latent))
        return {
            "predicted_observations": torch.stack(predictions, dim=1),
            "predicted_uncertainty": torch.stack(uncertainties, dim=1).squeeze(-1),
        }

    def rollout(self, initial_observation: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        return self.imagine(initial_observation, action_sequence)["predicted_observations"]
