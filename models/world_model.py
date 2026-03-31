from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import ObservationDecoder
from .encoder import ObservationEncoder
from .transition import LatentTransition


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
        self.state_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def _one_hot(self, action: torch.Tensor) -> torch.Tensor:
        return F.one_hot(action.long(), num_classes=self.num_actions).float()

    def init_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.latent_dim, device=device)

    def observe(self, observation: torch.Tensor, prior_latent: torch.Tensor | None = None) -> torch.Tensor:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        encoded = self.encoder(observation)
        if prior_latent is None:
            return encoded
        correction = self.posterior_update(torch.cat([encoded, prior_latent], dim=-1))
        return encoded + correction

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.observe(observation)

    def predict_next_latent(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transition(latent, self._one_hot(action))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(latent)
        state_logits = self.state_head(latent)
        beacon_probability = torch.sigmoid(state_logits[..., 1:2])
        return torch.cat([decoded[..., :-1], beacon_probability], dim=-1)

    def predict_state_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self.state_head(latent)

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> dict:
        latent = self.observe(observation)
        next_latent = self.predict_next_latent(latent, action)
        next_observation = self.decode(next_latent)
        return {
            "latent": latent,
            "next_latent": next_latent,
            "next_observation": next_observation,
            "state_logits": self.predict_state_logits(next_latent),
        }

    def forward_sequence(self, observation_sequence: torch.Tensor, action_sequence: torch.Tensor) -> dict:
        latent = self.observe(observation_sequence[:, 0])
        predicted_observations = []
        prior_latents = []
        posterior_latents = []
        prior_state_logits = []

        for step in range(action_sequence.shape[1]):
            prior_next_latent = self.predict_next_latent(latent, action_sequence[:, step])
            posterior_next_latent = self.observe(observation_sequence[:, step + 1], prior_next_latent)
            predicted_observations.append(self.decode(prior_next_latent))
            prior_latents.append(prior_next_latent)
            posterior_latents.append(posterior_next_latent)
            prior_state_logits.append(self.predict_state_logits(prior_next_latent))
            latent = posterior_next_latent

        return {
            "predicted_observations": torch.stack(predicted_observations, dim=1),
            "prior_latents": torch.stack(prior_latents, dim=1),
            "posterior_latents": torch.stack(posterior_latents, dim=1),
            "prior_state_logits": torch.stack(prior_state_logits, dim=1),
        }

    def infer_sequence(self, observation_sequence: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        latent = self.observe(observation_sequence[:, 0])
        latents = [latent]
        for step in range(action_sequence.shape[1]):
            prior_next_latent = self.predict_next_latent(latent, action_sequence[:, step])
            latent = self.observe(observation_sequence[:, step + 1], prior_next_latent)
            latents.append(latent)
        return torch.stack(latents, dim=1)

    def rollout(self, initial_observation: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        if action_sequence.dim() == 1:
            action_sequence = action_sequence.unsqueeze(0)
        if initial_observation.dim() == 1:
            initial_observation = initial_observation.unsqueeze(0)

        latent = self.observe(initial_observation)
        return self.rollout_from_latent(latent, action_sequence)

    def rollout_from_latent(self, initial_latent: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        if action_sequence.dim() == 1:
            action_sequence = action_sequence.unsqueeze(0)
        if initial_latent.dim() == 1:
            initial_latent = initial_latent.unsqueeze(0)

        latent = initial_latent
        predictions = []
        for step in range(action_sequence.shape[1]):
            latent = self.predict_next_latent(latent, action_sequence[:, step])
            predictions.append(self.decode(latent))
        return torch.stack(predictions, dim=1)
