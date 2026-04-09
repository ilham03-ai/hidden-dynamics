from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    def __init__(self, npz_path: str | Path) -> None:
        data = np.load(npz_path)
        self.observations = torch.from_numpy(data["episode_observations"]).float()
        self.actions = torch.from_numpy(data["episode_actions"]).long()
        self.mode = torch.from_numpy(data["episode_mode"]).float()
        self.charge_active = torch.from_numpy(data["episode_charge_active"]).float()
        self.beacon_lit = torch.from_numpy(data["episode_beacon_lit"]).float()

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, index: int) -> dict:
        return {
            "observations": self.observations[index],
            "actions": self.actions[index],
            "mode": self.mode[index],
            "charge_active": self.charge_active[index],
            "beacon_lit": self.beacon_lit[index],
        }


def load_split(npz_path: str | Path) -> dict:
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
