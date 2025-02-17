from dataclasses import dataclass

import torch
import torch.nn as nn

from flowermd.library import AnisotropicCustomForce
from flowermd.modules.coarse_graining_ML.anisotropic import (
    orientation_dependent_features,
)


@dataclass
class ForTorPredictorNNModelConfig:
    number_neighbors: int
    hidden_dim: int
    n_layers: int
    box_len: float
    act_fn: str = "Tanh"
    dropout: float = 0.3
    neighbors_pool: str = "mean"
    batch_norm: bool = False


class ForTorPredictorNN(nn.Module):
    def __init__(
        self,
        config: ForTorPredictorNNModelConfig,
        device: torch.device,
        prior_force=False,
        prior_force_sigma=1.0,
        prior_force_n=12,
    ):
        super(ForTorPredictorNN, self).__init__()
        self.neighbor_hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.box_len = config.box_len
        self.act_fn = config.act_fn
        self.dropout = config.dropout
        self.device = device
        self.in_dim = 77
        self.batch_norm = config.batch_norm
        self.neighbor_pool = config.neighbors_pool
        self.prior_force = prior_force
        self.prior_force_sigma = prior_force_sigma
        self.prior_force_n = prior_force_n

        self.neighbors_net = self._neighbors_net().to(self.device)

    def _neighbors_net(self):
        layers = [
            nn.Linear(self.in_dim, self.neighbor_hidden_dim),
            self._get_act_fn(),
        ]
        for i in range(self.n_layers - 1):
            layers.append(
                nn.Linear(self.neighbor_hidden_dim, self.neighbor_hidden_dim)
            )
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.neighbor_hidden_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.neighbor_hidden_dim, 3))
        return nn.Sequential(*layers)

    def _get_act_fn(self) -> nn.Module:
        """
        Get the activation function.
        """
        act = getattr(nn, self.act_fn)
        return act()

    def _pool_neighbors(self, neighbor_features):
        # neighbor_features: (B, N, N_neighbors, hidden_dim)
        if self.neighbor_pool == "mean":
            return neighbor_features.mean(dim=2)
        elif self.neighbor_pool == "max":
            return neighbor_features.max(dim=2)[0]
        elif self.neighbor_pool == "sum":
            return neighbor_features.sum(dim=2)
        else:
            raise ValueError("Invalid neighbor pooling method")

    def _calculate_prior_force(self, R):
        F_0 = (
            (-1)
            * (self.prior_force_n / R)
            * torch.pow(self.prior_force_sigma / R, self.prior_force_n)
        )
        return F_0.sum(dim=2)

    def forward(self, position, orientation_R, neighbor_list):
        # position: particle positions (B, N, 3)
        # orientation_R: particle orientation rotation matrix (B, N, 3, 3)
        # neighbor_list: list of neighbors for each particle
        # (B, N * N_neighbors, 2)

        # features: (B, N, N_neighbors, 80)
        features, R = orientation_dependent_features(
            position, orientation_R, neighbor_list, self.box_len, self.device
        )

        neighbor_features = self.neighbors_net(
            features
        )  # (B, N, N_neighbors, neighbor_hidden_dim)
        # pool over the neighbors dimension
        prediction = self._pool_neighbors(
            neighbor_features
        )  # (B, N, neighbor_hidden_dim)
        if self.prior_force:
            F_0 = self._calculate_prior_force(R)

            prediction = prediction + F_0.to(self.device)

        return prediction


def load_anisotropic_custom_force(
    model_config: ForTorPredictorNNModelConfig,
    force_model_path: str,
    torque_model_path: str,
) -> AnisotropicCustomForce:
    """
    Load the anisotropic custom force model for anisotropic coarse-graining.
    Parameters
    ----------
    model_config: ForTorPredictorNNModelConfig
        Configuration for the force and torque matching models.
    force_model_path: str
        Path to the pre-trained force model.
    torque_model_path: str
        Path to the pre-trained torque model.


    Returns
    -------
    AnisotropicCustomForce
        The custom force object that updates forces and torques during simulation.

    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    force_model = ForTorPredictorNN(model_config, device)
    force_model.load_state_dict(
        torch.load(force_model_path, map_location=device, weights_only=True)[
            "model"
        ]
    )
    torque_model = ForTorPredictorNN(model_config, device)
    torque_model.load_state_dict(
        torch.load(torque_model_path, map_location=device, weights_only=True)[
            "model"
        ]
    )
    custom_force = AnisotropicCustomForce(
        force_model,
        torque_model,
        model_config.box_len,
        model_config.number_neighbors,
        device,
    )
    return custom_force
