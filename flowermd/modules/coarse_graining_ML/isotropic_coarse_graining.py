from dataclasses import dataclass

import torch
import torch.nn as nn

from flowermd.internal import adjust_periodic_boundary, neighbors_dr
from flowermd.library import IsotropicCustomForce


@dataclass
class LJNeighborModelConfig:
    number_neighbors: int
    hidden_dim: int
    n_layers: int
    box_len: float
    act_fn: str = "ReLU"
    dropout: float = 0.3
    batch_norm: bool = True
    prior_energy: bool = True
    prior_energy_sigma: float = 1.0
    prior_energy_n: int = 12


class LennardJonesNeighborModel(nn.Module):
    def __init__(self, config: LJNeighborModelConfig, device: torch.device):
        """
        Initialize the Lennard-Jones Neighbor Model.
        """
        super(LennardJonesNeighborModel, self).__init__()
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.box_len = config.box_len
        self.act_fn = config.act_fn
        self.dropout = config.dropout
        self.batch_norm = config.batch_norm
        self.prior_energy = config.prior_energy
        self.prior_energy_sigma = config.prior_energy_sigma
        self.prior_energy_n = config.prior_energy_n
        self.in_dim = 5  # Assuming 5 input features: dr (3), R (1), inv_R (1)
        self.energy_net = self._build_energy_net()
        self.device = device

    def init_net_weights(self, m: nn.Module):
        """
        Initialize network weights.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _get_act_fn(self) -> nn.Module:
        """
        Get the activation function.
        """
        act = getattr(nn, self.act_fn)
        return act()

    def _distance_features(
        self, positions: torch.Tensor, neighbor_list: torch.Tensor
    ) -> torch.Tensor:
        batch_size = positions.shape[0]
        N_particles = positions.shape[1]
        neighbor_list = neighbor_list.reshape(
            batch_size, N_particles, -1, neighbor_list.shape[-1]
        )[:, :, :, 1].to(self.device)  # (B, N, N_neighbors)
        self.dr = neighbors_dr(
            positions, neighbor_list
        )  # (B, N, N_neighbors, 3)
        self.dr = adjust_periodic_boundary(self.dr, self.box_len)

        self.R = torch.norm(
            self.dr, dim=-1, keepdim=True
        )  # (B, N, N_neighbors, 1)
        inv_R = 1.0 / self.R  # (B, N, N_neighbors, 1)

        features = torch.cat((self.dr, self.R, inv_R), dim=-1)
        return features.to(self.device)

    def _build_energy_net(self) -> nn.Sequential:
        """
        Build the energy network.
        """
        layers = [nn.Linear(self.in_dim, self.hidden_dim), self._get_act_fn()]
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers)

    def _calculate_prior_energy(self) -> torch.Tensor:
        """
        Calculate the prior energy.
        """
        U_0 = torch.pow(self.prior_energy_sigma / self.R, self.prior_energy_n)
        return U_0  # (B, N, 1)

    def forward(
        self, positions: torch.Tensor, neighbor_list: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        """
        features = self._distance_features(
            positions, neighbor_list
        )  # (B, N, N_neighbors, 5)
        pair_energies = self.energy_net(features)  # (B, N, N_neighbors, 1)
        if self.prior_energy:
            U_0 = self._calculate_prior_energy()
            pair_energies = pair_energies + U_0.to(self.device)
        pair_forces = -torch.autograd.grad(
            pair_energies.sum(), self.dr, create_graph=True
        )[0].to(self.device)  # (B, N, N_neighbors, 3)
        return pair_forces.sum(dim=2)  # (B, N, 3)


def load_isotropic_custom_force(
    model_config: LJNeighborModelConfig, best_model_path: str
) -> IsotropicCustomForce:
    """
    Load the isotropic custom force.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LennardJonesNeighborModel(model_config, device)
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)[
            "model"
        ]
    )
    force = IsotropicCustomForce(
        model, model_config.box_len, model_config.number_neighbors, device
    )
    return force
