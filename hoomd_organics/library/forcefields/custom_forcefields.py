import itertools

import hoomd
import numpy as np
import torch
import torch.nn as nn


class BeadSpring:
    def __init__(
        self,
        r_cut,
        beads,
        bonds=None,
        angles=None,
        dihedrals=None,
        exclusions=["bond", "1-3"],
    ):
        self.beads = beads
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.r_cut = r_cut
        self.exclusions = exclusions
        self.hoomd_forcefield = self._create_forcefield()

    def _create_forcefield(self):
        forces = []
        # Create pair force:
        nlist = hoomd.md.nlist.Cell(buffer=0.40, exclusions=self.exclusions)
        lj = hoomd.md.pair.LJ(nlist=nlist)
        bead_types = [key for key in self.beads.keys()]
        all_pairs = list(itertools.combinations_with_replacement(bead_types, 2))
        for pair in all_pairs:
            epsilon0 = self.beads[pair[0]]["epsilon"]
            epsilon1 = self.beads[pair[1]]["epsilon"]
            pair_epsilon = (epsilon0 + epsilon1) / 2

            sigma0 = self.beads[pair[0]]["sigma"]
            sigma1 = self.beads[pair[1]]["sigma"]
            pair_sigma = (sigma0 + sigma1) / 2

            lj.params[pair] = dict(epsilon=pair_epsilon, sigma=pair_sigma)
            lj.r_cut[pair] = self.r_cut
        forces.append(lj)
        # Create bond-stretching force:
        if self.bonds:
            harmonic_bond = hoomd.md.bond.Harmonic()
            for bond_type in self.bonds:
                harmonic_bond.params[bond_type] = self.bonds[bond_type]
            forces.append(harmonic_bond)
        # Create bond-bending force:
        if self.angles:
            harmonic_angle = hoomd.md.angle.Harmonic()
            for angle_type in self.angles:
                harmonic_angle.params[angle_type] = self.angles[angle_type]
            forces.append(harmonic_angle)
        # Create torsion force:
        if self.dihedrals:
            periodic_dihedral = hoomd.md.dihedral.Periodic()
            for dih_type in self.dihedrals:
                periodic_dihedral.params[dih_type] = self.dihedrals[dih_type]
            forces.append(periodic_dihedral)
        return forces


class NN(nn.Module):
    def __init__(self, hidden_dim, n_layers, act_fn="Tanh", dropout=0.5):
        super(NN, self).__init__()
        self.in_dim = 4  # (relative position vector, center-to-center distance)
        self.out_dim = 1  # predicted energy
        self.hidden_dim = hidden_dim

        self.n_layers = n_layers
        self.act_fn = act_fn
        self.dropout = dropout

        self.net = nn.Sequential(*self._get_net())

    def _get_act_fn(self):
        act = getattr(nn, self.act_fn)
        return act()

    def _get_net(self):
        layers = [nn.Linear(self.in_dim, self.hidden_dim), self._get_act_fn()]
        for i in range(self.n_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self._get_act_fn())
            layers.append(nn.Dropout(p=self.dropout))
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        return layers

    def forward(self, pos_1, pos_2):
        x = pos_1 - pos_2
        r = torch.norm(x, dim=1, keepdim=True)
        x = torch.cat((x, r), dim=1)
        energy = self.net(x)
        force = (-1.0) * torch.autograd.grad(
            energy, pos_1, retain_graph=True, create_graph=True
        )[0][0]
        return force


class TorchCustomForce(hoomd.md.force.Custom):
    """
    Custom force that uses a PyTorch model to predict the forces from particle
    positions.
    """

    def __init__(self, model):
        """

        Parameters
        ----------
        model: torch.nn.Module
            pretrained model that predicts forces
        """
        super().__init__()
        # load ML model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.model.to(self.device)

    def set_forces(self, timestep):
        """
        Set the forces on all particles in the system.
        """
        # get positions for all particles
        with self._state.cpu_local_snapshot as snap:
            particle_rtags = np.copy(snap.particles.rtag)
            positions = np.array(
                snap.particles.position[particle_rtags], copy=True
            )

        num_particles = len(positions)
        particle_forces = []
        for i, pos_1 in enumerate(positions):
            pos_1_tensor = (
                torch.from_numpy(pos_1)
                .type(torch.FloatTensor)
                .unsqueeze(0)
                .to(self.device)
            )
            other_particles_idx = list(range(num_particles))
            other_particles_idx.remove(i)
            particle_force = 0
            for pos_2 in positions[other_particles_idx]:
                pos_2_tensor = (
                    torch.from_numpy(pos_2)
                    .type(torch.FloatTensor)
                    .unsqueeze(0)
                    .to(self.device)
                )
                pos_1_tensor.requires_grad = True
                particle_force += (
                    self.model(pos_1_tensor, pos_2_tensor)
                    .cpu()
                    .detach()
                    .numpy()
                )
            particle_forces.append(particle_force)

        with self.cpu_local_force_arrays as arrays:
            arrays.force[particle_rtags] = np.asarray(particle_forces)
