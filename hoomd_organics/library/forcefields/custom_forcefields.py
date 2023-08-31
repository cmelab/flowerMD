import itertools

import hoomd
import numpy as np
import torch


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
        self.model.eval()

    def set_forces(self, timestep):
        """
        Set the forces on all particles in the system.
        """
        # get positions for all particles
        with self._state.cpu_local_snapshot as snap:
            particle_rtags = snap.particles.rtag
            positions = np.array(
                snap.particles.position[particle_rtags], copy=True
            )

        num_particles = len(positions)
        predicted_force = []
        for i, pos_1 in enumerate(positions):
            pos_1_tensor = (
                torch.from_numpy(pos_1)
                .type(torch.FloatTensor)
                .unsqueeze(0)
                .to(self.device)
            )
            other_particles_idx = list(range(num_particles))
            other_particles_idx.remove(i)
            for pos_2 in positions[other_particles_idx]:
                pos_2_tensor = (
                    torch.from_numpy(pos_2)
                    .type(torch.FloatTensor)
                    .unsqueeze(0)
                    .to(self.device)
                )
                predicted_force.append(
                    self.model(pos_1_tensor, pos_2_tensor)
                    .cpu()
                    .detach()
                    .numpy()
                )

        with self.cpu_local_force_arrays as arrays:
            arrays.force[particle_rtags] = predicted_force
