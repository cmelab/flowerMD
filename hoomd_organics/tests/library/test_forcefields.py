import os

import hoomd
import torch
import torch.nn as nn

from hoomd_organics import Molecule, Pack, Simulation
from hoomd_organics.library import (
    GAFF,
    OPLS_AA,
    OPLS_AA_BENZENE,
    OPLS_AA_DIMETHYLETHER,
    OPLS_AA_PPS,
    BeadSpring,
    FF_from_file,
    TorchCustomForce,
)
from hoomd_organics.tests.base_test import ASSETS_DIR, BaseTest


class NN(nn.Module):
    def __init__(
        self, hidden_dim, out_dim, n_layers, act_fn="Tanh", dropout=0.5
    ):
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
        pos_1.requires_grad = True
        x = torch.tensor(pos_1 - pos_2).to(pos_1.device)
        x = torch.cat(
            (x, torch.norm(x, dim=1, keepdim=True).to(x.device)), dim=1
        ).to(x.device)
        energy = self.net(x)
        force = (-1.0) * torch.autograd.grad(
            energy, pos_1, retain_graph=True, create_graph=True
        )[0].to(energy.device)
        return force


class TestXMLForceFields:
    def test_GAFF(self):
        ff = GAFF()
        assert ff.gmso_ff is not None

    def test_OPLS_AA(self):
        ff = OPLS_AA()
        assert ff.gmso_ff is not None

    def test_OPLS_AA_PPS(self):
        ff = OPLS_AA_PPS()
        assert ff.gmso_ff is not None

    def test_OPPLS_AA_BENZENE(self):
        ff = OPLS_AA_BENZENE()
        assert ff.gmso_ff is not None

    def test_OPPLS_AA_DIMETHYLETHER(self):
        ff = OPLS_AA_DIMETHYLETHER()
        assert ff.gmso_ff is not None

    def test_FF_from_file(self):
        xml_file = os.path.join(ASSETS_DIR, "test_ff.xml")
        ff = FF_from_file(xml_file)
        assert ff.gmso_ff is not None


class TestCustomForceFields(BaseTest):
    def test_BeadSpring(self):
        ff = BeadSpring(
            r_cut=2.5,
            beads={
                "A": dict(epsilon=1.0, sigma=1.0),
                "B": dict(epsilon=2.0, sigma=2.0),
            },
            bonds={
                "A-A": dict(r0=1.1, k=300),
                "A-B": dict(r0=1.1, k=300),
            },
            angles={"A-A-A": dict(t0=2.0, k=200), "A-B-A": dict(t0=2.0, k=200)},
            dihedrals={"A-A-A-A": dict(phi0=0.0, k=100, d=-1, n=1)},
        )

        assert isinstance(ff.hoomd_forcefield[0], hoomd.md.pair.pair.LJ)
        assert isinstance(ff.hoomd_forcefield[1], hoomd.md.bond.Harmonic)
        assert isinstance(ff.hoomd_forcefield[2], hoomd.md.angle.Harmonic)
        assert isinstance(ff.hoomd_forcefield[3], hoomd.md.dihedral.Periodic)

        pair_types = [("A", "A"), ("A", "B"), ("B", "B")]
        for param in ff.hoomd_forcefield[0].params:
            assert param in pair_types
            if param == ("A", "A"):
                assert ff.hoomd_forcefield[0].params[param]["sigma"] == 1.0
            if param == ("B", "B"):
                assert ff.hoomd_forcefield[0].params[param]["epsilon"] == 2.0
            if param == ("A", "B"):
                assert ff.hoomd_forcefield[0].params[param]["epsilon"] == 1.5

        bond_types = [("A-A"), ("A-B")]
        for param in ff.hoomd_forcefield[1].params:
            assert param in bond_types
            assert ff.hoomd_forcefield[1].params[param]["r0"] == 1.1
            assert ff.hoomd_forcefield[1].params[param]["k"] == 300

        angle_types = [("A-A-A"), ("A-B-A")]
        for param in ff.hoomd_forcefield[2].params:
            assert param in angle_types
            assert ff.hoomd_forcefield[2].params[param]["t0"] == 2.0
            assert ff.hoomd_forcefield[2].params[param]["k"] == 200

        dihedral_types = [("A-A-A-A")]
        for param in ff.hoomd_forcefield[3].params:
            assert param in dihedral_types
            assert ff.hoomd_forcefield[3].params[param]["phi0"] == 0.0
            assert ff.hoomd_forcefield[3].params[param]["k"] == 100
            assert ff.hoomd_forcefield[3].params[param]["d"] == -1
            assert ff.hoomd_forcefield[3].params[param]["n"] == 1

    def test_TorchCustomForce(self, benzene_smiles):
        molecule = Molecule(num_mols=2, smiles=benzene_smiles)
        molecule.coarse_grain(beads={"A": benzene_smiles})
        system = Pack(
            molecules=[molecule],
            density=0.5,
            r_cut=2.5,
        )
        model = NN(
            hidden_dim=32, out_dim=1, n_layers=2, act_fn="Tanh", dropout=0.5
        )
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        custom_force = TorchCustomForce(model)
        sim = Simulation(
            initial_state=system.hoomd_snapshot,
            forcefield=[custom_force],
        )
        sim.run_NVT(n_steps=10, kT=1.0, tau_kt=1.0)
