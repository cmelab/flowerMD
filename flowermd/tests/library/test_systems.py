import numpy as np
import pytest
import unyt as u

from flowermd.library import (
    EllipsoidChainRand,
    LJChain,
    PolyEthylene,
    RandomWalk,
    SingleChainSystem,
    mbuildSystem,
)
from flowermd.base import Molecule
from flowermd.tests import BaseTest

class TestSystems(BaseTest):
    def test_single_chain_buffer(self):
        chain = PolyEthylene(lengths=10, num_mols=1)
        system = SingleChainSystem(molecules=chain, buffer=1.05)
        chain2 = PolyEthylene(lengths=10, num_mols=1)
        system2 = SingleChainSystem(molecules=chain2, buffer=2.10)

        assert np.allclose(system.box.Lx * 2, system2.box.Lx, atol=1e-3)
        assert np.allclose(system.box.Ly * 2, system2.box.Ly, atol=1e-3)
        assert np.allclose(system.box.Lz * 2, system2.box.Lz, atol=1e-3)

    def test_single_chain_lengths(self):
        lj_chain = LJChain(lengths=10, num_mols=1)
        system = SingleChainSystem(molecules=lj_chain, buffer=1.05)
        chain = lj_chain._molecules[0]
        chain_length = np.linalg.norm(
            chain.children[-1].pos - chain.children[0].pos
        )
        assert np.allclose(chain_length * 1.05, system.box.Lx, atol=1e-3)

    def test_multiple_molecules(self):
        lj_chain = LJChain(lengths=10, num_mols=2)
        with pytest.raises(ValueError):
            SingleChainSystem(molecules=lj_chain, buffer=1.05)

    def test_ell_rand_walk(self):
        chains = EllipsoidChainRand(
            lengths=10, num_mols=10, lpar=0.5, bead_mass=1.0, density=0.85
        )
        system = mbuildSystem(molecules=chains)
        assert system.box
        assert system.n_particles == 400  # 10 x 10 x 4

    def test_RandomWalk(self):
        lj_chain = LJChain(
            lengths=[10],
            num_mols=[20],
            bead_sequence=["_A"],
            bead_mass={"_A": 1.0},
            bond_lengths={"_A-_A": 1.0},
            identify_connections=False
        )
        bond_length=1.0
        ref_length = 1.0 * u.Unit("nm")
        ref_mass = 1.0 * u.Unit("g/mol")
        ref_energy = 1.0 * u.Unit("kcal / mol")
        ref_values_dict = {"length": ref_length, "mass": ref_mass, "energy": ref_energy}
        system = RandomWalk(
            molecules=lj_chain,
            buffer=0.2,
            bond_length=bond_length,
            density=1.0 * u.Unit("nm**-3"),
            base_units=ref_values_dict,
        )
        snapshot = system.hoomd_snapshot
        positions = np.asarray(snapshot.particles.position)
        box = np.array(snapshot.configuration.box[:3])
        bond_group = np.asarray(snapshot.bonds.group) 

        r_ij = positions[bond_group[:, 0]] - positions[bond_group[:, 1]]
        r_ij -= box * np.round(r_ij / box)
        bond_lengths = np.linalg.norm(r_ij, axis=1)

        np.testing.assert_allclose(
            bond_lengths,
            bond_length,
            atol=1e-3,
            err_msg=(
                f"Bond length deviates from target {bond_length}. "
                f"min={bond_lengths.min():.6f}, max={bond_lengths.max():.6f}, "
                f"n_bad={np.sum(~np.isclose(bond_lengths, bond_length, atol=1e-3))}"
            ),
        )
  
    def test_RandomWalk_different_seeds(self):
        lj_chain1 = LJChain(lengths=10, num_mols=10)
        lj_chain2 = LJChain(lengths=10, num_mols=10)
        system1 = RandomWalk(
            molecules=lj_chain1,
            buffer=0.2,
            bond_length=1.0,
            density=1.1,
            seed=1234,
        )
        system2 = RandomWalk(
            molecules=lj_chain2,
            buffer=0.2,
            bond_length=1.0,
            density=1.1,
            seed=4567,
        )
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                system1.gmso_system.positions, system2.gmso_system.positions
            )
