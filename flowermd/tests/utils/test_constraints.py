import numpy as np
import pytest

from flowermd.base import Pack
from flowermd.library.polymers import EllipsoidChain
from flowermd.tests import BaseTest
from flowermd.utils import create_rigid_ellipsoid_chain, set_bond_constraints


class TestBondConstraint(BaseTest):
    def test_ellipsoid_fixed_bonds(self):
        ellipsoid_chain = EllipsoidChain(
            lengths=4,
            num_mols=2,
            lpar=1,
            bead_mass=50,
        )
        system = Pack(
            molecules=ellipsoid_chain,
            density=0.1,
            base_units=dict(),
        )
        snap, d = set_bond_constraints(
            system.hoomd_snapshot, constraint_values=[1.0], bond_types=["_C-_H"]
        )
        assert snap.constraints.N == (4 * 2 * 2) - 2
        for group in snap.bonds.group:
            assert group in snap.constraints.group
        assert d.tolerance == 1e-5
        assert all([val == 1.0 for val in snap.constraints.value])

    def test_ellipsoid_fixed_bonds_bad_val(self):
        ellipsoid_chain = EllipsoidChain(
            lengths=4,
            num_mols=2,
            lpar=1,
            bead_mass=50,
        )
        system = Pack(
            molecules=ellipsoid_chain,
            density=0.1,
            base_units=dict(),
        )
        with pytest.raises(ValueError):
            set_bond_constraints(
                system.hoomd_snapshot,
                constraint_values=[2.0],
                bond_types=["_C-_H"],
            )


class TestRigidBody(BaseTest):
    def test_ellipsoid_create_rigid_body(self):
        chains = EllipsoidChain(
            lengths=4,
            num_mols=2,
            lpar=1.0,
            bead_mass=1,
            bond_L=0.0,
        )
        system = Pack(
            molecules=chains,
            density=0.0001,
            edge=5,
            overlap=1,
            fix_orientation=True,
        )
        snap = system.hoomd_snapshot
        rigid_frame, rigid = create_rigid_ellipsoid_chain(snap)
        assert rigid_frame.particles.N == 8 + chains.n_particles
        assert rigid_frame.particles.types == ["R"] + snap.particles.types
        assert rigid_frame.particles.mass[0] == 1

        center_idx = snap.particles.types.index("X")
        rigid_idx = rigid_frame.particles.types.index("R")
        center_indices = np.where(snap.particles.typeid == center_idx)[0]
        rigid_indices = np.where(rigid_frame.particles.typeid == rigid_idx)[0]
        center_pos = snap.particles.position[center_indices]
        rigid_pos = rigid_frame.particles.position[rigid_indices]
        for pos1, pos2 in zip(center_pos, rigid_pos):
            assert np.all(np.isclose(pos1, pos2))

        assert np.all(
            np.isclose(
                rigid_frame.particles.moment_inertia[0],
                np.array((0, 2, 2)),
            )
        )

        assert np.all(rigid_frame.particles.body[:8] == np.arange(8))
        assert np.all(rigid_frame.particles.body[8:12] == [0])

        assert rigid_frame.bonds.N == 20
        assert rigid_frame.bonds.types == system.hoomd_snapshot.bonds.types
        assert rigid_frame.angles.N == 12
        assert rigid_frame.angles.types == system.hoomd_snapshot.angles.types

        assert "R" in list(rigid.body.keys())
        assert rigid.body["R"]["constituent_types"] == ["X", "A", "T", "T"]
