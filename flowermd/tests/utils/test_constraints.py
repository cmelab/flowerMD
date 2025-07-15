import numpy as np
import pytest

from flowermd.base import Pack
from flowermd.library import LJChain
from flowermd.library.polymers import EllipsoidChain
from flowermd.tests import BaseTest
from flowermd.utils import create_rigid_ellipsoid_chain, set_bond_constraints


class TestBondConstraint(BaseTest):
    def test_single_fixed_bonds(self):
        chains = LJChain(lengths=10, num_mols=1)
        system = Pack(molecules=chains, density=0.001, base_units=dict())
        snap, d = set_bond_constraints(
            snapshot=system.hoomd_snapshot,
            bond_types=["A-A"],
            constraint_values=[1.0],
        )
        assert snap.constraints.N == 9
        for group in snap.bonds.group:
            assert group in snap.constraints.group
        assert d.tolerance == 1e-5
        assert all([val == 1.0 for val in snap.constraints.value])

    def test_multiple_fixed_bonds(self):
        chains = LJChain(
            lengths=15,
            num_mols=1,
            bead_sequence=["A", "B", "B"],
            bead_mass={"A": 1.0, "B": 0.7},
            bond_lengths={"A-B": 1.0, "B-B": 0.80},
        )
        system = Pack(molecules=chains, density=0.0001, base_units=dict())
        snap, d = set_bond_constraints(
            snapshot=system.hoomd_snapshot,
            bond_types=["A-B", "B-B"],
            constraint_values=[1.0, 0.8],
        )
        assert snap.constraints.N == 44
        for group in snap.bonds.group:
            assert group in snap.constraints.group

        ab_index = snap.bonds.types.index("A-B")
        bb_index = snap.bonds.types.index("B-B")
        for group, val in zip(snap.constraints.group, snap.constraints.value):
            group_index = snap.bonds.group.index(group)
            bond_id = snap.bonds.typeid[group_index]
            if bond_id == bb_index:
                assert np.allclose(val, 0.80)
            elif bond_id == ab_index:
                assert np.allclose(val, 1.0)

    def test_ellipsoid_fixed_bonds_bad_val(self):
        chains = LJChain(lengths=10, num_mols=1)
        system = Pack(molecules=chains, density=0.001, base_units=dict())
        with pytest.raises(ValueError):
            set_bond_constraints(
                system.hoomd_snapshot,
                constraint_values=[2.0],
                bond_types=["A-A"],
            )


class TestRigidBody(BaseTest):
    def test_ellipsoid_create_rigid_body(self):
        LPAR = 1.0
        LPERP = 0.5
        BEAD_MASS = 1.0

        chains = EllipsoidChain(
            lengths=4,
            num_mols=2,
            lpar=LPAR,
            bead_mass=BEAD_MASS,
            bond_L=0.0,
        )
        system = Pack(
            molecules=chains,
            density=0.0001,
            edge=5,
            overlap=1,
            fix_orientation=True,
            base_units=dict(),
        )
        snap = system.hoomd_snapshot
        rigid_frame, rigid = create_rigid_ellipsoid_chain(snap, LPERP)
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

        Ixx = BEAD_MASS / 5 * (LPAR * LPAR + LPERP * LPERP)
        Iyy = Ixx  # both a and b axes are the same

        assert np.all(
            np.isclose(
                rigid_frame.particles.moment_inertia[0],
                np.array((Ixx, Iyy, 0)),
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
