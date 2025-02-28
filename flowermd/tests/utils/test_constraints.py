import numpy as np
import pytest

from flowermd.base import Pack
from flowermd.library.polymers import EllipsoidChain
from flowermd.tests import BaseTest
from flowermd.utils import create_rigid_body, set_bond_constraints


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
            system.hoomd_snapshot, constrain_value=1.0, bond_type="_C-_H"
        )
        assert snap.constraints.N == (4 * 2 * 2) - 2
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
                system.hoomd_snapshot, constrain_value=2.0, bond_type="_C-_H"
            )


class TestRigidBody(BaseTest):
    @pytest.mark.skip(reason="Not implemented.")
    def test_ellipsoid_create_rigid_body(self):
        ellipsoid_chain = EllipsoidChain(
            lengths=4,
            num_mols=2,
            lpar=0.5,
            bead_mass=100,
            bond_length=0.01,
        )
        system = Pack(
            molecules=ellipsoid_chain,
            density=0.1,
            base_units=dict(),
            fix_orientation=True,
        )

        rigid_frame, rigid = create_rigid_body(
            system.hoomd_snapshot,
            ellipsoid_chain.bead_constituents_types,
            bead_name="R",
        )
        assert rigid_frame.particles.N == 8 + ellipsoid_chain.n_particles
        assert (
            rigid_frame.particles.types
            == ["R"] + system.hoomd_snapshot.particles.types
        )
        assert rigid_frame.particles.mass[0] == 100
        assert np.all(
            np.isclose(
                rigid_frame.particles.position[0],
                np.mean(system.hoomd_snapshot.particles.position[:4], axis=0),
            )
        )

        points = (
            rigid_frame.particles.position[0]
            - system.hoomd_snapshot.particles.position[:4]
        )

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        I_xx = np.sum((y**2 + z**2) * system.hoomd_snapshot.particles.mass[:4])
        I_yy = np.sum((x**2 + z**2) * system.hoomd_snapshot.particles.mass[:4])
        I_zz = np.sum((x**2 + y**2) * system.hoomd_snapshot.particles.mass[:4])
        assert np.all(
            np.isclose(
                rigid_frame.particles.moment_inertia[0],
                np.array((I_xx, I_yy, I_zz)),
            )
        )

        assert np.all(rigid_frame.particles.body[:8] == np.arange(8))
        assert np.all(rigid_frame.particles.body[8:12] == [0])

        assert rigid_frame.bonds.N == 20
        assert rigid_frame.bonds.types == system.hoomd_snapshot.bonds.types
        assert rigid_frame.angles.N == 12
        assert rigid_frame.angles.types == system.hoomd_snapshot.angles.types

        assert "R" in list(rigid.body.keys())
        assert rigid.body["R"]["constituent_types"] == ["A", "A", "B", "B"]
