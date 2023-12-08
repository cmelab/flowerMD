import numpy as np

from flowermd.base import Pack
from flowermd.library.polymers import EllipsoidChain
from flowermd.tests import BaseTest
from flowermd.utils import create_rigid_body


class TestRigidBody(BaseTest):
    def test_ellipsoid_create_rigid_body(self):
        ellipsoid_chain = EllipsoidChain(
            lengths=4,
            num_mols=2,
            bead_length=1,
            bead_mass=100,
            bond_length=0.01,
        )
        system = Pack(
            molecules=ellipsoid_chain, density=0.1, fix_orientation=True
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
        I_xx = np.sum(
            (y**2 + z**2) * system.hoomd_snapshot.particles.mass[:4]
        )
        I_yy = np.sum(
            (x**2 + z**2) * system.hoomd_snapshot.particles.mass[:4]
        )
        I_zz = np.sum(
            (x**2 + y**2) * system.hoomd_snapshot.particles.mass[:4]
        )
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
