import gsd.hoomd
import hoomd
import numpy as np

from flowermd.base import Lattice, Molecule, Polymer
from flowermd.library import UFF_DPD
from flowermd.tests import BaseTest
from flowermd.utils import (
    audit_stereochemistry,
    find_ring_spears,
    protect_stereochemistry,
)

PS = "CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C"


class TestStereochemistry(BaseTest):
    def build(self, smiles=None, molecules=None):
        molecules = molecules or Molecule(num_mols=2, smiles=smiles)
        system = Lattice(molecules=molecules, x=1.5, y=1.5, n=1)
        system.apply_forcefield(r_cut=0.35, force_field=UFF_DPD(system), kT=1)
        return system

    def test_protect_and_audit(self):
        system = self.build(PS)
        ref = protect_stereochemistry(system)
        snap = system.hoomd_snapshot
        assert len(ref["centers"]) == 6
        stereo = [t for t in snap.dihedrals.types if t.startswith("stereo_")]
        assert 0 < len(stereo) <= 6
        periodic = [
            f
            for f in system.hoomd_forcefield
            if isinstance(f, hoomd.md.dihedral.Periodic)
        ]
        assert all(t in f.params for f in periodic for t in stereo)
        # the restraint is its own force, zero for every other type
        restraint = system.hoomd_forcefield[-1]
        assert restraint is periodic[-1] and len(periodic) == 2
        for t in snap.dihedrals.types:
            assert (restraint.params[t]["k"] > 0) == (t in stereo)
        assert all(periodic[0].params[t]["k"] == 0 for t in stereo)
        assert audit_stereochemistry(ref, snap)["inverted"] == 0
        snap.particles.position[:, 0] *= -1  # mirror image
        assert audit_stereochemistry(ref, snap)["inverted"] == 6

    def test_polymer_chain(self):
        # mbuild leaves the bonds between repeat units without a bond order
        styrene = Polymer(
            lengths=5,
            num_mols=2,
            smiles="CC(c1ccccc1)",
            bond_indices=[8, 11],
            bond_length=0.154,
            bond_orientation=[None, None],
        )
        system = self.build(molecules=styrene)
        ref = protect_stereochemistry(system)
        assert len(ref["centers"]) == 8  # 4 per chain; one end is benzylic
        audit = audit_stereochemistry(ref, system.hoomd_snapshot)
        assert audit == {"n_centers": 8, "inverted": 0, "near_planar": 0}

    def test_no_centers(self):
        system = self.build("CCCC")
        n_dihedrals = system.hoomd_snapshot.dihedrals.N
        ref = protect_stereochemistry(system)
        assert len(ref["centers"]) == 0
        assert system.hoomd_snapshot.dihedrals.N == n_dihedrals
        assert np.size(ref["sign"]) == 0

    def test_find_ring_spears(self):
        # a hexagon in the xy plane and a C-C bond along z
        angle = np.arange(6) * np.pi / 3
        ring = np.c_[0.14 * np.cos(angle), 0.14 * np.sin(angle), np.zeros(6)]
        frame = gsd.hoomd.Frame()
        frame.particles.N = 8
        frame.configuration.box = [3, 3, 3, 0, 0, 0]
        frame.bonds.N = 7
        frame.bonds.group = [(k, (k + 1) % 6) for k in range(6)] + [(6, 7)]
        for x, z, threaded in [
            (0, 0, 1),
            (0.1, 0, 1),
            (0.2, 0, 0),
            (0, 0.2, 0),
        ]:
            bond = [[x, 0, z + 0.077], [x, 0, z - 0.077]]
            # also across the periodic boundary
            for shift in (0.0, 1.45):
                frame.particles.position = np.r_[ring, bond] + shift
                spears = find_ring_spears(frame)
                assert len(spears) == threaded
        frame.particles.position = np.r_[ring, [[0, 0, 0.077], [0, 0, -0.077]]]
        assert find_ring_spears(frame) == [((6, 7), (4, 3, 2, 1, 0, 5))]
