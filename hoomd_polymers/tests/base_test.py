import os

import hoomd
import mbuild as mb
import pytest
from gmso.external.convert_mbuild import from_mbuild

from hoomd_polymers import Molecule, Polymer, Pack
from hoomd_polymers.library import OPLS_AA
# from hoomd_polymers.systems import *
# from hoomd_polymers.molecules import *
# from hoomd_polymers.forcefields import *
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


class BaseTest:
    # @pytest.fixture(autouse=True)
    # def initdir(self, tmpdir):
    #     tmpdir.chdir()
    #
    # @pytest.fixture()
    # def polyethylene_system(self):
    #     system = Pack(
    #             molecule=PolyEthylene,
    #             n_mols=5,
    #             mol_kwargs={"length": 5},
    #             density=0.5
    #     )
    #     system.apply_forcefield(forcefield=GAFF(), remove_hydrogens=False)
    #     return system
    #
    # @pytest.fixture()
    # def ua_polyethylene_system(self):
    #     system = Pack(
    #             molecule=PolyEthylene,
    #             n_mols=5,
    #             mol_kwargs={"length": 5},
    #             density=0.5
    #     )
    #     system.apply_forcefield(forcefield=GAFF(), remove_hydrogens=True)
    #     return system

    @pytest.fixture()
    def benzene_smiles(self):
        return "c1ccccc1"

    @pytest.fixture()
    def dimethylether_smiles(self):
        return "COC"

    @pytest.fixture()
    def ethane_smiles(self):
        return "CC"

    @pytest.fixture()
    def octane_smiles(self):
        return "CCCCCCCC"

    @pytest.fixture()
    def pps_smiles(self):
        return "c1ccc(S)cc1"

    @pytest.fixture(autouse=True)
    def benzene_mb(self, benzene_smiles):
        benzene = mb.load(benzene_smiles, smiles=True)
        return benzene

    @pytest.fixture()
    def benzene_mol2(self):
        return os.path.join(ASSETS_DIR, "benzene.mol2")

    @pytest.fixture()
    def benzene_gmso(self, benzene_mb):
        topology = from_mbuild(benzene_mb)
        topology.identify_connections()
        return topology

    @pytest.fixture()
    def benzene_xml(self):
        return os.path.join(ASSETS_DIR, "benzene_oplsaa.xml")

    def benzene_hoomd_pair(self, include_hydrogen=True, invalid_pair=False):
        cell = hoomd.md.nlist.Cell(buffer=0.4)
        lj = hoomd.md.pair.LJ(nlist=cell)
        if invalid_pair:
            lj.params[('C', 'N')] = dict(epsilon=0.35, sigma=0.29)
            lj.r_cut[('C', 'N')] = 2.5
        else:
            lj.params[('C', 'C')] = dict(epsilon=0.35, sigma=0.29)
            lj.r_cut[('C', 'C')] = 2.5
        if include_hydrogen:
            lj.params[('C', 'H')] = dict(epsilon=0.35, sigma=0.29)
            lj.r_cut[('C', 'H')] = 2.5
            lj.params[('H', 'H')] = dict(epsilon=0.35, sigma=0.65)
            lj.r_cut[('H', 'H')] = 2.5

        return lj

    def benzene_hoomd_bond(self, include_hydrogen=True):
        bond = hoomd.md.bond.Harmonic()
        bond.params['C-C'] = dict(k=3.0, r0=2.38)
        if include_hydrogen:
            bond.params['C-H'] = dict(k=3.0, r0=2.38)
        return bond

    def benzene_hoomd_angle(self, include_hydrogen=True):
        angle = hoomd.md.angle.Harmonic()
        angle.params['C-C-C'] = dict(k=3.0, t0=0.7851)
        if include_hydrogen:
            angle.params['C-C-H'] = dict(k=3.0, t0=0.7851)
        return angle

    def benzene_hoomd_dihedral(self, include_hydrogen=True):
        harmonic = hoomd.md.dihedral.Periodic()
        harmonic.params['C-C-C-C'] = dict(k=3.0, d=0, n=1)
        if include_hydrogen:
            harmonic.params['C-C-C-H'] = dict(k=3.0, d=0, n=1)
            harmonic.params['H-C-C-H'] = dict(k=3.0, d=-1, n=3, phi0=0)
        return harmonic

    @pytest.fixture()
    def benzene_hoomd_ff(self):
        def _hoomd_ff(include_hydrogen, invalid_pair=False):
            pairs = self.benzene_hoomd_pair(include_hydrogen=include_hydrogen,
                                            invalid_pair=invalid_pair)
            bonds = self.benzene_hoomd_bond(include_hydrogen=include_hydrogen)
            angles = self.benzene_hoomd_angle(include_hydrogen=include_hydrogen)
            dihedrals = self.benzene_hoomd_dihedral(
                include_hydrogen=include_hydrogen)
            return [pairs, bonds, angles, dihedrals]

        return _hoomd_ff

    @pytest.fixture()
    def benzene_molecule(self, benzene_smiles):
        def _benzene_molecule(n_mols):
            benzene = Molecule(num_mols=n_mols, smiles=benzene_smiles)
            return benzene

        return _benzene_molecule

    @pytest.fixture()
    def ethane_molecule(self, ethane_smiles):
        def _ethane_molecule(n_mols):
            ethane = Molecule(num_mols=n_mols, smiles=ethane_smiles)
            return ethane

        return _ethane_molecule

    @pytest.fixture()
    def pps_molecule(self, pps_smiles):
        def _pps_molecule(n_mols):
            pps = Molecule(num_mols=n_mols, smiles=pps_smiles)
            return pps

        return _pps_molecule

    @pytest.fixture()
    def dimethylether_molecule(self, dimethylether_smiles):
        def _dimethylether_molecule(n_mols):
            dimethylether = Molecule(num_mols=n_mols,
                                     smiles=dimethylether_smiles)
            return dimethylether

        return _dimethylether_molecule

    @pytest.fixture()
    def polyethylene(self, ethane_smiles):
        class _PolyEthylene(Polymer):
            def __init__(self, lengths, num_mols, **kwargs):
                smiles = ethane_smiles
                bond_indices = [2, -2]
                bond_length = 0.15
                bond_orientation = [None, None]
                super().__init__(
                    lengths=lengths, num_mols=num_mols,
                    smiles=smiles, bond_indices=bond_indices,
                    bond_length=bond_length, bond_orientation=bond_orientation,
                    **kwargs)

        return _PolyEthylene

    @pytest.fixture()
    def polyDME(self, dimethylether_smiles):
        class _PolyDME(Polymer):
            def __init__(self, lengths, num_mols, **kwargs):
                smiles = dimethylether_smiles
                bond_indices = [3, -1]
                bond_length = 0.15
                bond_orientation = [None, None]
                super().__init__(
                    lengths=lengths, num_mols=num_mols,
                    smiles=smiles, bond_indices=bond_indices,
                    bond_length=bond_length, bond_orientation=bond_orientation,
                    **kwargs)

        return _PolyDME

    @pytest.fixture()
    def benzene_system(self, benzene_mb):
        benzene = Molecule(num_mols=5, compound=benzene_mb)
        system = Pack(molecules=[benzene], density=0.5,
                      r_cut=2.5,
                      force_field=OPLS_AA(), auto_scale=True)
        return system
