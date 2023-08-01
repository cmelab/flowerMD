import pytest

from hoomd_polymers import Molecule
from hoomd_polymers.tests import BaseTest
from hoomd_polymers.utils import FF_Types, exceptions


class TestMolecule(BaseTest):
    def test_molecule_from_mb_compound(self, benzene_mb):
        molecule = Molecule(num_mols=2, compound=benzene_mb)
        assert len(molecule.molecules) == 2

    def test_molecule_from_gmso_topology(self, benzene_gmso):
        molecule = Molecule(num_mols=2, compound=benzene_gmso)
        assert len(molecule.molecules) == 2

    def test_molecule_from_smiles(self, benzene_smiles):
        molecule = Molecule(num_mols=2, smiles=benzene_smiles)
        assert len(molecule.molecules) == 2

    def test_molecule_from_file(self, benzene_mol2):
        molecule = Molecule(num_mols=2, file=benzene_mol2)
        assert len(molecule.molecules) == 2

    def test_molecule_topology_benzene(self, benzene_mb):
        molecule = Molecule(num_mols=2, compound=benzene_mb)
        assert set(molecule.topology_information["particle_types"]) == {'C',
                                                                        'H'}
        assert (set(molecule.topology_information["pair_types"]) == {('C', 'C'),
                                                                     ('C', 'H'),
                                                                     (
                                                                         'H',
                                                                         'H')})
        assert len(set(molecule.topology_information["particle_typeid"])) == 2
        assert len(molecule.topology_information["bond_types"]) == 2
        assert len(molecule.topology_information["angle_types"]) == 2
        assert len(molecule.topology_information["dihedral_types"]) == 3
        assert not any(molecule.topology_information["particle_charge"])

    def test_validate_force_field_oplsaa(self, benzene_mb):
        molecule = Molecule(num_mols=2, force_field="oplsaa",
                            compound=benzene_mb)
        assert molecule.ff_type == FF_Types.oplsaa
        assert set(molecule.topology_information["particle_types"]) == {
            'opls_145', 'opls_146'}
        assert any(molecule.topology_information["particle_charge"])

    def test_validate_force_field_xml_file(self, benzene_mb):
        molecule = Molecule(num_mols=2, force_field="oplsaa.xml",
                            compound=benzene_mb)
        assert molecule.ff_type == FF_Types.oplsaa
        assert set(molecule.topology_information["particle_types"]) == {
            'opls_145', 'opls_146'}
        assert any(molecule.topology_information["particle_charge"])

    def test_validate_force_field_xml_file_path(self, benzene_mb, benzene_xml):
        molecule = Molecule(num_mols=2, force_field=benzene_xml,
                            compound=benzene_mb)
        assert molecule.ff_type == FF_Types.custom
        assert set(molecule.topology_information["particle_types"]) == {
            'opls_145', 'opls_146'}
        assert any(molecule.topology_information["particle_charge"])

    def test_validate_force_field_not_xml_file(self, benzene_mb):
        with pytest.raises(ValueError):
            molecule = Molecule(num_mols=2, force_field="oplsaa.txt",
                                compound=benzene_mb)

    def test_validate_force_field_not_supported(self, benzene_mb):
        with pytest.raises(ValueError):
            molecule = Molecule(num_mols=2, force_field="oplsaa2",
                                compound=benzene_mb)

    def test_validate_force_field_invalid_xml_file(self, benzene_mb):
        with pytest.raises(ValueError):
            molecule = Molecule(num_mols=2, force_field="oplsaa2.xml",
                                compound=benzene_mb)

    def test_validate_force_field_hoomd_ff_aa(self, benzene_mb,
                                              benzene_hoomd_ff):
        hoomd_ff = benzene_hoomd_ff(include_hydrogen=True)
        molecule = Molecule(num_mols=2, force_field=hoomd_ff,
                            compound=benzene_mb)
        assert molecule.ff_type == FF_Types.Hoomd

    def test_validate_fore_field_hoomd_ff_ua(self, benzene_mb,
                                             benzene_hoomd_ff):
        hoomd_ff = benzene_hoomd_ff(include_hydrogen=False)
        molecule = Molecule(num_mols=2, force_field=hoomd_ff,
                            compound=benzene_mb)
        assert molecule.ff_type == FF_Types.Hoomd

    def test_validate_force_field_hoomd_ff_missing_pair(self, benzene_mb,
                                                        benzene_hoomd_ff):
        hoomd_ff = benzene_hoomd_ff(include_hydrogen=True)
        hoomd_ff.pop(0)
        with pytest.raises(exceptions.MissingPairPotentialError):
            molecule = Molecule(num_mols=2, force_field=hoomd_ff,
                                compound=benzene_mb)

    def test_validate_force_field_hoomd_ff_missing_bond(self, benzene_mb,
                                                        benzene_hoomd_ff):
        hoomd_ff = benzene_hoomd_ff(include_hydrogen=True)
        hoomd_ff.pop(1)
        with pytest.raises(exceptions.MissingBondPotentialError):
            molecule = Molecule(num_mols=2, force_field=hoomd_ff,
                                compound=benzene_mb)

    def test_validate_force_field_hoomd_ff_invalid_pair(self, benzene_mb,
                                                        benzene_hoomd_ff):
        hoomd_ff = benzene_hoomd_ff(include_hydrogen=True, invalid_pair=True)
        with pytest.raises(exceptions.MissingPairPotentialError):
            molecule = Molecule(num_mols=2,
                                force_field=hoomd_ff,
                                compound=benzene_mb)

    def test_validate_force_field_hoomd_ff_missing_Coulomb(self, benzene_mb,
                                                           benzene_xml,
                                                           benzene_hoomd_ff):
        hoomd_ff = benzene_hoomd_ff(include_hydrogen=True)
        typed_molecule = Molecule(num_mols=2, force_field=benzene_xml,
                                  compound=benzene_mb)
        with pytest.raises(exceptions.MissingCoulombPotentialError):
            molecule = Molecule(num_mols=2, force_field=hoomd_ff,
                                compound=typed_molecule.gmso_molecule)
