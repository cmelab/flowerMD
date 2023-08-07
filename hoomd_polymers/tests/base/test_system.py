from hoomd_polymers.tests import BaseTest
from hoomd_polymers.library import OPLS_AA, OPLS_AA_PPS, OPLS_AA_DIMETHYLETHER
from hoomd_polymers import Pack
import hoomd


class TestSystem(BaseTest):
    def test_pack_system_single_mol_type(self, benzene_molecule):
        system = Pack(molecules=[benzene_molecule], density=0.8, r_cut=2.5,
                      force_field=OPLS_AA(), auto_scale=True)
        assert system.n_mol_types == 1
        assert len(system.all_molecules) == len(benzene_molecule.molecules)
        assert len(system.hoomd_forcefield) > 0
        assert system.hoomd_snapshot is not None
        assert system.hoomd_snapshot.particles.types == ['opls_145', 'opls_146']
        assert system.reference_values.keys() == {'energy', 'length', 'mass'}

    def test_pack_system_multiple_mol_types(self, benzene_molecule,
                                            ethane_molecule):
        system = Pack(molecules=[benzene_molecule, ethane_molecule],
                      density=0.8, r_cut=2.5,
                      force_field=OPLS_AA(), auto_scale=True)
        assert system.n_mol_types == 2
        assert len(system.all_molecules) == len(benzene_molecule.molecules) + \
               len(ethane_molecule.molecules)
        assert system.all_molecules[0].name == '0'
        assert system.all_molecules[-1].name == '1'
        assert len(system.hoomd_forcefield) > 0
        assert system.hoomd_snapshot is not None
        assert system.hoomd_snapshot.particles.types == \
               ['opls_135', 'opls_140', 'opls_145', 'opls_146']

    def test_pack_system_multiple_mol_types_different_ff(self,
                                                         dimethylether_molecule,
                                                         pps_molecule):
        system = Pack(molecules=[dimethylether_molecule, pps_molecule],
                      density=0.8, r_cut=2.5,
                      force_field=[OPLS_AA_DIMETHYLETHER(), OPLS_AA_PPS()],
                      auto_scale=True)
        assert system.n_mol_types == 2
        assert len(system.all_molecules) == len(
            dimethylether_molecule.molecules) + \
               len(pps_molecule.molecules)
        assert system.all_molecules[0].name == '0'
        assert system.all_molecules[-1].name == '1'
        assert len(system.hoomd_forcefield) > 0
        for hoomd_force in system.hoomd_forcefield:
            if isinstance(hoomd_force, hoomd.md.pair.LJ):
                pairs = list(hoomd_force.params.keys())
                assert ('os', 'sh') in pairs
        assert system.hoomd_snapshot is not None
        assert system.hoomd_snapshot.particles.types == \
               ['ca', 'ct', 'ha', 'hc', 'hs', 'os', 'sh']

    def test_pack_system_remove_hydrogen(self, benzene_molecule):
        system = Pack(molecules=[benzene_molecule], density=0.8, r_cut=2.5,
                      force_field=OPLS_AA(), auto_scale=True,
                      remove_hydrogens=True)
        assert len(system.hoomd_forcefield) > 0
        assert list(system.hoomd_forcefield[0].params.keys()) == [
            ('opls_145', 'opls_145')]
        assert system.hoomd_snapshot is not None
        assert system.hoomd_snapshot.particles.types == ['opls_145']

    def test_pack_system_target_box(self, benzene_molecule):
        low_density_system = Pack(molecules=[benzene_molecule], density=0.1, r_cut=2.5,
                      force_field=OPLS_AA(), auto_scale=True)
        high_density_system = Pack(molecules=[benzene_molecule], density=0.9, r_cut=2.5,
                      force_field=OPLS_AA(), auto_scale=True)
        assert all(low_density_system.target_box > high_density_system.target_box)