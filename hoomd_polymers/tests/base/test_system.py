import hoomd
import numpy as np

from hoomd_polymers import Pack, Lattice
from hoomd_polymers.library import OPLS_AA, GAFF, OPLS_AA_PPS, \
    OPLS_AA_DIMETHYLETHER
from hoomd_polymers.tests import BaseTest
import unyt as u

class TestSystem(BaseTest):
    def test_single_mol_type(self, benzene_molecule):
        benzene_mols = benzene_molecule(n_mols=3)
        system = Pack(molecules=[benzene_mols], density=0.8,
                      r_cut=2.5,
                      force_field=OPLS_AA(), auto_scale=True)
        assert system.n_mol_types == 1
        assert len(system.all_molecules) == len(benzene_mols.molecules)
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.hoomd_snapshot.particles.types == ['opls_145', 'opls_146']
        assert system.reference_values.keys() == {'energy', 'length', 'mass'}

    def test_multiple_mol_types(self, benzene_molecule,
                                ethane_molecule):
        benzene_mol = benzene_molecule(n_mols=3)
        ethane_mol = ethane_molecule(n_mols=2)
        system = Pack(
            molecules=[benzene_mol, ethane_mol],
            density=0.8, r_cut=2.5,
            force_field=OPLS_AA(), auto_scale=True)
        assert system.n_mol_types == 2
        assert len(system.all_molecules) == len(benzene_mol.molecules) + \
               len(ethane_mol.molecules)
        assert system.all_molecules[0].name == '0'
        assert system.all_molecules[-1].name == '1'
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.hoomd_snapshot.particles.types == \
               ['opls_135', 'opls_140', 'opls_145', 'opls_146']

    def test_multiple_mol_types_different_ff(self,
                                             dimethylether_molecule,
                                             pps_molecule):
        dimethylether_mol = dimethylether_molecule(n_mols=3)
        pps_mol = pps_molecule(n_mols=2)
        system = Pack(molecules=[dimethylether_mol, pps_mol],
                      density=0.8, r_cut=2.5,
                      force_field=[OPLS_AA_DIMETHYLETHER(), OPLS_AA_PPS()],
                      auto_scale=True)
        assert system.n_mol_types == 2
        assert len(system.all_molecules) == len(
            dimethylether_mol.molecules) + \
               len(pps_mol.molecules)
        assert system.all_molecules[0].name == '0'
        assert system.all_molecules[-1].name == '1'
        assert len(system.hoomd_forcefield) > 0
        for hoomd_force in system.hoomd_forcefield:
            if isinstance(hoomd_force, hoomd.md.pair.LJ):
                pairs = list(hoomd_force.params.keys())
                assert ('os', 'sh') in pairs
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.hoomd_snapshot.particles.types == \
               ['ca', 'ct', 'ha', 'hc', 'hs', 'os', 'sh']

    def test_remove_hydrogen(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=3)
        system = Pack(molecules=[benzene_mol], density=0.8,
                      r_cut=2.5,
                      force_field=OPLS_AA(), auto_scale=True,
                      remove_hydrogens=True)
        assert len(system.hoomd_forcefield) > 0
        assert list(system.hoomd_forcefield[0].params.keys()) == [
            ('opls_145', 'opls_145')]
        assert system.hoomd_snapshot.particles.N == \
               sum(mol.n_particles for mol in benzene_mol.molecules) - 3 * 6
        assert system.hoomd_snapshot.particles.types == ['opls_145']

    def test_target_box(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=3)
        low_density_system = Pack(molecules=[benzene_mol],
                                  density=0.1,
                                  r_cut=2.5,
                                  force_field=OPLS_AA(), auto_scale=True)
        high_density_system = Pack(molecules=[benzene_mol],
                                   density=0.9,
                                   r_cut=2.5,
                                   force_field=OPLS_AA(), auto_scale=True)
        assert all(
            low_density_system.target_box > high_density_system.target_box)

    def test_mass(self, pps_molecule):
        pps_mol = pps_molecule(n_mols=20)
        system = Pack(molecules=[pps_mol], density=1.0, r_cut=2.5)
        assert np.allclose(system.mass,
                           ((12.011 * 6) + (1.008 * 6) + 32.06) * 20, atol=1e-4)

    def test_ref_length(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(molecules=[polyethylene], force_field=[GAFF()],
                      density=1.0, r_cut=2.5, auto_scale=True)

        assert np.allclose(system.reference_length.to('angstrom').value,
                           3.39966951, atol=1e-3)
        reduced_box = system.hoomd_snapshot.configuration.box[0:3]
        calc_box = reduced_box * system.reference_length.to("nm").value
        assert np.allclose(calc_box[0], system.box.Lx, atol=1e-2)
        assert np.allclose(calc_box[1], system.box.Ly, atol=1e-2)
        assert np.allclose(calc_box[2], system.box.Lz, atol=1e-2)

    def test_ref_mass(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(molecules=[polyethylene], force_field=[GAFF()],
                      density=1.0, r_cut=2.5, auto_scale=True)
        total_red_mass = sum(system.hoomd_snapshot.particles.mass)
        assert np.allclose(
            system.mass,
            total_red_mass * system.reference_mass.to("amu").value,
            atol=1e-1
        )

    def test_ref_energy(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(molecules=[polyethylene], force_field=[GAFF()],
                      density=1.0, r_cut=2.5, auto_scale=True)
        assert np.allclose(system.reference_energy.to('kcal/mol').value, 0.1094,
                           atol=1e-3)

    # TODO: test system with base units.
    def test_update_ref_values(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(molecules=[polyethylene], force_field=[OPLS_AA()],
                      density=1.0, r_cut=2.5, auto_scale=False)
        system.reference_length = 1 * u.angstrom
        system.reference_energy = 1 * u.kcal / u.mol
        system.reference_mass = 1 * u.amu
        assert np.allclose(system.hoomd_snapshot.configuration.box[:3],
                           system.gmso_system.box.lengths.to('angstrom').value)


    def test_lattice_polymer(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        Lattice(molecules=[polyethylene], force_field=[OPLS_AA()], density=1.0,
                r_cut=2.5, x=1, y=1, n=4)

    # TODO: asserts for lattice?

    def test_lattice_molecule(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=32)
        Lattice(molecules=[benzene_mol], force_field=OPLS_AA(),
                         density=1.0,
                         r_cut=2.5, x=1, y=1, n=4)
