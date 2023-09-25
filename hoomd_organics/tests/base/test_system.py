import gmso
import hoomd
import numpy as np
import pytest
import unyt as u
from unyt import Unit

from hoomd_organics import Lattice, Pack
from hoomd_organics.library import OPLS_AA, OPLS_AA_DIMETHYLETHER, OPLS_AA_PPS
from hoomd_organics.tests import BaseTest
from hoomd_organics.utils.exceptions import ReferenceUnitError


class TestSystem(BaseTest):
    def test_single_mol_type(self, benzene_molecule):
        benzene_mols = benzene_molecule(n_mols=3)
        system = Pack(
            molecules=[benzene_mols],
            density=0.8,
            force_field=OPLS_AA(),
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5)
        assert system.n_mol_types == 1
        assert len(system.all_molecules) == len(benzene_mols.molecules)
        assert system.gmso_system.is_typed()
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.hoomd_snapshot.particles.types == ["opls_145", "opls_146"]
        assert system.reference_values.keys() == {"energy", "length", "mass"}

    def test_multiple_mol_types(self, benzene_molecule, ethane_molecule):
        benzene_mol = benzene_molecule(n_mols=3)
        ethane_mol = ethane_molecule(n_mols=2)
        system = Pack(
            molecules=[benzene_mol, ethane_mol],
            density=0.8,
            force_field=OPLS_AA(),
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5)
        assert system.n_mol_types == 2
        assert len(system.all_molecules) == len(benzene_mol.molecules) + len(
            ethane_mol.molecules
        )
        assert system.all_molecules[0].name == "0"
        assert system.all_molecules[-1].name == "1"
        assert system.gmso_system.is_typed()
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.hoomd_snapshot.particles.types == [
            "opls_135",
            "opls_140",
            "opls_145",
            "opls_146",
        ]

    def test_multiple_mol_types_different_ff(
        self, dimethylether_molecule, pps_molecule
    ):
        dimethylether_mol = dimethylether_molecule(n_mols=3)
        pps_mol = pps_molecule(n_mols=2)
        system = Pack(
            molecules=[dimethylether_mol, pps_mol],
            density=0.8,
            force_field=[OPLS_AA_DIMETHYLETHER(), OPLS_AA_PPS()],
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5)
        assert system.n_mol_types == 2
        assert len(system.all_molecules) == len(
            dimethylether_mol.molecules
        ) + len(pps_mol.molecules)
        assert system.all_molecules[0].name == "0"
        assert system.all_molecules[-1].name == "1"
        assert system.gmso_system.is_typed()
        assert len(system.hoomd_forcefield) > 0
        for hoomd_force in system.hoomd_forcefield:
            if isinstance(hoomd_force, hoomd.md.pair.LJ):
                pairs = list(hoomd_force.params.keys())
                assert ("os", "sh") in pairs
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.hoomd_snapshot.particles.types == [
            "ca",
            "ct",
            "ha",
            "hc",
            "hs",
            "os",
            "sh",
        ]

    def test_system_from_mol2_mol_parameterization(self, benzene_molecule_mol2):
        benzene_mol = benzene_molecule_mol2(n_mols=3)
        system = Pack(
            molecules=[benzene_mol],
            density=0.8,
            force_field=OPLS_AA(),
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5)
        assert system.gmso_system.is_typed()
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N

    def test_remove_hydrogen(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=3)
        system = Pack(
            molecules=[benzene_mol],
            density=0.8,
            force_field=OPLS_AA(),
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5, remove_hydrogens=True)
        assert not any(
            [s.element.atomic_number == 1 for s in system.gmso_system.sites]
        )
        assert system.gmso_system.is_typed()
        assert len(system.hoomd_forcefield) > 0
        assert list(system.hoomd_forcefield[0].params.keys()) == [
            ("opls_145", "opls_145")
        ]
        assert (
            system.hoomd_snapshot.particles.N
            == sum(mol.n_particles for mol in benzene_mol.molecules) - 3 * 6
        )
        assert system.hoomd_snapshot.particles.types == ["opls_145"]

    def test_remove_hydrogen_no_atomic_num(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=1)
        system = Pack(
            molecules=[benzene_mol],
            density=0.8,
            force_field=OPLS_AA(),
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5, remove_hydrogens=False)
        for site in system.gmso_system.sites:
            if site.name == "H":
                site.element = gmso.core.element.Element(
                    symbol="C",
                    name="carbon",
                    atomic_number=12,
                    mass=1.008 * Unit("amu"),
                )

        system.remove_hydrogens()
        assert system.gmso_system.n_sites == 6

    def test_remove_hydrogen_no_hydrogen(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=1)
        system = Pack(
            molecules=[benzene_mol],
            density=0.8,
            force_field=OPLS_AA(),
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5, remove_hydrogens=False)
        hydrogens = [
            site
            for site in system.gmso_system.sites
            if site.element.atomic_number == 1
        ]
        for h_site in hydrogens:
            system.gmso_system.remove_site(h_site)

        with pytest.warns():
            system.remove_hydrogens()

    def test_add_mass_charges(self, benzene_molecule):
        benzene_mols = benzene_molecule(n_mols=1)
        system = Pack(
            molecules=[benzene_mols],
            density=0.8,
            force_field=OPLS_AA(),
            auto_scale=False,
        )
        system.apply_forcefield(
            r_cut=2.5, remove_hydrogens=True, scale_charges=False
        )
        for site in system.gmso_system.sites:
            assert site.mass.value == (12.011 + 1.008)
            assert site.charge == 0

        snap = system.hoomd_snapshot
        assert np.allclose(
            sum(snap.particles.mass), 6 * (12.011 + 1.008), atol=1e-4
        )
        assert sum(snap.particles.charge) == 0

    def test_target_box(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=3)
        low_density_system = Pack(
            molecules=[benzene_mol],
            density=0.1,
            force_field=OPLS_AA(),
            auto_scale=True,
        )
        low_density_system.apply_forcefield(r_cut=2.5)

        high_density_system = Pack(
            molecules=[benzene_mol],
            density=0.9,
            force_field=OPLS_AA(),
            auto_scale=True,
        )
        high_density_system.apply_forcefield(r_cut=2.5)
        assert all(
            low_density_system.target_box > high_density_system.target_box
        )

    def test_mass(self, pps_molecule):
        pps_mol = pps_molecule(n_mols=20)
        system = Pack(molecules=[pps_mol], density=1.0)
        assert np.allclose(
            system.mass, ((12.011 * 6) + (1.008 * 6) + 32.06) * 20, atol=1e-4
        )

    def test_ref_length(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5)

        assert np.allclose(
            system.reference_length.to("angstrom").value, 3.5, atol=1e-3
        )
        reduced_box = system.hoomd_snapshot.configuration.box[0:3]
        calc_box = reduced_box * system.reference_length.to("nm").value
        assert np.allclose(calc_box[0], system.box.Lx, atol=1e-2)
        assert np.allclose(calc_box[1], system.box.Ly, atol=1e-2)
        assert np.allclose(calc_box[2], system.box.Lz, atol=1e-2)

    def test_ref_mass(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5)
        total_red_mass = sum(system.hoomd_snapshot.particles.mass)
        assert np.allclose(
            system.mass,
            total_red_mass * system.reference_mass.to("amu").value,
            atol=1e-1,
        )

    def test_ref_energy(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5)
        assert np.allclose(
            system.reference_energy.to("kcal/mol").value, 0.066, atol=1e-3
        )

    def test_rebuild_snapshot(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        assert system._snap_refs == system.reference_values
        assert system._ff_refs == system.reference_values
        init_snap = system.hoomd_snapshot
        new_snap = system.hoomd_snapshot
        assert init_snap == new_snap
        system.reference_length = 1 * u.angstrom
        system.reference_energy = 1 * u.kcal / u.mol
        system.reference_mass = 1 * u.amu
        assert system._snap_refs != system.reference_values
        assert system._ff_refs != system.reference_values
        new_snap = system.hoomd_snapshot
        assert init_snap != new_snap

    def test_ref_values_no_autoscale(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        system.reference_length = 1 * u.angstrom
        system.reference_energy = 1 * u.kcal / u.mol
        system.reference_mass = 1 * u.amu
        assert np.allclose(
            system.hoomd_snapshot.configuration.box[:3],
            system.gmso_system.box.lengths.to("angstrom").value,
        )
        assert dict(system.hoomd_forcefield[3].params)["opls_135", "opls_135"][
            "epsilon"
        ] == system.gmso_system.sites[0].atom_type.parameters["epsilon"].to(
            "kcal/mol"
        )

    def test_set_ref_values(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        ref_value_dict = {
            "length": 1 * u.angstrom,
            "energy": 3.0 * u.kcal / u.mol,
            "mass": 1.25 * u.Unit("amu"),
        }
        system.reference_values = ref_value_dict
        assert system.reference_length == ref_value_dict["length"]
        assert system.reference_energy == ref_value_dict["energy"]
        assert system.reference_mass == ref_value_dict["mass"]

    def test_set_ref_values_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        ref_value_dict = {
            "length": "1 angstrom",
            "energy": "3 kcal/mol",
            "mass": "1.25 amu",
        }
        system.reference_values = ref_value_dict
        assert system.reference_length == 1 * u.angstrom
        assert system.reference_energy == 3 * u.kcal / u.mol
        assert system.reference_mass == 1.25 * u.amu

    def test_set_ref_values_missing_key(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        ref_value_dict = {
            "length": 1 * u.angstrom,
            "energy": 3.0 * u.kcal / u.mol,
        }
        with pytest.raises(ValueError):
            system.reference_values = ref_value_dict

    def test_set_ref_values_invalid_type(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        ref_value_dict = {
            "length": 1 * u.angstrom,
            "energy": 3.0 * u.kcal / u.mol,
            "mass": 1.25,
        }
        with pytest.raises(ReferenceUnitError):
            system.reference_values = ref_value_dict

    def test_set_ref_length(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        system.reference_length = 1 * u.angstrom
        assert system.reference_length == 1 * u.angstrom

    def test_set_ref_length_invalid_type(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_length = 1.0

    def test_ref_length_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        system.reference_length = "1 angstrom"
        assert system.reference_length == 1 * u.angstrom

    def test_ref_length_invalid_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_length = "1.0"

    def test_ref_length_invalid_unit_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_length = "1.0 invalid_unit"

    def test_ref_length_invalid_dimension(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_length = 1.0 * u.g

    def test_ref_length_invalid_dimension_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_length = "1.0 g"

    def test_set_ref_energy(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        system.reference_energy = 1 * u.kcal / u.mol
        assert system.reference_energy == 1 * u.kcal / u.mol

    def test_set_ref_energy_invalid_type(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_energy = 1.0

    def test_ref_energy_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        system.reference_energy = "1 kJ"
        assert system.reference_energy == 1 * u.kJ

    def test_ref_energy_string_comb_units(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        system.reference_energy = "1 kcal/mol"
        assert system.reference_energy == 1 * u.kcal / u.mol

    def test_ref_energy_invalid_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_energy = "1.0"

    def test_ref_energy_invalid_unit_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_energy = "1.0 invalid_unit"

    def test_ref_energy_invalid_dimension(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_energy = 1.0 * u.g

    def test_ref_energy_invalid_dimension_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_energy = "1.0 m"

    def test_set_ref_mass(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)

        system.reference_mass = 1.0 * u.amu
        assert system.reference_mass == 1.0 * u.amu

    def test_set_ref_mass_invalid_type(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_mass = 1.0

    def test_ref_mass_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        system.reference_mass = "1 g"
        assert system.reference_mass == 1.0 * u.g

    def test_ref_mass_invalid_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_mass = "1.0"

    def test_ref_mass_invalid_unit_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_mass = "1.0 invalid_unit"

    def test_ref_mass_invalid_dimension(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_energy = 1.0 * u.m

    def test_ref_mass_invalid_dimension_string(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(r_cut=2.5)
        with pytest.raises(ReferenceUnitError):
            system.reference_mass = "1.0 m"

    def test_apply_forcefield_no_forcefield(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=None,
            density=1.0,
            auto_scale=False,
        )
        with pytest.raises(ValueError):
            system.apply_forcefield(r_cut=2.5)

    def test_forcefield_kwargs_attr(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            auto_scale=False,
        )
        system.apply_forcefield(
            r_cut=2.5, nlist_buffer=0.5, pppm_resolution=(4, 4, 4), pppm_order=3
        )
        assert system._ff_kwargs["r_cut"] == 2.5
        assert system._ff_kwargs["nlist_buffer"] == 0.5
        assert system._ff_kwargs["pppm_kwargs"]["resolution"] == (4, 4, 4)
        assert system._ff_kwargs["pppm_kwargs"]["order"] == 3

    def test_lattice_polymer(self, polyethylene):
        polyethylene = polyethylene(lengths=2, num_mols=32)
        system = Lattice(
            molecules=[polyethylene],
            force_field=[OPLS_AA()],
            density=1.0,
            x=1,
            y=1,
            n=4,
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5)

        assert system.n_mol_types == 1
        assert len(system.all_molecules) == len(polyethylene.molecules)
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.reference_values.keys() == {"energy", "length", "mass"}
        # TODO: specific asserts for lattice system?

    def test_lattice_molecule(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=32)
        system = Lattice(
            molecules=[benzene_mol],
            force_field=OPLS_AA(),
            density=1.0,
            x=1,
            y=1,
            n=4,
            auto_scale=True,
        )
        system.apply_forcefield(r_cut=2.5)
        assert system.n_mol_types == 1
        assert len(system.all_molecules) == len(benzene_mol.molecules)
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.reference_values.keys() == {"energy", "length", "mass"}

    def test_scale_charges(self, pps):
        pps_mol = pps(num_mols=5, lengths=5)
        no_scale = Pack(
            molecules=pps_mol,
            density=0.5,
            force_field=OPLS_AA_PPS(),
            auto_scale=True,
        )
        no_scale.apply_forcefield(r_cut=2.5, scale_charges=False)

        with_scale = Pack(
            molecules=pps_mol,
            density=0.5,
            force_field=OPLS_AA_PPS(),
            auto_scale=True,
        )
        with_scale.apply_forcefield(r_cut=2.5, scale_charges=True)
        assert abs(no_scale.net_charge.value) > abs(with_scale.net_charge.value)
        assert np.allclose(0, with_scale.net_charge.value, atol=1e-30)
