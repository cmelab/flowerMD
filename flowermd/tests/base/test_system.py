import os
import pickle

import gmso
import hoomd
import numpy as np
import pytest
from cmeutils.geometry import get_backbone_vector
from unyt import Unit

from flowermd import Lattice, Pack, Units
from flowermd.internal.exceptions import ForceFieldError, UnitError
from flowermd.library import (
    OPLS_AA,
    OPLS_AA_DIMETHYLETHER,
    OPLS_AA_PPS,
    LJChain,
)
from flowermd.tests import BaseTest


class TestSystem(BaseTest):
    def test_single_mol_type(self, benzene_molecule):
        benzene_mols = benzene_molecule(n_mols=3)
        system = Pack(molecules=[benzene_mols], density=0.8)
        system.apply_forcefield(
            r_cut=2.5, force_field=OPLS_AA(), auto_scale=True
        )
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
        system = Pack(molecules=[benzene_mol, ethane_mol], density=0.8)
        system.apply_forcefield(
            r_cut=2.5, force_field=OPLS_AA(), auto_scale=True
        )
        assert system.n_mol_types == 2
        assert len(system.all_molecules) == len(benzene_mol.molecules) + len(
            ethane_mol.molecules
        )
        assert system.gmso_system.sites[0].group == "0"
        assert system.gmso_system.sites[-1].group == "1"
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
        system = Pack(molecules=[dimethylether_mol, pps_mol], density=0.8)
        system.apply_forcefield(
            r_cut=2.5,
            force_field=[OPLS_AA_DIMETHYLETHER(), OPLS_AA_PPS()],
            auto_scale=True,
        )
        assert system.n_mol_types == 2
        assert len(system.all_molecules) == len(
            dimethylether_mol.molecules
        ) + len(pps_mol.molecules)
        assert system.gmso_system.sites[0].group == "0"
        assert system.gmso_system.sites[-1].group == "1"
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
        system = Pack(molecules=[benzene_mol], density=0.8)
        system.apply_forcefield(
            r_cut=2.5, force_field=OPLS_AA(), auto_scale=True
        )
        assert system.gmso_system.is_typed()
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N

    def test_remove_hydrogen(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=3)
        system = Pack(molecules=[benzene_mol], density=0.8)
        system.apply_forcefield(
            r_cut=2.5,
            force_field=OPLS_AA(),
            remove_hydrogens=True,
            auto_scale=True,
        )
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
        )
        system.apply_forcefield(
            r_cut=2.5,
            force_field=OPLS_AA(),
            remove_hydrogens=False,
            auto_scale=True,
        )
        for site in system.gmso_system.sites:
            # this does not work as the name of all sites are changed to reflect
            # the molecule type id
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
        )
        system.apply_forcefield(
            r_cut=2.5,
            force_field=OPLS_AA(),
            remove_hydrogens=False,
            auto_scale=True,
        )
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
        system = Pack(molecules=[benzene_mols], density=0.8, base_units=dict())
        system.apply_forcefield(
            r_cut=2.5,
            force_field=OPLS_AA(),
            remove_hydrogens=True,
            scale_charges=False,
            auto_scale=False,
        )
        for site in system.gmso_system.sites:
            assert site.mass.value == (12.011 + 1.008)
            assert site.charge == 0

        snap = system.hoomd_snapshot
        assert np.allclose(
            sum(snap.particles.mass), 6 * (12.011 + 1.008), atol=1e-4
        )
        assert sum(snap.particles.charge) == 0

    def test_pack_box(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=3)
        low_density_system = Pack(molecules=[benzene_mol], density=0.1)
        high_density_system = Pack(molecules=[benzene_mol], density=0.9)
        assert np.prod(low_density_system.box.lengths) > np.prod(
            high_density_system.box.lengths
        )

    def test_pack_seed(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=3)
        default_seed = Pack(molecules=[benzene_mol], density=0.1)
        change_seed = Pack(molecules=[benzene_mol], density=0.1, seed=12340)
        assert not np.array_equal(
            default_seed.system.xyz, change_seed.system.xyz
        )

    # adding test for kwargs argument in system.py Pack class
    def test_pack_kwargs_attr(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system1 = Pack(
            molecules=[polyethylene],
            density=1.0,
            overlap=0.2,
            seed=12345,
            fix_orientation=True,
        )
        system2 = Pack(
            molecules=[polyethylene],
            density=1.0,
            overlap=0.2,
            seed=12345,
            fix_orientation=False,
        )
        assert not np.array_equal(system1.system.xyz, system2.system.xyz)

    def test_mass(self, pps_molecule):
        pps_mol = pps_molecule(n_mols=20)
        system = Pack(molecules=[pps_mol], density=1.0)
        assert np.allclose(
            system.mass.value,
            ((12.011 * 6) + (1.008 * 6) + 32.06) * 20,
            atol=1e-4,
        )

    def test_ref_length(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(molecules=[polyethylene], density=1.0, base_units=dict())
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )

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
        system = Pack(molecules=[polyethylene], density=1.0, base_units=dict())
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )
        total_red_mass = sum(system.hoomd_snapshot.particles.mass)
        assert np.allclose(
            system.mass.value,
            total_red_mass * system.reference_mass.to("amu").value,
            atol=1e-1,
        )

    def test_ref_energy(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(molecules=[polyethylene], density=1.0, base_units=dict())
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )
        assert np.allclose(
            system.reference_energy.to("kcal/mol").value, 0.066, atol=1e-3
        )

    def test_rebuild_snapshot(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        assert system._snap_refs == system.reference_values
        assert system._ff_refs == system.reference_values
        init_snap = system.hoomd_snapshot
        new_snap = system.hoomd_snapshot
        assert init_snap == new_snap
        system.reference_length = 1 * Units.angstrom
        system.reference_energy = 1 * Units.kcal_mol
        system.reference_mass = 1 * Units.amu
        assert system._snap_refs != system.reference_values
        assert system._ff_refs != system.reference_values
        new_snap = system.hoomd_snapshot
        assert init_snap != new_snap

    def test_ref_values_no_autoscale(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        system.reference_length = 1 * Units.angstrom
        system.reference_energy = 1 * Units.kcal_mol
        system.reference_mass = 1 * Units.amu
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
        system = Pack(molecules=[polyethylene], density=1.0)
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        ref_value_dict = {
            "length": 1 * Units.angstrom,
            "energy": 3.0 * Units.kcal_mol,
            "mass": 1.25 * Units.amu,
        }
        system.reference_values = ref_value_dict
        assert system.reference_length == ref_value_dict["length"]
        assert system.reference_energy == ref_value_dict["energy"]
        assert system.reference_mass == ref_value_dict["mass"]

    def test_set_ref_values_missing_key(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        ref_value_dict = {
            "length": 1 * Units.angstrom,
            "energy": 3.0 * Units.kcal_mol,
        }
        with pytest.raises(ValueError):
            system.reference_values = ref_value_dict

    def test_set_ref_values_invalid_type(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        ref_value_dict = {
            "length": 1 * Units.angstrom,
            "energy": 3.0 * Units.kcal_mol,
            "mass": 1.25,
        }
        with pytest.raises(UnitError):
            system.reference_values = ref_value_dict

    def test_set_ref_values_auto_scale_true(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )
        ref_value_dict = {
            "length": 1 * Units.angstrom,
            "energy": 3.0 * Units.kcal_mol,
            "mass": 1.25 * Units.amu,
        }
        with pytest.warns():
            system.reference_values = ref_value_dict

    def test_set_ref_length(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        system.reference_length = 1 * Units.angstrom
        assert system.reference_length == 1 * Units.angstrom

    def test_set_ref_length_invalid_type(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=5)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        with pytest.raises(UnitError):
            system.reference_length = 1.0

    def test_ref_length_invalid_dimension(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        with pytest.raises(UnitError):
            system.reference_length = 1.0 * Units.g

    def test_ref_length_auto_scale_true(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )
        system.reference_length = 1 * Units.angstrom
        assert system.reference_length == 1 * Units.angstrom

    def test_set_ref_energy(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        system.reference_energy = 1 * Units.kcal_mol
        assert system.reference_energy == 1 * Units.kcal_mol

    def test_set_ref_energy_invalid_type(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        with pytest.raises(UnitError):
            system.reference_energy = 1.0

    def test_ref_energy_invalid_dimension(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        with pytest.raises(UnitError):
            system.reference_energy = 1.0 * Units.g

    def test_set_ref_energy_auto_scale_true(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )
        system.reference_energy = 1 * Units.kcal_mol
        assert system.reference_energy == 1 * Units.kcal_mol

    def test_set_ref_mass(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )

        system.reference_mass = 1.0 * Units.amu
        assert system.reference_mass == 1.0 * Units.amu

    def test_set_ref_mass_invalid_type(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        with pytest.raises(UnitError):
            system.reference_mass = 1.0

    def test_ref_mass_invalid_dimension(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        with pytest.raises(UnitError):
            system.reference_energy = 1.0 * Units.m

    def test_set_ref_mass_auto_scale_true(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )

        system.reference_mass = 1.0 * Units.amu
        assert system.reference_mass == 1.0 * Units.amu

    def test_apply_forcefield_no_forcefield(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        with pytest.raises(ForceFieldError):
            system.apply_forcefield(
                r_cut=2.5, force_field=None, auto_scale=False
            )

    def test_apply_forcefield_no_forcefield_w_mol_ff(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=3, force_field=OPLS_AA())
        system = Pack(
            molecules=[benzene_mol],
            density=1.0,
        )
        system.apply_forcefield(r_cut=2.5, auto_scale=True)
        assert system.gmso_system.is_typed()
        assert len(system.hoomd_forcefield) > 0
        assert system.hoomd_snapshot.particles.N == benzene_mol.n_particles

    def test_apply_forcefield_w_mol_ff(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=3, force_field=OPLS_AA())
        system = Pack(
            molecules=[benzene_mol],
            density=1.0,
        )
        with pytest.raises(ForceFieldError):
            system.apply_forcefield(
                r_cut=2.5, force_field=OPLS_AA(), auto_scale=True
            )

    def test_validate_forcefield_invalid_ff_type(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=1)
        system = Pack(
            molecules=[benzene_mol],
            density=1.0,
        )
        with pytest.raises(ForceFieldError):
            system.apply_forcefield(
                r_cut=2.5, force_field="invalid_ff.xml", auto_scale=True
            )

    def test_validate_forcefield_mult_ff_invalid_type(
        self, dimethylether_molecule, pps_molecule
    ):
        dimethylether_mol = dimethylether_molecule(n_mols=3)
        pps_mol = pps_molecule(n_mols=2)
        system = Pack(molecules=[dimethylether_mol, pps_mol], density=0.8)
        with pytest.raises(ForceFieldError):
            system.apply_forcefield(
                r_cut=2.5,
                force_field=["invalid_ff", OPLS_AA_PPS()],
                auto_scale=True,
            )

    def test_forcefield_kwargs_attr(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5,
            force_field=[OPLS_AA()],
            auto_scale=False,
            nlist_buffer=0.5,
            pppm_resolution=(4, 4, 4),
            pppm_order=3,
        )
        assert system._ff_kwargs["r_cut"] == 2.5
        assert system._ff_kwargs["nlist_buffer"] == 0.5
        assert system._ff_kwargs["pppm_kwargs"]["resolution"] == (4, 4, 4)
        assert system._ff_kwargs["pppm_kwargs"]["order"] == 3

    def test_forcefield_list_hoomd_ff(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )

        hoomd_ff = system.hoomd_forcefield

        with pytest.raises(ForceFieldError):
            system = Pack(
                molecules=[polyethylene],
                density=1.0,
            )
            system.apply_forcefield(r_cut=2.5, force_field=hoomd_ff)

    def test_lattice_polymer(self, polyethylene):
        polyethylene = polyethylene(lengths=2, num_mols=32)
        system = Lattice(
            molecules=[polyethylene],
            x=1,
            y=1,
            n=4,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )

        assert system.n_mol_types == 1
        assert len(system.all_molecules) == len(polyethylene.molecules)
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.reference_values.keys() == {"energy", "length", "mass"}
        for mol_class in system._molecules:
            for mol in mol_class.molecules:
                backbone = get_backbone_vector(mol.xyz)
                assert np.allclose(
                    np.abs(backbone), np.array([0, 0, 1]), atol=1e-1
                )

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_lattice_periodic_polymer(self, polyethylene, axis):
        polyethylene = polyethylene(
            lengths=10, num_mols=32, periodic_bond_axis=axis
        )
        system = Lattice(
            molecules=[polyethylene],
            x=1,
            y=1,
            n=4,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )

        assert system.n_mol_types == 1
        assert len(system.all_molecules) == len(polyethylene.molecules)
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.reference_values.keys() == {"energy", "length", "mass"}
        for mol_class in system._molecules:
            for mol in mol_class.molecules:
                backbone = get_backbone_vector(mol.xyz)
                assert np.allclose(
                    np.abs(backbone), np.array([0, 0, 1]), atol=1e-1
                )

    def test_lattice_molecule(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=32)
        system = Lattice(
            molecules=[benzene_mol],
            x=1,
            y=1,
            n=4,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=OPLS_AA(), auto_scale=True
        )
        assert system.n_mol_types == 1
        assert len(system.all_molecules) == len(benzene_mol.molecules)
        assert len(system.hoomd_forcefield) > 0
        assert system.n_particles == system.hoomd_snapshot.particles.N
        assert system.reference_values.keys() == {"energy", "length", "mass"}

    def test_lattice_bead_spring(self):
        chains = LJChain(lengths=10, num_mols=32)
        Lattice(molecules=chains, x=1, y=1, n=4)

    def test_scale_charges(self, pps):
        pps_mol = pps(num_mols=5, lengths=5)
        no_scale = Pack(
            molecules=pps_mol,
            density=0.5,
        )
        no_scale.apply_forcefield(
            r_cut=2.5,
            force_field=OPLS_AA_PPS(),
            scale_charges=False,
            auto_scale=True,
        )

        with_scale = Pack(
            molecules=pps_mol,
            density=0.5,
        )
        with_scale.apply_forcefield(
            r_cut=2.5,
            force_field=OPLS_AA_PPS(),
            scale_charges=True,
            auto_scale=True,
        )
        assert abs(no_scale.net_charge.value) > abs(with_scale.net_charge.value)
        assert np.allclose(0, with_scale.net_charge.value, atol=1e-13)

    def test_to_gsd(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        system.to_gsd("test.gsd")
        assert os.path.isfile(os.path.join(os.getcwd(), "test.gsd"))
        os.remove(os.path.join(os.getcwd(), "test.gsd"))

    def test_to_gsd_no_ff(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.to_gsd("test.gsd")
        assert os.path.isfile(os.path.join(os.getcwd(), "test.gsd"))
        os.remove(os.path.join(os.getcwd(), "test.gsd"))

    def test_pickle_ff(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=False
        )
        system.pickle_forcefield("forcefield.pickle")
        assert os.path.isfile(os.path.join(os.getcwd(), "forcefield.pickle"))
        os.remove(os.path.join(os.getcwd(), "forcefield.pickle"))

    def test_pickle_ff_not_ff(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        with pytest.raises(ValueError):
            system.pickle_forcefield("forcefield.pickle")

    def test_save_reference_values(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
        )
        system.apply_forcefield(
            r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        )
        system.save_reference_values()
        assert os.path.isfile(
            os.path.join(os.getcwd(), "reference_values.pickle")
        )
        ref_values = pickle.load(
            open(os.path.join(os.getcwd(), "reference_values.pickle"), "rb")
        )
        assert ref_values == system.reference_values
        os.remove(os.path.join(os.getcwd(), "reference_values.pickle"))

    def test_save_empty_reference_values(self, polyethylene):
        polyethylene = polyethylene(lengths=5, num_mols=1)
        system = Pack(
            molecules=[polyethylene],
            density=1.0,
            base_units=dict(),
        )
        with pytest.raises(ValueError):
            system.save_reference_values()

    def test_mass_density(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=100)
        system = Pack(molecules=[benzene_mol], density=0.8 * Units.g_cm3)
        assert system.density.units == Units.g_cm3

    def test_number_density(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=100)
        system = Pack(molecules=[benzene_mol], density=0.8 * Units.n_cm3)
        assert system.density.units == Units.n_cm3

    def test_bad_density(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=100)
        with pytest.raises(ValueError):
            Pack(molecules=[benzene_mol], density=0.8 * Units.J / Units.kg)

    def test_density_warning(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=100)
        with pytest.warns():
            system = Pack(molecules=[benzene_mol], density=0.8)
            assert system.density.units == Units.g_cm3

    def test_n_particles_no_ff(self, benzene_molecule):
        benzene_mol = benzene_molecule(n_mols=100)
        system = Pack(
            molecules=[benzene_mol], density=0.8 * Units.g / Units.cm**3
        )
        assert system.n_particles == 100 * 12
