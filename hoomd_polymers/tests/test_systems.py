import os
import pytest
import random
import hoomd

from hoomd_polymers.systems import * 
from hoomd_polymers.molecules import *
from hoomd_polymers.forcefields import *
from base_test import BaseTest


class TestSystems(BaseTest):
    def test_pack(self):
        system = Pack(molecule=PPS, n_mols=5, mol_kwargs={"length": 5}, density=1.0)

    def test_lattice(self):
        system = Lattice(
                molecule=PPS,
                n_mols=32,
                mol_kwargs={"length": 5},
                x=1,
                y=1,
                n=4,
                density=1.0
        )

    def test_set_target_box(self):
        pass

    def test_hoomd_ff(self):
        pass

    def test_hoomd_snap(self):
        system = Pack(PEKK_para, n_mols=20, mol_kwargs={"length": 1}, density=1.0)
        system.apply_forcefield(forcefield=GAFF())
        assert system.hoomd_snapshot.particles.N == system.system.n_particles

    def test_mass(self):
        system = Pack(PPS, n_mols=20, mol_kwargs={"length": 1}, density=1.0)
        assert np.allclose(system.mass, ((12.011*6) + (1.008*6) + 32.06)*20, atol=1e-4) 

    def test_box(self):
        pass

    def test_density(self):
        pass

    def test_ref_distance(self):
        system = Pack(
                molecule=PolyEthylene,
                n_mols=5,
                mol_kwargs={"length": 5},
                density=1.0
        )
        system.apply_forcefield(forcefield=GAFF())
        assert np.allclose(system.reference_distance.value, 3.39966951, atol=1e-3)
        reduced_box = system.hoomd_snapshot.configuration.box[0:3]
        calc_box = reduced_box * system.reference_distance.to("nm").value
        assert np.allclose(calc_box[0], system.box.Lx, atol=1e-2)
        assert np.allclose(calc_box[1], system.box.Ly, atol=1e-2)
        assert np.allclose(calc_box[2], system.box.Lz, atol=1e-2)

    def test_ref_mass(self):
        system = Pack(
                molecule=PolyEthylene,
                n_mols=5,
                mol_kwargs={"length": 5},
                density=1.0
        )
        system.apply_forcefield(forcefield=GAFF())
        total_red_mass = sum(system.hoomd_snapshot.particles.mass)
        assert np.allclose(
                system.mass,
                total_red_mass * system.reference_mass.to("amu").value,
                atol=1e-1
        )

    def test_ref_energy(self):
        system = Pack(
                molecule=PolyEthylene,
                n_mols=5,
                mol_kwargs={"length": 5},
                density=1.0
        )
        system.apply_forcefield(forcefield=GAFF())
        assert np.allclose(system.reference_energy.value, 0.1094, atol=1e-3)


    def test_apply_forcefield(self):
        system = Pack(molecule=PolyEthylene, n_mols=5, mol_kwargs={"length": 5}, density=1.0)
        system.apply_forcefield(forcefield=GAFF())
        assert isinstance(system.hoomd_snapshot, hoomd.snapshot.Snapshot)
        assert isinstance(system.hoomd_forcefield, list)

    def test_remove_hydrogens(self):
        system = Pack(
                molecule=PolyEthylene,
                n_mols=5,
                mol_kwargs={"length": 5},
                density=1.0
        )
        system.apply_forcefield(forcefield=OPLS_AA(), remove_hydrogens=True)
        assert system.hoomd_snapshot.particles.N == 5*5*2
        assert np.allclose(
                system.mass,
                sum([a.mass for a in system.typed_system.atoms]), atol=1e-1
        )

    def test_remove_charges(self):
        system = Pack(
                molecule=PolyEthylene,
                n_mols=5,
                mol_kwargs={"length": 5},
                density=1.0
        )
        system.apply_forcefield(forcefield=OPLS_AA(), remove_charges=True)
        assert sum(a.charge for a in system.typed_system.atoms) == 0
        assert sum(system.hoomd_snapshot.particles.charge) == 0

    def test_make_charge_neutral(self):
        system = Pack(
                molecule=PPS,
                n_mols=5,
                mol_kwargs={"length": 5},
                density=1.0
        )
        system.apply_forcefield(forcefield=OPLS_AA_PPS(), make_charge_neutral=True)
        assert np.allclose(0, sum([a.charge for a in system.typed_system.atoms]), atol=1e-5)

    def test_scale_parameters(self):
        pass
