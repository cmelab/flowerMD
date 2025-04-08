import gsd.hoomd
import numpy as np
import pytest

from flowermd.base import Pack
from flowermd.internal import Units
from flowermd.library.forcefields import OPLS_AA
from flowermd.library.surfaces import Graphene
from flowermd.modules.surface_wetting import (
    DropletSimulation,
    InterfaceBuilder,
    WettingSimulation,
)
from flowermd.tests.base_test import BaseTest


class TestDropletSimulation(BaseTest):
    def test_droplet_sim(self, polyethylene):
        drop_mol = polyethylene(num_mols=10, lengths=5)
        drop_system = Pack(molecules=drop_mol, density=0.1)
        drop_system.apply_forcefield(
            force_field=OPLS_AA(),
            auto_scale=True,
            remove_charges=True,
            remove_hydrogens=True,
            r_cut=2.5,
        )
        drop_sim = DropletSimulation(
            initial_state=drop_system.hoomd_snapshot,
            forcefield=drop_system.hoomd_forcefield,
            reference_values=drop_system.reference_values,
        )
        drop_sim.run_droplet(
            shrink_temperature=5.0,
            shrink_duration=200,
            shrink_period=10,
            shrink_density=0.2 * Units.g_cm3,
            expand_temperature=0.5,
            expand_duration=200,
            expand_period=10,
            hold_temperature=1.0,
            hold_duration=100,
            final_density=0.05 * Units.g_cm3,
            tau_kt=drop_sim.dt * 100,
        )
        assert np.isclose(
            drop_sim.density.to(Units.g_cm3).value, 0.05, atol=1e-2
        )

    def test_droplet_sim_no_units(self, polyethylene):
        drop_mol = polyethylene(num_mols=10, lengths=5)
        drop_system = Pack(molecules=drop_mol, density=0.1)
        drop_system.apply_forcefield(
            force_field=OPLS_AA(),
            auto_scale=True,
            remove_charges=True,
            remove_hydrogens=True,
            r_cut=2.5,
        )
        drop_sim = DropletSimulation(
            initial_state=drop_system.hoomd_snapshot,
            forcefield=drop_system.hoomd_forcefield,
            reference_values=drop_system.reference_values,
        )
        with pytest.warns():
            drop_sim.run_droplet(
                shrink_temperature=5.0,
                shrink_duration=200,
                shrink_period=10,
                shrink_density=0.2,
                expand_temperature=0.5,
                expand_duration=200,
                expand_period=10,
                hold_temperature=1.0,
                hold_duration=100,
                final_density=0.05,
                tau_kt=drop_sim.dt * 100,
            )
            assert np.isclose(
                drop_sim.density.to(Units.g_cm3).value, 0.05, atol=1e-2
            )

    def test_droplet_sim_bad_units(self, polyethylene):
        drop_mol = polyethylene(num_mols=10, lengths=5)
        drop_system = Pack(molecules=drop_mol, density=0.1)
        drop_system.apply_forcefield(
            force_field=OPLS_AA(),
            auto_scale=True,
            remove_charges=True,
            remove_hydrogens=True,
            r_cut=2.5,
        )
        drop_sim = DropletSimulation(
            initial_state=drop_system.hoomd_snapshot,
            forcefield=drop_system.hoomd_forcefield,
            reference_values=drop_system.reference_values,
        )
        with pytest.raises(ValueError):
            drop_sim.run_droplet(
                shrink_temperature=5.0,
                shrink_duration=200,
                shrink_period=10,
                shrink_density=0.2 * (Units.n_cm3),
                expand_temperature=0.5,
                expand_duration=200,
                expand_period=10,
                hold_temperature=1.0,
                hold_duration=100,
                final_density=0.05 * (Units.n_cm3),
                tau_kt=drop_sim.dt * 100,
            )
            assert np.isclose(
                drop_sim.density.to(Units.g_cm3).value, 0.05, atol=1e-2
            )


class TestInterfaceBuilder(BaseTest):
    def test_interface_builder(
        self, polyethylene_system, polyethylene_droplet, graphene_snapshot
    ):
        # load droplet snapshot
        drop_snapshot = gsd.hoomd.open(polyethylene_droplet)[0]
        # recreate droplet forcefield
        polyethylene_ff = polyethylene_system.hoomd_forcefield
        drop_refs = {
            "energy": 0.276144 * Units.kJ_mol,
            "length": 0.35 * Units.nm,
            "mass": 12.011 * Units.amu,
        }

        # load surface snapshot
        surface_snapshot = gsd.hoomd.open(graphene_snapshot)[0]
        # recreate surface forcefield
        graphene = Graphene(
            x_repeat=2,
            y_repeat=2,
            n_layers=2,
            base_units=drop_refs,
        )
        graphene.apply_forcefield(force_field=OPLS_AA(), r_cut=2.5)
        graphene_ff = graphene.hoomd_forcefield

        # create interface
        interface = InterfaceBuilder(
            surface_snapshot=surface_snapshot,
            surface_ff=graphene_ff,
            drop_snapshot=drop_snapshot,
            drop_ff=polyethylene_ff,
            drop_ref_values=drop_refs,
            box_height=15 * Units.nm,
            gap=0.4 * Units.nm,
        )
        assert (
            interface.hoomd_snapshot.particles.N
            == drop_snapshot.particles.N + surface_snapshot.particles.N
        )
        assert (
            interface.hoomd_snapshot.particles.types
            == [f"_{ptype}" for ptype in surface_snapshot.particles.types]
            + drop_snapshot.particles.types
        )
        assert np.isclose(
            interface.hoomd_snapshot.configuration.box[2]
            * drop_refs["length"].value,
            15,
            atol=1e-2,
        )

        # test z gap
        assert np.isclose(
            np.abs(
                np.max(
                    interface.hoomd_snapshot.particles.position[
                        : surface_snapshot.particles.N
                    ],
                    axis=0,
                )[2]
                - np.min(
                    interface.hoomd_snapshot.particles.position[
                        surface_snapshot.particles.N :  # noqa: E203
                    ],
                    axis=0,
                )[2]
            )
            * drop_refs["length"].value,
            0.4,
            atol=1e-2,
        )


class TestWettingSimulation(BaseTest):
    def test_wetting_sim(
        self, surface_wetting_init_snapshot, surface_wetting_init_ff
    ):
        # load surface wetting init snapshot
        snapshot = gsd.hoomd.open(surface_wetting_init_snapshot)[0]
        # load ff from pickle
        import pickle

        with open(surface_wetting_init_ff, "rb") as handle:
            ff = pickle.load(handle)
        wetting_sim = WettingSimulation(
            initial_state=snapshot,
            forcefield=ff,
            fix_surface=True,
        )
        wetting_sim.run_NVT(
            temperature=1.0,
            tau_kt=1,
            duration=1e3,
        )
