import hoomd
import unyt as u

from flowermd.base import Simulation
from flowermd.library import DPD, PPS, LJChain, PhantomWalk, RandomWalk
from flowermd.tests import BaseTest


class TestPhantomWalkSimulation(BaseTest):
    def test_phantom_walk_run(self):
        pps = PPS(lengths=6, num_mols=32)
        pps.coarse_grain(beads={"_A": "c1cc(S)ccc1"})

        ref_length = 0.3438 * u.Unit("nm")
        ref_mass = 32.06 * u.Unit("amu")
        ref_energy = 1.065 * u.Unit("kJ/mol")
        ref_values_dict = {
            "length": ref_length,
            "mass": ref_mass,
            "energy": ref_energy,
        }

        system = RandomWalk(
            molecules=pps,
            density=1.32 * u.Unit("g/cm**3"),
            bond_length=1.4226,
            buffer=0.58,
            base_units=ref_values_dict,
        )

        dpd_ff = DPD(
            A=25000, gamma=800, kT=1.5, r_cut=1.5, bond_k=25000, bond_r0=1.4226
        )

        sim = PhantomWalk(
            initial_state=system.hoomd_snapshot,
            forcefield=dpd_ff.hoomd_forces,
            gsd_write_freq=10,
            log_write_freq=50,
            n_steps_dpd=500,
            n_steps_fire=100,
        )

    def test_phantom_walk_comp(self):
        molecules = LJChain(
            num_mols=[10],
            lengths=[50],
            bead_sequence=["_A"],
            bead_mass={"_A": 1.0},
            bond_lengths={"_A-_A": 1.0},
        )

        ref_length = 1.0 * u.Unit("nm")
        ref_mass = 1.0 * u.Unit("g/mol")
        ref_energy = 1.0 * u.Unit("kcal / mol")
        ref_values_dict = {
            "length": ref_length,
            "mass": ref_mass,
            "energy": ref_energy,
        }

        system = RandomWalk(
            molecules=molecules,
            density=1.1 * u.Unit("nm**-3"),
            bond_length=1.0,
            buffer=0.5,
            base_units=ref_values_dict,
        )

        dpd_ff = DPD(
            A=25000, gamma=800, kT=1.0, r_cut=1.01, bond_k=25000, bond_r0=1.0
        )

        sim = Simulation(
            initial_state=system.hoomd_snapshot,
            forcefield=dpd_ff.hoomd_forces,
            reference_values=ref_values_dict,
        )

        sim.run_NVE(n_steps=10, write_at_start=False)
        assert isinstance(sim.forces[0], hoomd.md.pair.pair.DPD)
        assert isinstance(sim.integrator, hoomd.md.Integrator)
        assert isinstance(sim.method, hoomd.md.methods.ConstantVolume)

        sim.run_FIRE(n_steps=10)
        assert isinstance(sim.forces[0], hoomd.md.pair.pair.DPD)
        assert isinstance(sim.integrator, hoomd.md.minimize.FIRE)
        assert isinstance(sim.method, hoomd.md.methods.ConstantVolume)
