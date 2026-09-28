import numpy as np
import unyt as u

from flowermd.library import PPS, PhantomWalk, RandomWalk, DPD
from flowermd.tests import BaseTest


class TestPhantomWalkSimulation(BaseTest):
    def test_tensile(self):
        pps = PPS(lengths=6, num_mols=32)
        pps.coarse_grain(beads={"_A": "c1cc(S)ccc1"})
        
        ref_length = 0.3438 * u.Unit("nm")
        ref_mass = 32.06 * u.Unit("amu")
        ref_energy = 1.065 * u.Unit("kJ/mol")
        ref_values_dict = {"length": ref_length, "mass": ref_mass, "energy": ref_energy}
        
        system = RandomWalk(
            molecules=pps,
            density=1.32 * u.Unit("g/cm**3"),
            bond_length=1.4226,
            buffer=0.58,
            base_units=ref_values_dict,
        )

        dpd_ff = DPD(
            A=25000,
            gamma=800,
            kT=1.5,
            r_cut=1.5,
            bond_k=25000,
            bond_r0=1.4226
        )

        sim = PhantomWalk(
            initial_state=system.hoomd_snapshot,
            forcefield=dpd_ff.hoomd_forces,
            gsd_write_freq=10,
            log_write_freq=50,
            n_steps_dpd=500,
            n_steps_fire=100,
        )
