import numpy as np

from hoomd_organics.base.system import Lattice
from hoomd_organics.library import OPLS_AA_PPS, PPS, Tensile
from hoomd_organics.tests import BaseTest


class TestTensileSimulation(BaseTest):
    def test_tensile(self):
        pps = PPS(lengths=6, num_mols=32)
        system = Lattice(
            molecules=[pps],
            force_field=[OPLS_AA_PPS()],
            density=1.0,
            r_cut=2.5,
            x=1.2,
            y=1.2,
            n=4,
            auto_scale=True,
        )
        tensile_sim = Tensile(
            initial_state=system.hoomd_snapshot,
            forcefield=system.hoomd_forcefield,
            tensile_axis=(1, 0, 0),
            log_write_freq=1e6,
            gsd_write_freq=1e6,
        )
        tensile_sim.run_tensile(
            strain=0.05, kT=2.0, n_steps=1e3, period=10, tau_kt=0.001
        )
        assert np.allclose(tensile_sim.strain, 0.05, 1e-4)
