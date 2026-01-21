import copy

import hoomd
import numpy as np

from flowermd.tests import BaseTest
from flowermd.utils import ShiftEpsilon, ShiftSigma


class TestActions(BaseTest):
    def test_shift_epsilon(self, benzene_simulation):
        sim = benzene_simulation
        old_lj_force = copy.deepcopy(
            [f for f in sim._pair_force() if isinstance(f, hoomd.md.pair.LJ)][
                0
            ].params
        )

        epsilon_shift = ShiftEpsilon(sim=sim, shift_by=0.5)
        energy_operation = hoomd.update.CustomUpdater(
            action=epsilon_shift, trigger=10
        )
        sim.operations.updaters.append(energy_operation)
        sim.run_NVT(n_steps=10, kT=1.0, tau_kt=1.0)

        new_lj_force = [
            f for f in sim._pair_force() if isinstance(f, hoomd.md.pair.LJ)
        ][0].params
        for k in old_lj_force.keys():
            assert (
                new_lj_force[k]["epsilon"] == old_lj_force[k]["epsilon"] + 0.5
            )

    def test_shift_sigma(self, benzene_simulation):
        sim = benzene_simulation
        old_lj_force = copy.deepcopy(
            [f for f in sim._pair_force() if isinstance(f, hoomd.md.pair.LJ)][
                0
            ].params
        )
        sigma_shift = ShiftSigma(sim=sim, shift_by=0.5)
        energy_operation = hoomd.update.CustomUpdater(
            action=sigma_shift, trigger=10
        )
        sim.operations.updaters.append(energy_operation)
        sim.run_NVT(n_steps=10, kT=1.0, tau_kt=1.0)
        new_lj_force = [
            f for f in sim._pair_force() if isinstance(f, hoomd.md.pair.LJ)
        ][0].params
        for k in old_lj_force.keys():
            assert np.isclose(
                new_lj_force[k]["sigma"],
                old_lj_force[k]["sigma"] + 0.5,
                atol=0.0001,
            )
