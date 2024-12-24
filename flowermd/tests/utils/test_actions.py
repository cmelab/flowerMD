import copy

import hoomd
import numpy as np

from flowermd.tests import BaseTest
from flowermd.utils import ScaleEpsilon, ScaleSigma


class TestActions(BaseTest):
    def test_scale_epsilon(self, benzene_simulation):
        sim = benzene_simulation
        epsilon_scale = ScaleEpsilon(sim=sim, scale_factor=0.5)
        energy_operation = hoomd.update.CustomUpdater(
            action=epsilon_scale, trigger=10
        )
        sim.operations.updaters.append(energy_operation)
        old_lj_force = copy.deepcopy(sim._lj_force().params)
        sim.run_NVT(duration=10, temperature=1.0, tau_kt=1.0)
        new_lj_force = sim._lj_force().params
        for k in old_lj_force.keys():
            assert (
                new_lj_force[k]["epsilon"] == old_lj_force[k]["epsilon"] + 0.5
            )

    def test_scale_sigma(self, benzene_simulation):
        sim = benzene_simulation
        sigma_scale = ScaleSigma(sim=sim, scale_factor=0.5)
        energy_operation = hoomd.update.CustomUpdater(
            action=sigma_scale, trigger=10
        )
        sim.operations.updaters.append(energy_operation)
        old_lj_force = copy.deepcopy(sim._lj_force().params)
        sim.run_NVT(duration=10, temperature=1.0, tau_kt=1.0)
        new_lj_force = sim._lj_force().params
        for k in old_lj_force.keys():
            assert np.isclose(
                new_lj_force[k]["sigma"],
                old_lj_force[k]["sigma"] + 0.5,
                atol=0.0001,
            )
