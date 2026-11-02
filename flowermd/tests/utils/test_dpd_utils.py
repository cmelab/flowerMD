import numpy as np
import pytest

from flowermd.utils import EnergyStationarity


class _Sim:
    def __init__(self, *energies):
        self.forces = [type("Force", (), {"energy": e})() for e in energies]
        self.state = type("State", (), {"N_particles": 10})()


class TestEnergyStationarity:
    def test_requires_consecutive_passes(self):
        stop = EnergyStationarity(tol=0.02, consecutive=2)
        assert stop(_Sim(100.0, -50.0)) is False
        assert stop(_Sim(100.5, -50.2)) is False
        assert stop(_Sim(100.9, -50.1)) is True
        np.testing.assert_allclose(stop.history[0], [10.0, -5.0])

    def test_any_force_changing_resets(self):
        stop = EnergyStationarity(tol=0.02, consecutive=2)
        stop(_Sim(100.0, -50.0))
        stop(_Sim(100.5, -50.2))
        assert stop(_Sim(100.5, -60.0)) is False
        assert stop(_Sim(100.5, -60.1)) is False
        assert stop(_Sim(100.5, -60.2)) is True

    def test_forces_subset(self):
        first = _Sim(100.0, -50.0)
        stop = EnergyStationarity(
            tol=0.02, consecutive=1, forces=first.forces[:1]
        )
        assert stop(first) is False
        first.forces[1].energy = -80.0  # not monitored
        assert stop(first) is True
        assert np.shape(stop.history) == (2, 1)

    def test_nonfinite_energy_resets(self):
        stop = EnergyStationarity(tol=0.02, consecutive=1)
        stop(_Sim(1.0))
        assert stop(_Sim(np.nan)) is False
        assert stop(_Sim(1.0)) is False
        assert stop(_Sim(1.0)) is True

    def test_samples_are_averaged(self):
        stop = EnergyStationarity(tol=0.02, consecutive=1)
        stop.sample(_Sim(90.0))
        stop.sample(_Sim(110.0))
        assert stop(_Sim(100.0)) is False
        np.testing.assert_allclose(stop.history[0], [10.0])

    def test_reset_and_arguments(self):
        stop = EnergyStationarity(tol=0.02, consecutive=1)
        stop.sample(_Sim(1.0))
        stop(_Sim(1.0))
        stop.reset()
        assert stop.history == [] and stop(_Sim(1.0)) is False
        with pytest.raises(ValueError):
            EnergyStationarity(tol=0.0)
        with pytest.raises(ValueError):
            EnergyStationarity(consecutive=0)
