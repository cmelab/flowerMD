import json

import hoomd
import pytest
import unyt as u

from flowermd.base import Polymer, Simulation
from flowermd.library import (
    DPD,
    PPS,
    UFF_DPD,
    AtomisticRandomWalk,
    LJChain,
    PhantomWalk,
    PolyEthylene,
    RandomWalk,
    UniformTorsion,
)
from flowermd.library.simulations import phantom_walk
from flowermd.library.simulations.phantom_walk import (
    run_FIRE_until_converged,
    run_until,
)
from flowermd.tests import BaseTest
from flowermd.utils.stereochemistry import StereoRestraint


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

        PhantomWalk(
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


class TestAllAtomPhantomWalk(BaseTest):
    def sim(self, n_steps_dpd=1000, chains=None, **kwargs):
        system = AtomisticRandomWalk(
            chains or PolyEthylene(lengths=10, num_mols=12),
            density=0.85,
            seed=1,
            torsion_sampler=UniformTorsion(),
            **kwargs,
        )
        ff = UFF_DPD(system)
        system.apply_forcefield(r_cut=0.35, force_field=ff, kT=4.184)
        sim = PhantomWalk(
            system.hoomd_snapshot,
            system.hoomd_forcefield,
            n_steps_dpd=n_steps_dpd,
            n_steps_fire=50,
            dt=2e-4,
            device=hoomd.device.CPU(),
            run_now=False,
        )
        return sim

    def styrene(self):
        return Polymer(
            lengths=5,
            num_mols=4,
            smiles="CC(c1ccccc1)",
            bond_indices=[8, 11],
            bond_length=0.154,
            bond_orientation=[None, None],
        )

    def run(self, sim, **kwargs):
        return sim.run_initialization(
            stop=lambda s: True, min_steps=100, chunk=100, **kwargs
        )

    def test_run_initialization(self):
        sim = self.sim(chains=self.styrene())
        forces = list(sim.forces)
        record = self.run(sim, record_file="record.json")
        assert record["dpd_steps"] == 200 and record["dpd_converged"]
        # stereocenters are restrained and audited without being asked
        assert record["stereochemistry"] == {
            "n_centers": 16,
            "inverted": 0,
            "near_planar": 0,
        }
        assert isinstance(record["ring_piercings"], int)
        assert json.load(open("record.json")) == record
        assert sim.timestep == 250
        # every force, the DPD force included, is back for the next MD run
        assert {id(f) for f in sim.forces} == {id(f) for f in forces}
        sim.run_NVE(n_steps=10)
        assert type(sim.integrator) is hoomd.md.Integrator

    def test_checks_off(self):
        sim = self.sim(chains=self.styrene(), protect_stereo=False)
        assert not any(isinstance(f, StereoRestraint) for f in sim.forces)
        record = self.run(sim, check_ring_piercing=False)
        assert "stereochemistry" not in record
        assert "ring_piercings" not in record

    def test_no_centers(self):
        record = self.run(self.sim())  # polyethylene: nothing to restrain
        assert "stereochemistry" not in record

    def test_ring_piercing_warning(self, monkeypatch):
        sim = self.sim()
        monkeypatch.setattr(
            phantom_walk, "find_ring_piercings", lambda snap: [((0, 1), ())]
        )
        with pytest.warns(UserWarning, match="pierce rings"):
            record = self.run(sim)
        assert record["ring_piercings"] == 1

    def test_run_FIRE_until_converged(self):
        sim = self.sim()
        tolerances = dict(force_tol=1e9, angmom_tol=1e9, energy_tol=1e9)
        result = run_FIRE_until_converged(sim, 50, 200, **tolerances)
        assert result == {"steps": 50, "converged": True}
        with pytest.raises(ValueError):
            run_FIRE_until_converged(sim, n_steps=50, max_steps=10)

    def test_require_convergence(self):
        sim = self.sim(n_steps_dpd=200)
        with pytest.raises(RuntimeError):
            sim.run_initialization(
                stop=lambda s: False,
                min_steps=100,
                chunk=100,
                require_convergence=True,
            )


class TestRunUntil:
    class Sim:
        timestep = 0

        def run(self, steps, write_at_start=False):
            self.timestep += steps

    class Stop:
        def __init__(self, at):
            self.at, self.samples, self.calls = at, [], []

        def sample(self, sim):
            self.samples.append(sim.timestep)

        def __call__(self, sim):
            self.calls.append(sim.timestep)
            return sim.timestep >= self.at

    def test_chunks_and_samples(self):
        sim, stop = self.Sim(), self.Stop(at=400)
        result = run_until(
            sim, 1000, stop, chunk=100, min_steps=200, samples_per_chunk=4
        )
        assert result == {"steps": 400, "stopped_by_criterion": True}
        assert stop.calls == [300, 400]
        assert stop.samples == [225, 250, 275, 325, 350, 375]

    def test_cap(self):
        sim, stop = self.Sim(), self.Stop(at=10**9)
        result = run_until(sim, 250, stop, chunk=100)
        assert result == {"steps": 300, "stopped_by_criterion": False}
        with pytest.raises(ValueError):
            run_until(sim, 250, stop, chunk=100, samples_per_chunk=3)
