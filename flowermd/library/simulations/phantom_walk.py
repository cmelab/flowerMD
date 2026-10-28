"""DPD energy relaxation simulation class and its run loops."""

import json
import warnings

import hoomd
import numpy as np

from flowermd.base.simulation import Simulation
from flowermd.internal.exceptions import StereoIntegrityError
from flowermd.utils import (
    EnergyStationarity,
    audit_stereochemistry,
    compute_closest_rdf,
    find_ring_piercings,
)
from flowermd.utils.stereochemistry import StereoRestraint


class PhantomWalk(Simulation):
    """Run an initial energy relaxation of overlapping particles with DPD and FIRE.

    By default `n_steps_dpd` DPD steps and `n_steps_fire` FIRE steps run on
    construction. With ``run_now=False``, call `run_initialization` instead, which
    runs DPD until the energies are stationary (at most `n_steps_dpd`
    steps) and returns a record of the run; this is the protocol for
    all-atom systems built with `flowermd.library.AtomisticRandomWalk` and
    `flowermd.library.UFF_DPD`.

    Parameters
    ----------
    initial_state : gsd.hoomd.Frame or str, required
        The initial overlapping configuration.
    forcefield : list of hoomd.md.force.Force, required
        Forces including a `hoomd.md.pair.DPD`.
    n_steps_dpd : int, required
        DPD steps, or the cap on DPD steps for `run_initialization`.
    n_steps_fire : int, required
        FIRE steps.
    dt : float, default 0.001
        Time step. The default suits reduced units; for all-atom systems in
        kJ/mol, nm and amu (`flowermd.library.UFF_DPD`) use about 5e-5 ps,
        since 0.001 ps is unstable under the stiffened bonded terms.
    write_at_start : bool, default False
        Trigger the writers before the first DPD step.
    run_now : bool, default True
        Run DPD and FIRE on construction.

    """

    def __init__(
        self,
        initial_state,
        forcefield,
        n_steps_dpd,
        n_steps_fire,
        reference_values=dict(),
        dt=0.001,
        device=hoomd.device.auto_select(),
        seed=42,
        gsd_write_freq=1e4,
        gsd_file_name="trajectory.gsd",
        log_write_freq=1e3,
        log_file_name="log.txt",
        write_at_start=False,
        run_now=True,
    ):
        self.n_steps_dpd = n_steps_dpd
        self.n_steps_fire = n_steps_fire
        self.write_at_start = write_at_start
        super(PhantomWalk, self).__init__(
            initial_state=initial_state,
            forcefield=forcefield,
            reference_values=reference_values,
            dt=dt,
            device=device,
            seed=seed,
            gsd_write_freq=gsd_write_freq,
            gsd_file_name=gsd_file_name,
            log_write_freq=log_write_freq,
            log_file_name=log_file_name,
        )
        if not run_now:
            return
        self.run_NVE(
            n_steps=self.n_steps_dpd, write_at_start=self.write_at_start
        )
        self.run_FIRE(n_steps=self.n_steps_fire)
        self.flush_writers()
        self.particle_separation = compute_closest_rdf(
            self, bins=100, r_max=1.05
        )
        print(
            "The closest particles after DPD + FIRE are: ",
            self.particle_separation,
        )

    def run_initialization(
        self,
        stop=None,
        min_steps=3500,
        chunk=500,
        samples_per_chunk=5,
        conservative_fire=True,
        check_stereo=True,
        check_ring_piercing=True,
        require_convergence=False,
        record_file=None,
        **fire_kwargs,
    ):
        """Run DPD until `stop` is satisfied, then FIRE, and return a record.

        Parameters
        ----------
        stop : callable, optional
            ``stop(sim) -> bool``, see `run_until`. Defaults to
            `flowermd.utils.EnergyStationarity`.
        min_steps : int, default 3500
            DPD steps before the first check.
        chunk : int, default 500
            DPD steps between checks.
        samples_per_chunk : int, default 5
            Energy samples averaged per chunk, see `run_until`.
        conservative_fire : bool, default True
            Run FIRE with `hoomd.md.pair.DPDConservative` (the repulsion
            only) in place of the DPD pair force, whose noise and friction
            keep FIRE from converging.
        check_stereo : bool, default True
            After FIRE, audit the stereocenters restrained by
            `flowermd.utils.protect_stereochemistry` (which
            `flowermd.library.AtomisticRandomWalk` adds); an inverted or
            flattened center raises `StereoIntegrityError`. Without the
            restraint there is nothing to audit.
        check_ring_piercing : bool, default True
            After FIRE, count bonds that pierce a ring
            (`flowermd.utils.find_ring_piercings`), and warn if any do.
        require_convergence : bool, default False
            Raise ``RuntimeError`` instead of warning when DPD reaches
            `n_steps_dpd` without satisfying `stop`.
        record_file : str, optional
            Also write the record to this JSON file.
        **fire_kwargs
            Passed to `run_FIRE`.

        Returns
        -------
        dict
            DPD steps, convergence and energy history, FIRE convergence,
            the stereochemistry audit and the number of ring piercings.

        """
        stop = stop or EnergyStationarity()
        dpd = run_DPD(
            self,
            n_steps=self.n_steps_dpd,
            stop=stop,
            chunk=chunk,
            min_steps=min_steps,
            samples_per_chunk=samples_per_chunk,
            write_at_start=self.write_at_start,
        )
        if not dpd["stopped_by_criterion"]:
            message = f"DPD reached {dpd['steps']} steps without converging."
            if require_convergence:
                raise RuntimeError(message)
            warnings.warn(message)
        if conservative_fire:
            pair = next(f for f in self.forces if type(f) is hoomd.md.pair.DPD)
            repulsion = hoomd.md.pair.DPDConservative(nlist=pair.nlist)
            for types, params in pair.params.items():
                repulsion.params[types] = dict(A=params["A"])
                repulsion.r_cut[types] = pair.r_cut[types]
            self.remove_force(pair)
            self.add_force(repulsion)
        self.run_FIRE(n_steps=self.n_steps_fire, **fire_kwargs)
        record = {
            "dpd_steps": dpd["steps"],
            "dpd_converged": dpd["stopped_by_criterion"],
            "dpd_energy_history": np.asarray(
                getattr(stop, "history", [])
            ).tolist(),
            "fire_converged": bool(self.integrator.converged),
        }
        if conservative_fire:  # the next MD run gets the DPD force back
            self.remove_force(repulsion)
            self.add_force(pair)
        snapshot = self.state.get_snapshot()
        restraint = next(
            (f for f in self.forces if isinstance(f, StereoRestraint)), None
        )
        if check_stereo and restraint is not None:
            audit = audit_stereochemistry(restraint.reference, snapshot)
            record["stereochemistry"] = audit
            if audit["inverted"] or audit["near_planar"]:
                raise StereoIntegrityError(str(audit))
        if check_ring_piercing:
            record["ring_piercings"] = len(find_ring_piercings(snapshot))
            if record["ring_piercings"]:
                warnings.warn(
                    f"{record['ring_piercings']} bonds pierce rings after "
                    "relaxation; minimization rarely removes them."
                )
        if record_file:
            with open(record_file, "w") as f:
                json.dump(record, f, indent=2)
        self.flush_writers()
        return record


def run_until(
    sim,
    n_steps,
    stop,
    chunk=500,
    min_steps=0,
    samples_per_chunk=1,
    write_at_start=True,
):
    """Run `sim` with its current integrator until ``stop(sim)`` is True.

    Runs `min_steps`, then chunks of `chunk` steps, calling ``stop(sim)``
    after each, until it returns True or at least `n_steps` steps have run.
    When ``samples_per_chunk > 1`` each chunk is split into that many equal
    pieces and ``stop.sample(sim)`` is called between them, so the
    criterion can average over the chunk.

    Parameters
    ----------
    sim : flowermd.base.Simulation, required
        A simulation whose integrator is already set.
    n_steps : int, required
        Cap on the number of steps, rounded up to whole chunks.
    stop : callable, required
        ``stop(sim) -> bool``, optionally with a ``sample(sim)`` method.
    chunk : int, default 500
        Steps between calls to `stop`.
    min_steps : int, default 0
        Steps before the first chunk.
    samples_per_chunk : int, default 1
        Pieces per chunk; must divide `chunk`.
    write_at_start : bool, default True
        When set to True, triggers writers that evaluate to True
        for the initial step to execute before the next simulation
        time step.

    Returns
    -------
    dict
        ``{"steps": int, "stopped_by_criterion": bool}``.

    """
    if chunk % samples_per_chunk:
        raise ValueError("chunk must be a multiple of samples_per_chunk.")
    sample = getattr(stop, "sample", None) if samples_per_chunk > 1 else None
    piece = chunk // samples_per_chunk if sample else chunk
    steps = 0
    if min_steps:
        sim.run(steps=min_steps, write_at_start=write_at_start)
        steps, write_at_start = min_steps, False
    while steps < n_steps:
        for i in range(chunk // piece):
            if i:
                sample(sim)
            sim.run(steps=piece, write_at_start=write_at_start)
            write_at_start = False
        steps += chunk
        if stop(sim):
            return {"steps": steps, "stopped_by_criterion": True}
    return {"steps": steps, "stopped_by_criterion": False}


def run_FIRE_until_converged(
    sim, n_steps, max_steps, write_at_start=False, **fire_kwargs
):
    """Run FIRE in chunks of `n_steps` until it converges or `max_steps`.

    One minimizer is kept for the whole call, so its adaptive time step
    and mixing carry over between chunks.

    Parameters
    ----------
    sim : flowermd.base.Simulation, required
    n_steps : int, required
        Steps per chunk; convergence is checked after each chunk.
    max_steps : int, required
        Cap on the number of FIRE steps, rounded up to whole chunks.
    write_at_start : bool, default False
    **fire_kwargs
        Tolerances and FIRE parameters, as in
        `flowermd.base.Simulation.run_FIRE`.

    Returns
    -------
    dict
        ``{"steps": int, "converged": bool}``.

    """
    if max_steps < n_steps:
        raise ValueError("max_steps must be at least n_steps.")
    tolerances = dict(force_tol=1e-1, angmom_tol=1000, energy_tol=1e-1)
    sim.set_fire_minimizer(
        fire_kwargs={**tolerances, **fire_kwargs},
        integrator_method=hoomd.md.methods.ConstantVolume,
        method_kwargs={"filter": sim.integrate_group},
    )
    result = run_until(
        sim,
        max_steps,
        lambda s: s.integrator.converged,
        chunk=n_steps,
        write_at_start=write_at_start,
    )
    return {
        "steps": result["steps"],
        "converged": result["stopped_by_criterion"],
    }


def run_DPD(sim, n_steps, stop=None, write_at_start=True, **kwargs):
    """Run NVE dynamics thermostatted by the DPD pair force.

    With ``stop=None`` this is `run_NVE` for `n_steps`; otherwise it runs
    until ``stop(sim)`` is True, see `run_until`.

    Parameters
    ----------
    sim : flowermd.base.Simulation, required
    n_steps : int, required
        Number of steps, or the cap when `stop` is given.
    stop : callable, optional
        ``stop(sim) -> bool``, such as `flowermd.utils.EnergyStationarity`.
    write_at_start : bool, default True
    **kwargs
        ``chunk``, ``min_steps`` and ``samples_per_chunk`` for `run_until`.

    Returns
    -------
    dict
        ``{"steps": int, "stopped_by_criterion": bool}``.

    """
    if stop is None:
        sim.run_NVE(n_steps=n_steps, write_at_start=write_at_start)
        return {"steps": n_steps, "stopped_by_criterion": False}
    sim.set_integrator_method(
        integrator_method=hoomd.md.methods.ConstantVolume,
        method_kwargs={"filter": sim.integrate_group},
    )
    return run_until(
        sim, n_steps, stop, write_at_start=write_at_start, **kwargs
    )
