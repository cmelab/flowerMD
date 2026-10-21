All-atom PhantomWalk
====================

`PhantomWalk` also initializes dense all-atom polymer melts. Chains are
placed at the target density as overlapping random coils, relaxed with
dissipative particle dynamics (DPD) under a soft pair force and stiffened
bonded terms, and cleaned up with FIRE. The result is a set of coordinates
at the right density that a standard force field can minimize, not an
equilibrated melt.

.. code-block:: python

    import unyt as u

    from flowermd.library import (
        AtomisticRandomWalk, LocalBoltzmannSampler, PhantomWalk,
        PolyEthylene, UFF_DPD,
    )
    from flowermd.utils import protect_stereochemistry

    chains = PolyEthylene(lengths=20, num_mols=10)
    system = AtomisticRandomWalk(
        chains,
        density=0.78 * u.g / u.cm**3,
        torsion_sampler=LocalBoltzmannSampler(temperature=413),
    )
    ff = UFF_DPD(system, file_path="pe_dpd.xml")
    system.apply_forcefield(r_cut=0.35, force_field=ff, kT=4.184)
    reference = protect_stereochemistry(system)
    sim = PhantomWalk(
        system.hoomd_snapshot, system.hoomd_forcefield,
        n_steps_dpd=40000, n_steps_fire=100, dt=5e-5, run_now=False,
    )
    record = sim.run_initialization(stereo_reference=reference)

The steps:

1. **Placement**: `AtomisticRandomWalk` gives each chain a random start
   and orientation and grows it rigid unit by rigid unit. The default
   torsion sampler, `LocalBoltzmannSampler`, first relaxes each chain under
   the bonded terms of a SMIRNOFF force field (Sage 2.3.0 by default), then
   turns every rotatable bond to a torsion drawn from local Boltzmann
   weights: the force field's torsions about that bond plus the
   Lennard-Jones energy of atoms at most four bonds apart (1-4 pairs
   halved), with a 0.30 nm heavy-atom contact check. Pass the melt
   temperature as ``temperature``. Ring bonds, ladder junctions, bond
   lengths, bond angles and stereocenters keep the relaxed geometry.
   `UniformTorsion` (no force field needed) turns only the bonds between
   repeat units to uniform random torsions, which gives unrealistic chain
   dimensions (see below). Without OpenFF, pass
   ``LocalBoltzmannSampler(force_field=OPLS_AA())`` to take the energies
   from FlowerMD's OPLS-AA instead, typed with GMSO as `System` types it
   (geometric combining rule, 1-4 Lennard-Jones halved); see the
   polyethylene caveat below.
2. **Force field**: `UFF_DPD` (RDKit UFF) or `OpenFF_DPD` (a SMIRNOFF
   force field, through OpenFF Interchange) writes a GMSO XML file for the
   system: bonded terms scaled by ``bonded_scale`` (30; proper torsions by
   ``dihedral_scale``, 300) and a DPD pair
   force whose ``A`` and ``gamma`` are weighted by
   ``sqrt(eps_i eps_j) / eps_max``. Atom types are refined until every
   bonded parameter is a function of its types, and the particles are
   renamed to match. Units are kJ/mol, nm and amu, so pass ``dt=5e-5``
   (ps) to `PhantomWalk`; its default of 0.001 is meant for reduced units
   and is unstable here.
3. **Stereochemistry**: DPD lets atoms pass through each other, so a
   tetrahedral center can invert. `protect_stereochemistry` adds a
   one-term periodic dihedral restraint per center, with its phase taken
   from the input geometry (rounded to 1 degree). The restraints form
   their own force, so the stop rule monitors their energy separately.
4. **Relaxation**: `PhantomWalk.run_initialization` runs DPD until every
   force's per-particle energy is stationary (`EnergyStationarity`: five
   samples per 500-step chunk after 3,500 steps, at most 2 % change for two
   chunks in a row, capped at ``n_steps_dpd``), then FIRE with the
   conservative part of the DPD force, audits the stereocenters, and
   returns a record of the run.

Defaults
--------

=============================  ============================================
Setting                        Default
=============================  ============================================
DPD ``A`` / ``gamma``          52,300 kJ/(mol nm) / 4,091 amu/ps, weighted
``kT``, cutoff, time step      4.184 kJ/mol, 0.35 nm, 5e-5 ps (pass it)
Neighbor-list buffer           0.04 nm
Bonded / dihedral scale        30 / 300
Stereochemistry restraint      ``k`` = 125,520 kJ/mol
=============================  ============================================

Chain dimensions
----------------

DPD and FIRE keep the chain dimensions that the placement gives (C_n
changes by less than 3 % with six times more DPD), and so does a later
nanosecond-scale anneal for entangled chains, so the placement sets them.
After Sage minimization and a 200 ps Sage NPT anneal, C_n with
`LocalBoltzmannSampler` was within about 7 % of the experimental melt
values (scaled to chain length) for polyethylene (C40 and C160), atactic
polystyrene (20- and 40-mers) and atactic polypropylene, -2 % for
bisphenol-A polycarbonate and -14 % for PET. Turning only the bonds
between repeat units uniformly (`UniformTorsion`) was off by +712 %
(polyethylene C160), -61 % (polypropylene), -36 % (polystyrene) and -38 %
(polycarbonate). Local torsion populations, which the stiffened DPD
bonded terms disturb, return to the force field's own within about 100 ps
of MD. Crowded chains such as PMMA come out too compact (about -35 %);
`LocalBoltzmannSampler` documents why.

With OPLS-AA energies the placement of atactic polystyrene (20-mers) is
close to Sage's (single-chain C_n about 7.3), but polyethylene is too stiff:
single-chain C_n of C40 is about 9, against about 6.5 expected, because
OPLS-AA over-stabilizes trans states in long alkanes (the reason for L-OPLS;
Siu et al., J. Chem. Theory Comput. 8, 1459 (2012)). Sage 2.3.0 remains the
validated default.

API
---

.. autoclass:: flowermd.library.AtomisticRandomWalk
   :noindex:

.. automodule:: flowermd.internal.local_boltzmann
   :members: LocalBoltzmannSampler, relax_bonded, rotatable_bonds

.. automodule:: flowermd.internal.placement
   :members: UniformTorsion, random_chain_conformation

See ``tutorials/9-flowerMD-all-atom-phantomwalk.ipynb`` for a complete
example.
