---
title: 'JankFlow: A Flexible Python Library for Organic Workflows'
tags:
  - molecular simulation
  - materials science
  - molecular dynamics
  - polymers
  - HOOMD-blue
authors:
  - name: Chris Jones
    orcid: 0000-0002-6196-5274
    equal-contrib: true
    affiliation: 1
  - name: Marjan Albooyeh
    orcid: 0009-0001-9565-3076
    equal-contrib: true
    affiliation: 1
  - name: Rainier Barrett
    orcid: 0000-0002-5728-9074
    equal-contrib: false
    affiliation: 1
  - name: Eric Jankowski
    orcid: 0000-0002-3267-1410
    corresponding: true
    affiliation: 1
affiliations:
 - name: Boise State University, Boise, ID, USA
   index: 1
date: 01 January 2001
bibliography: paper.bib

---
# Summary
`JankFlow` is a package for reproducibly performing complex HOOMD-Blue simulation workflows.
It enables the programmatic specification of tasks including
definition of molecular structures, forcefield definition and application and chaining
together simulation stages (e.g., shrinking, equilibration, simulating a sequence
of ensembles, tensile testing) through an extensible set
of Python classes. The modular design supports a library of complex workflows
for organic macrmolecular and polymer simulations.
Tutorials are provided to demonstrate package features and flexibility.


# Statement of need

High-level programmatic specification of molecular simulation workflows are
needed for two reasons: First: They provide the information necessary for a
simulation study to be reproducible, and Second: They minimize the cognitive
load of getting started with running experiments [?].
For a researcher new to molecular simulations, building the necessary set
of computational tools needed to actually perform experiments simultaneously:
(a) requires skills and knowledge different from those needed to do research,
(b) involves repeating work that others have already done.

This is a well recognized problem, and recent advances in well-documented
open-source tools have made the programmatic specification of
molecular simulation components easier than ever [? ? ? ?].
Individually, each of these tools lower the cognitive load of one aspect of an
overall workflow such as representing molecules, building initial structures,
parameterizing and applying a forcefield, to running simulations.
However, the challenge of stitching the pieces together to create a complete
workflow still contains several barriers.

The computational researcher who follows best practices for accurate,
accessible and reproducible results may create a programmatic layer over these
individual software packages (i.e. wrapper) that serves to consolidate and
automate a complete workflow [?, ?, ?]. However, these efforts often use a bespoke approach
where the entire workflow design is tailored towards the specific question or
project. Design choices might include the materials studied, the model used
(e.g. atomistic or coarse-grained), the source of the forcefield in the model, and
the simulation protocols followed. As a result, this wrapper is likely unusable
for the next project where one of the aforementioned choices changes, and the
process of designing a workflow must begin again from scratch.

Software packages such as Radonpy exist that provide an automated workflow for
building molecules and bulk structures to calculating physical properties of polymers.
This doesn't work when modeling complex experimental processes that go beyond measuring
material properties such as fusion weding of polymer interface, surface wetting, [?, ?, ?]

Jankflow is a python package that consolidates and automates
end-to-end workflows for modeling such processes with a focus on organic molecules.
Following the principals of Transparent, Reproducible, Usable by others, and Extensible (TRUE) [?]
software design, the modular design of `JankFlow` facilitates building and
running workflows for specific materials science research applications,
while reducing the cognitive load and programming demands on the user's part.

# Building Blocks
`JankFlow` is extensible; flexible and modular base classes in form of building blocks lay the
foundations for constructing segregated workflow recipes designed for specific applications.
The recipes are agnostic to choices such as chemistry, model resolution
(atomistic or coarse grained) or forcefields. This is accomplished by utilizing three base classes:

• Molecule utilizes the mBuild and GMSO packages to initialize chemical
structures from a variety of input formats. This class provides methods
for building polymers and copolymer structures and supports straightforward
coarse-graining process.

• System class serves as an intermediary between molecular initialization
and simulation setup. This class builds the initial configuration and
generates the focefield that defines particle interactions.

• Simulation class adds a layer on top of the HOOMD-blue simulation object, which
adds additional methods and features that simplifies the process of starting and
resuming a HOOMD-blue simulation.

Additionally, `JankFlow` offers a library pre-defined subclasses of the above base classes
including common polymers, forcefields and bulk system initialization algorithms.

# Recipes
`JankFlow` offers the following two ready-to-go recipes to illustrate how the design creates
potential for expanding the library of open-source and version-controlled workflows.

• Welding: What does this recipe do. Simulation to create slabs, building up an
interface from slabs, simulation to preform welding.
• Tensile Testing
```python
from jankflow.library import PolyEthylene, OPLS_AA, Tensile
from jankflow.base import Pack
from jankflow.modules.welding import SlabSimulation, Interface, WeldSimulation

# initialize a polymer system using poly(ethylene) and OPLS-AA forcefield
molecule = PolyEthylene(num_mols=30, lengths=12)
system = Pack(molecules=molecule, density=1.1)
system.apply_forcefield(r_cut=2.5, force_field=OPLS_AA(), auto_scale=True,
                        remove_charges=True, remove_hydrogens=True)
# saving forces for later use
hoomd_forces = system.hoomd_forcefield
# run a slab simulation
sim = SlabSimulation.from_system(system=system, interface_axis=(1, 0, 0),
                                 gsd_file_name="slab.gsd")
# shrink the box to reach the desired density
sim.run_update_volume(final_density=1.2, n_steps=5e4, kT=5.0,
                      period=100, tau_kt=0.001)
# run NVT ensemble
sim.run_NVT(kT=5.0, n_steps=4e4, tau_kt=0.001)

# create an interface from the slab
interface = Interface(gsd_file="slab.gsd", interface_axis=(1, 0, 0), gap=0.05)
# run the welding simulation
weld_sim = WeldSimulation(initial_state=interface.hoomd_snapshot,
                          forcefield=hoomd_forces, interface_axis=(1, 0, 0),
                          gsd_file_name="weld.gsd", log_file_name="weld_log.txt",
                          log_write_freq=500, dt=0.0003)
weld_sim.run_NVT(kT=10.0, n_steps=7e4, tau_kt=0.001)
cooling_ramp = weld_sim.temperature_ramp(n_steps=2e4, kT_start=10.0, kT_final=2.0)
weld_sim.run_NVT(kT=cooling_ramp, n_steps=2e4, tau_kt=0.001)
weld_sim.save_restart_gsd("weld_restart.gsd")
# Running a tensile test simulation
tensile_sim = Tensile(initial_state="weld_restart.gsd", forcefield=hoomd_forces,
                      tensile_axis=(1,0,0), gsd_file_name="tensile.gsd",
                      gsd_write_freq=1000, log_file_name="tensile_log.txt",
                      log_write_freq=500, fix_ratio=0.30)
tensile_sim.run_tensile(n_steps=1e5, strain=0.70, period=500, kT=2.0, tau_kt=0.001)
# for more details on the tensile test results, please see the tutorials.
```

# Availability
`JankFlow` is freely available under the GNU General Public License (version 3)
on [github](https://github.com/cmelab/JankFlow). For installation instructions,
and Python API documentation
please visit the [documentation](https://jankflow.readthedocs.io/en/latest/).
For examples of how to use `JankFlow`,
please visit the [tutorials](https://github.com/cmelab/JankFlow/tree/main/tutorials)
# Acknowledgements
We acknowledge contributions from [ULI Advisory board, NASA, etc]

# References
