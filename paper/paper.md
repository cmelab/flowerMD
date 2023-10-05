---
title: 'JankFlow: A Flexible Python Library for Organic Workflows'
tags:
  - Python
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
    corresponding: true
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
JankFlow is a package for reproducibly performing complex HOOMD-Blue simulation workflows. It enables the programmatic specification of tasks including
definition of molecular structures, forcefield definition and application, chaining
together simulation stages (e.g., shrinking, equilibration, simulating a sequence
of ensembles, tensile testing), and trajectory analysis through an extensible set
of python classes. Common tasks and forcefields for organic macrmolecular and
polymer simulations are included, as are tutorials demonstrating customization
and extensibility.

# Statement of need

High-level programmatic specification of molecular simulation workflows are
needed for two reasons: First: They provide the information necessary for a
simulation study to be repeated, and Second: They facilitate accessible peda-
gogy of molecular simulation by minimizing the cognitive load needed to get
started[?].

Most molecular simulations are performed using a combination of tools:
Text editors (to copy and edit plain text files that specify molecular
configurations, forcefield files, simulation engine input files and/or simulation scripts),
command-line interfaces for submitting simulations to a local processor or HPC
cluster qeueing system, and numerical analysis and visualization software to
inspect simulation data. When simulations are set up manually and iteratively,
it is difficult to maintain data provenance, that is, a record of what files and
choices determine the outputs of a simulation.

Even for simulations of “relatively simple” single-component systems at a
single thermodynamic state point, it is challenging for users to manage records
of which exact forcefield was applied, whether any forcefield parameters were
modified, whether a sequence of short simulations was needed to create the
initial conditions, which microstates were included in analysis, and whether
human intervention occurred at any step to inform the next. Furthermore, for
a researcher new to molecular simulations, the details of generating an initial
condition for a thermodynamic simulation are simultaneously: (a) not germane
to understanding the concept of a simulation overall, (b) crucial to get right so
the researcher can get started, and (c) likely specifyable using community best
practices for configuration initialization.

Recent advances in well-documented open-source tools [?, ?, ?, ?] have made
the programmatic specification of molecular simulation components easier than
ever. Individually, each of these tools lower the cognitive load of one aspect of
an overall workflow tailored to answer a specific research question. However,
the challenge of stitching the pieces together to create a complete workflow still
contains several barriers.

The computational researcher who follows best practices for accurate,
accessible and reproducible results may create a programmatic layer over these
individual software packages (i.e. wrapper) that serves to consolidate and
automate a complete workflow. However, these efforts often use a bespoke approach
where the entire workflow design is tailored towards the specific question or
project. Design choices might include the materials studied, the model used
(e.g. atomistic or coarse-grained), the source of the forcefield in the model, and
the simulation protocols followed. As a result, this wrapper is likely unusable
for the next project where one of the aforementioned choices changes, and the
process of designing a workflow must begin again from scratch.

Regardless of the goal at hand, designing any MD workflow typically involves
the same preliminary steps:
1. Creating molecules.
2. Building up larger, complex compounds from smaller molecules (e.g. poly-
mers, surfaces)
3. Designate the initial structure and topology of a system of compounds.
4. Choose and correctly apply a forcefield to the system.
5. Pass off the initial topology and forcefield information to a simulation
engine

Therefore, the goal of a package that aims to consolidate and automate
complete workflows should have a TRUE foundational base from which workflows
can inherit, making it significantly easier to construct new workflows without
starting from scratch. It should be extensible; a workflow from beginning to
end should not depend on the chemistry chosen, whether or not the model is
atomistic or coarse-grained, or if interaction parameters come from established
forcefields or from a machine learned model. This tool should be modular,
allowing workflows to evolve into highly specific applications further down the
pipeline, without concerns about design choices limiting or interfering with other
use cases. Moreover, the continuous maintenance, updates, and addition of features to this foundational base permeate throughout the library of workflows.
If executed thoughtfully and accurately, this enables the creation of a library
of versatile, open-source, and version-controlled workflows. JankFlow is an
attempt at making this tool by creating a TRUE base and beginning a library of
workflow modules.

[//]: # (1-2 sentnces about the popular simulation engines &#40;gromacs, lammps, hoomd,)

[//]: # (openmm&#41;. Gromacs and lammps have lots of cool features, but don’t have rich)

[//]: # (APIs, and involve text based input files which make it hard to be TRUE. Hoomd)

[//]: # (and openmm have robust APIs, but don’t have the same level of featureizaiton)

[//]: # (as lammps and gromacs. This package aims to add a layer of featureizaiton on)

[//]: # (top of hoomd.)


# Building Blocks
JankFlow simplifies the execution of molecular dynamics simulations by
integrating the capabilities of molecular builder packages like GMSO [@gms] and
MBuild [@mbuild_2016]
with the HOOMD [@hoomd_2019] simulation engine, offering a comprehensive end-to-end simulation recipe development tool.
This is accomplished through three building block classes:

• Molecule utilizes the mBuild and GMSO packages to initialize chemical
structures from a variety of input formats. This class provides methods
for building polymers and copolymer structures and supports straightforward
coarse-graining process.

• System class serves as an intermediary between molecular initialization
and simulation setup. This class is responsible for arrangement of a mixture
of molecules in a box and facilitates the configuration of particle
interactions.

• Simulation class employs the HOOMD-blue simulation object, offering
additional strategies and simulation processes tailored to specific research
needs. It also facilitates the process of quickly resuming a simulation.


# Recipes
The JankFlow package, with its flexible and extendable design, allows users
to utilize its core classes as building blocks, enabling the formulation of
customized recipes for various molecular simulation processes in accordance with
their specific research needs. To illustrate this process, we offer the two
following examples of such recipes in this package, with the expectation of introducing
more recipes in the future.

1)Welding recipe with example code
2)Tensile recipe with example code

# Availability
JankFlow is freely available under the GNU General Public License (version 3)
on [github](https://github.com/cmelab/JankFlow). For installation instructions,
and Python API documentation
please visit the [documentation](https://jankflow.readthedocs.io/en/latest/).
For examples of how to use JankFlow,
please visit the [tutorials](https://github.com/cmelab/JankFlow/tree/main/tutorials)
# Acknowledgements
We acknowledge contributions from [ULI Advisory board, NASA, etc]

# References
