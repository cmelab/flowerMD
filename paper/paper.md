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
paramaterizing and applying a forcefield, to running simulations.
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
