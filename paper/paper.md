---
title: 'FlowerMD: Flexible Library of Organic Workflows and Extensible Recipes for Molecular Dynamics'
tags:
  - molecular simulation
  - materials science
  - molecular dynamics
  - polymers
  - HOOMD-blue
authors:
  - name: Marjan Albooyeh
    orcid: 0009-0001-9565-3076
    equal-contrib: true
    affiliation: 1
  - name: Chris Jones
    orcid: 0000-0002-6196-5274
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
 - name: Boise State University, Boise, ID, United States of America
   index: 1
date: 16 October 2023
bibliography: paper.bib
---
# Summary
`flowerMD` is a package for reproducibly performing multi-stage HOOMD-blue [@anderson2020hoomd]
simulation workflows. It enables the programmatic specification of tasks including
definition of molecular structures, forcefield definition and application and chaining
together simulation stages (e.g., shrinking, equilibration, simulating a sequence
of ensembles, tensile testing, etc.) through an extensible set of Python classes.
The modular design supports a library of workflows for organic
macromolecular and polymer simulations. Tutorials are provided to demonstrate
package features and flexibility.


# Statement of need

High-level programmatic specifications of molecular simulation workflows are
needed for two reasons. First, they provide the information necessary for a
simulation study to be reproduced, and second, they help lower the cognitive
load associated with learning and performing simulations in general.
Reproducible simulations benefit the research community by enabling studies
to be validated and extended.
Lowering the cognitive load of performing molecular simulations helps
computational researchers of all levels of expertise reason about the logic
of a simulation study. This is particularly important for researchers new to
the discipline because developing the tools needed to perform experiments often
involves: (a) gaining new software development skills and knowledge, and
(b) repeating work that others have already performed.


Recent advances in open-source tools have made the programmatic specification of
molecular simulation components easier than ever
[@anderson2020hoomd; @lammps_2022; @eastman2017openmm; @Klein2016mBuild; @gmso; @parmed;
@Santana-Bonilla_2023; @polyply_2022; @biosimspace_2019; @martin2018pyprism].
Individually, each of these tools lower the cognitive load of one aspect of an
overall workflow such as representing molecules, building initial structures,
parameterizing and applying a forcefield, and running simulations.
However, stitching these pieces together to create a complete workflow presents
a need that we address in the present work.

The computational researcher who follows best practices for accurate,
accessible and reproducible results may create a programmatic layer over these
individual software packages (i.e. wrapper) that serves to consolidate and
automate a complete workflow. However, these efforts often use a bespoke approach
where the entire workflow design is tailored toward the specific question or
project. Design choices might include the materials studied, the model used
(e.g. atomistic or coarse-grained), the source of the forcefield in the model, and
the simulation protocols followed. As a result, this wrapper is likely unusable
for the next project where one of the aforementioned choices changes, and the
process of designing a workflow must begin again from scratch.

Software packages such as Radonpy [@radonpy_2022] exist that provide an automated
workflow for building molecules and bulk structures to calculating physical
properties of polymers. However, these tools may not be suitable for modeling complex
experimental processes that extend beyond measuring material properties, such as
simulating fusion welding of polymer interfaces
[@aggarwal_molecular_2020; @bukowski_load-bearing_2021] and surface wetting
[@fan_wetting_1995; @bamane_wetting_2021].

`flowerMD` is a Python package that consolidates and automates
end-to-end workflows for modeling such engineering processes with a focus on organic
molecules. We expand the capabilities of MoSDeF [@TRUE_2020] and HOOMD-blue 
[@anderson2020hoomd], following TRUE principles of software design (Transparent, 
Reproducible, Usable by others, and Extensible [@TRUE_2020]) with modular components that
facilitate building and running workflows for specific materials science research
applications, while reducing the cognitive load and programming demands on the user's part.

# Building Blocks
`flowerMD` is extensible. Modular base classes serve as building blocks that lay the
foundation for constructing workflow recipes designed for specific applications.
The resultant recipes are agnostic to choices such as chemistry, model resolution
(e.g. atomistic vs. coarse grained) and forcefield selection.
This is accomplished via three base classes:

• The `Molecule` class utilizes the mBuild [@Klein2016mBuild] and GMSO [@gmso] packages to initialize chemical
structures from a variety of input formats. This class provides methods
for building polymers and copolymer structures, and supports a straightforward
coarse-graining process by leveraging SMARTS matching.

• The `System` class serves as an intermediary between molecular initialization
and simulation setup. This class builds the initial configuration and
applies a chosen forcefield that defines particle interactions.

• The `Simulation` class adds a layer on top of the HOOMD-blue simulation object,
adding additional methods and features to simplify the process of starting and
resuming a HOOMD-blue simulation.

Additionally, `flowerMD` offers a library pre-defined subclasses of these base classes
including common polymers, forcefields, and bulk system initialization algorithms.

# Recipes
The modular design of `flowerMD` enables version-controlled workflows to be created and shared. `flowerMD` currently includes two complete workflows: a polymer fusion welding recipe and a surface wetting recipe.
As an example, the polymer welding recipe uses the following subclasses:`flowerMD.modules.welding.SlabSimulation`, `flowerMD.modules.welding.Interface`, `flowerMD.modules.welding.WeldSimulation`,
and `flowerMD.library.simulations.Tensile`.

Applying these routines in sequence defines a polymer welding recipe:

1. `SlabSimulation` creates one "slab" with two flat surfaces of e.g. polyethylene.
2. `Interface` duplicates the resultant slab and creates an "interface" system.
3. `WeldSimulation` simulates thermal welding at this interface.
4. `Tensile` simulates a tensile test of the resultant weld to create a stress/strain curve.

In this example, three different kinds of molecular dynamics simulations are combined in sequence to enable the mechanical properties of a welded joint to be computed. By combining independent simulation steps and enabling any of them to be iterated over, a user can build more complex workflows. For example, we could specify a screening study by iterating over several weld temperatures, following each weld simulation with a sequence of tensile simulations to investigate how temperature and strain rate influence the debonding pressure of a polymer weld. For an in-depth understanding of the surface wetting workflow, interested readers can refer to the `flowerMD` tutorials.

Each of the steps in a recipe takes the molecular system and forcefield as arguments, enabling recipes that iterate over these concepts as well. For example, a user can create a recipe that generates welds of a set of polymer chemistries. Or, a user can create a recipe that measures the agreements in structural predictions for a set of forcefields.

`flowerMD`'s recipe focus allows researchers to concisely and programmatically specify their desired simulation logic. By offloading the cognitive load of workflow specification, researchers can more easily ensure this logic is correct before embarking on computationally expensive studies. We encourage simulators of all levels of expertise to file issues requesting new features and to submit pull requests to extend `flowerMD`'s utility.

# Availability
`flowerMD` is freely available under the GNU General Public License (version 3)
on [GitHub](https://github.com/cmelab/flowerMD). For installation instructions,
and Python API documentation
please visit the [documentation](https://flowermd.readthedocs.io/en/latest/).
For examples of how to use `flowerMD`,
please visit the [tutorials](https://github.com/cmelab/flowerMD/tree/main/tutorials).

# Acknowledgements
This research was partially supported by the National Aeronautics and Space
Administration (NASA) under the University Leadership Initiative program;
grant number 80NSSC20M0165.
This material is based upon work supported by the National Science Foundation
under Grant Numbers: 1653954, 1835593, and 2118217.
No sponsor had any involvement in the development of `flowerMD`.

# Conflict of Interest Statement
The authors declare the absence of any conflicts of interest: No author has any financial,
personal, professional, or other relationship that affect our objectivity toward this work.

# References
