---
title: 'Polybinder: A Python package for streamlined polymer molecular dynamics'
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
  - name: Rainier Barrett
    orcid: 0000-0002-5728-9074
    equal-contrib: true
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

# Statement of need

One of the steeper learning curves on the road to using molecular dynamics (MD) simulations
of polymer systems is
the initialization of particle positions, topology, and force field parameters.
Especially for researchers new to the computational community, generating particle
positions and initializing their velocities alone can be a daunting task, let alone
specifying bond topology. The concept of a force field file and how it interacts with
different MD engines is also non-trivial to understand.
Further, besides the difficulty in starting simulations, another part of the cognitive
load involved in learning to perform informative MD investigations is that it can be
difficult to keep track of many simulations, especially when scanning over a wide
range of thermodynamic state points, system chemistries, etc.
These wide scans can quickly create an unmanageable amount of data to analyze,
or even store, requiring further learning of data management skills on top of an already
cumbersome training process.
Moreover, even with a high-throughput workflow and good grasp of theory, it can still
prove difficult to produce TRUE (standing for "Transparent, Reproducible, Usable, and
Extensible") MD simulations with reliable, accurate results. `@JankowskiTRUE2020`

In particular, when we want to probe complex variable spaces,
such as process control variables in a composite material manufacture process,
we need to run several large-volume, high-density, long-time simulations,
many of which may not reveal relevant information for process control.
This search process is further delayed due to the increasing CPU time required to
simulate such large systems. A common solution for this problem of scale in MD
is coarse grain (CG) modeling, where atomistic accuracy is traded for speed.
This likewise comes with some subtle barriers between concept and practice.
For instance, to produce a CG model of a given polymer that is transferable across
state points, many simulations at various state points must be run and managed,
increasing the desirability of a reliable and easy way to keep track of these,
particularly for the multi-state iterative Boltzmann inversion (MSIBI)`@MooreMSIBI2014`
method of parameterization. All these aspects complicate and prolong the algready time-
and labor-intensive process of training new computational researchers
to perform sufficiently many simulations to meaningfully investigate polymer systems.

These factors indicate a need for a streamlined tool for creating (optionally CG)
MD polymer systems, launching one or many simulations across state points and/or
system type, and managing and analyzing the results. Ideally such a tool should be
approachable for novice programmers, to maximize time spent investigating research
questions and minimize the learning curve for new researchers, while preserving
the accuracy and rigor of the resultant research.

# Summary

Polybinder, the suite of tools introduced here, was built to enable materials scientists
to use molecular simulation to quickly and reproducibly simulate
large, coarse- or fine-grained polymer systems to investigate and predict trends in
their properties, all with a much lower barrier to entry than starting from scratch. Because it is designed
with modularity in mind, it will also ease adoption by other research groups,
and quicken the investigation process of new materials systems.

Polybinder is a Python package that uses the [foyer](https://github.com/mosdef-hub/foyer/),
[mbuild](https://github.com/mosdef-hub/mbuild/), and [signac](https://github.com/mosdef-hub/signac) packages from
the [MoSDeF suite of tools](https://github.com/mosdef-hub/) to quickly, easily, and reproducibly initialize and run polymer
simulations in the [HOOMD-blue](https://github.com/glotzerlab/hoomd-blue) engine.
Polybinder was made with the TRUE principles in mind `@JankowskiTRUE2020`, with the goal of allowing ease
of use and adoption, and reducing the learning curve for starting simulations.
This package allows for a variety of simulation types of interest,
such as bulk thermoplastic melts, annealing, welding interface interpenetration, and tensile testing of
the resultant welded systems.
Presently polybinder has methods for three common thermoplastic polymer chemistries:
polyether ether ketone (PEEK), polyether ketone ketone (PEKK),
and polyphenylene sulfide (PPS). However, polybinder is designed such that any
monomer units can be implemented and added to the internal library of available
structures via mbuild.


# Accessing the Software

Polybinder is freely available under the GNU General Public License (version 3) on [github](https://github.com/cmelab/polybinder).

# Acknowledgements

We acknowledge contributions from [ULI Advisory board, NASA, etc]

# References