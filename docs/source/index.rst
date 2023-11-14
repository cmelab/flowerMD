flowerMD: Flexible Library of Organic Workflows and Extensible Recipes for Molecular Dynamics
=============================================================================================


flowerMD is a modular “wrapper” package for molecular dynamics (MD)
simulation pipeline development, designed to enable fast, reproducible, end-to-
end simulation workflows with minimal user effort. An object-oriented design
makes flowerMD extensible and highly flexible. This is bolstered by a
library-based approach to system initialization, making flowerMD agnostic
to system identity, forcefield, and thermodynamic ensemble, and allowing
for growth on an as-needed basis.

Why flowerMD?
=============
flowerMD consolidates and automates end-to-end workflows for modeling complex engineering
processes, with a focus on emulating physical experiments on organic materials.
Following the principals of Transparent, Reproducible, Usable by others, and Extensible
[TRUE](https://doi.org/10.1016/j.commatsci.2019.109129) software design,
the modular design of flowerMD facilitates building and running
workflows for specific materials science research applications, while reducing the
cognitive load and programming demands on the user's part. flowerMD addresses a
longstanding need in moelcular dynamics simulation workflow building: reproducible and
low-complexity recipes for specifying research workflows, agnostic to specific system
identity.

In particular, flowerMD bridges the gaps between the steps of specifying molecules,
parametrizing and applying force fields, and launching simulations into one seamless
workflow, while remaining flexible enough to change any individual step without
necessitating changes elsewhere in the worklow. The recipe-book-style approach of
flowerMD reduces the amount of work needed to implement otherwise similar simulations
with different molecular species, different molecular representations (i.e. coarse vs
fine-grained), different force fields, or different state points.

Quick start
===========
.. toctree::
    installation


Resources
=========

`GitHub Repository <https://github.com/cmelab/flowerMD>`_: Source code and issue tracker.

`Tutorials <https://github.com/cmelab/flowerMD/tree/main/tutorials>`_: Examples of how to use flowerMD.


.. toctree::
   :maxdepth: 2
   :caption: Python API

   base
   modules
   library



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
