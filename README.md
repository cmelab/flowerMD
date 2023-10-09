[![pytest](https://github.com/cmelab/flower/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/flower/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/cmelab/flower/branch/main/graph/badge.svg?token=86LY9WHSH6)](https://codecov.io/gh/cmelab/flower)
## flower: Flexible Library of Organic Workflows and Extensible Recipes
flower is a modular “wrapper” package for molecular dynamics (MD)
simulation pipeline development, designed to enable fast, reproducible,
end-to- end simulation workflows with minimal user effort. This package is a
wrapper for [MoSDeF](https://github.com/mosdef-hub) packages and
[Hoomd-Blue](https://github.com/glotzerlab/hoomd-blue) with a focus on
simulating soft matter systems.

An object-oriented design makes flower extensible and highly flexible.
This is bolstered by a library-based approach to system initialization, making
flower agnostic to system identity, forcefield, and thermodynamic
ensemble, and allowing for growth on an as-needed basis.



## Installation

### 1. Clone this repository: ###

```
git clone git@github.com:cmelab/flower.git
cd flower
```

### 2. Set up and activate environment: ###
#### a. Using HOOMD-blue from conda:
```
conda env create -f environment-cpu.yml
conda activate flower
python -m pip install .
```

## Basic Usage
Please check out the [tutorials](tutorials) for a detailed description of
how to use flower and what functionalities it provides.

## Documentation
Documentation is available at [https://flower.readthedocs.io](https://flower.readthedocs.io)

[//]: # (#### Using the built in molecules, systems and forcefields:)

[//]: # (README, documentation and tutorials are a work in progress.)
