[![pytest](https://github.com/cmelab/JankFlow/actions/workflows/pytest.yml/badge.svg)](https://github.com/cmelab/JankFlow/actions/workflows/pytest.yml)
[![codecov](https://codecov.io/gh/cmelab/JankFlow/branch/main/graph/badge.svg?token=86LY9WHSH6)](https://codecov.io/gh/cmelab/JankFlow)
## JankFlow: Flexible Library of Organic Workflows
JankFlow is a modular “wrapper” package for molecular dynamics (MD)
simulation pipeline development, designed to enable fast, reproducible,
end-to- end simulation workflows with minimal user effort. This package is a
wrapper for [MoSDeF](https://github.com/mosdef-hub) packages and
[Hoomd-Blue](https://github.com/glotzerlab/hoomd-blue) with a focus on
simulating soft matter systems.

An object-oriented design makes JankFlow extensible and highly flexible.
This is bolstered by a library-based approach to system initialization, making
JankFlow agnostic to system identity, forcefield, and thermodynamic
ensemble, and allowing for growth on an as-needed basis.



## Installation

### 1. Clone this repository: ###

```
git clone git@github.com:cmelab/JankFlow.git
cd JankFlow
```

### 2. Set up and activate environment: ###
#### a. Using HOOMD-blue from conda:
```
conda env create -f environment-cpu.yml
conda activate jankflow
python -m pip install .
```

## Basic Usage
Please checkout the [tutorials](tutorials) for a detailed description of
how to use JankFlow and what functionalities it provides.

## Documentation
Documentation is available at [https://hoomd-organics.readthedocs.io](https://hoomd-organics.readthedocs.io)

[//]: # (#### Using the built in molecules, systems and forcefields:)

[//]: # (README, documentation and tutorials are a work in progress.)
