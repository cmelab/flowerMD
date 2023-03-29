[![pytest](https://github.com/chrisjonesbsu/hoomd-polymers/actions/workflows/pytest.yml/badge.svg)](https://github.com/chrisjonesbsu/hoomd-polymers/actions/workflows/pytest.yml)
[![.github/workflows/codecov.yml](https://github.com/chrisjonesBSU/hoomd-polymers/actions/workflows/codecov.yml/badge.svg?branch=main)](https://github.com/chrisjonesBSU/hoomd-polymers/actions/workflows/main.yml)
## About

Wrapper for [MoSDeF](https://github.com/mosdef-hub) packages and [Hoomd-Blue](https://github.com/glotzerlab/hoomd-blue) with 
a focus on simulating soft matter systems.

## Installation

### 1. Clone this repository: ###  

```
git clone git@github.com:chrisjonesbsu/hoomd-polymers.git  
cd hoomd-polymers  
```

### 2. Set up and activate environment: ###  
#### a. Using HOOMD-blue from conda:
```
conda env create -f environment-cpu.yml  
conda activate hoomd-polymers 
python -m pip install .
```  

## Basic Usage
#### Using the built in molecules, systems and forcefields:
```
from hoomd_polymers.molecules import PolyEthylene
from hoomd_polymers.systems import Pack
from hoomd_polymers.forcefields import GAFF
from hoomd_polymers.sim import Simulation

pe_system = Pack(
        molecule=PolyEthylene,
        density=1.0,
        n_mols=[20],
        chain_lengths=[10]
)

pe_system.apply_forcefield(forcefield=GAFF())

pe_sim = Simulation(
        initial_state=pe_system.hoomd_snapshot,
        forcefield=pe_system.hoomd_forcefield
)
pe_sim.run_NVT(kT=3.0, tau_kT=0.01, n_steps=1e6)
```

#### Using with your own molecule and forcefield:
```
import foyer
from hoomd_polymers.systems import Pack
from hoomd_polymers.sim import Simulation

def my_molecule(file_path):
    return mb.load(file_path)

system = Pack(
        molecule=my_molecule,
        molecule_kwargs={"file_path": "molecule.mol2"},
        n_mols=[20]

my_ff = foyer.Forcefield(forcefield_files="path-to-ff.xml")
system.apply_forcefield(forcefield=my_ff)

sim = Simulation(
        initial_state=system.hoomd_snapshot,
        forcefield=system.hoomd_forcefield
)
sim.run_NVT(kT=3.0, tau_kT=0.01, n_steps=1e6)
```
