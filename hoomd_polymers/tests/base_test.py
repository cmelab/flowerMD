import os

import pytest

from hoomd_polymers.systems import *
from hoomd_polymers.molecules import *
from hoomd_polymers.forcefields import *


class BaseTest:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    @pytest.fixture()
    def polyethylene_system(self):
        system = Pack(
                molecule=PolyEthylene,
                n_mols=5,
                mol_kwargs={"length": 5},
                density=0.5
        )
        system.apply_forcefield(forcefield=GAFF(), remove_hydrogens=False)
        return system

    @pytest.fixture()
    def ua_polyethylene_system(self):
        system = Pack(
                molecule=PolyEthylene,
                n_mols=5,
                mol_kwargs={"length": 5},
                density=0.5
        )
        system.apply_forcefield(forcefield=GAFF(), remove_hydrogens=True)
        return system
