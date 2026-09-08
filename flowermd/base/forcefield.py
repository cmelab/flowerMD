"""Base forcefield classes."""

import forcefield_utilities as ffutils
from gmso.core.forcefield import ForceField


class BaseXMLForcefield:
    """Base XML forcefield class."""

    def __init__(self, forcefield_files=None, name=None, gmso_xml=None):
        self.forcefield_files = forcefield_files
        self.name = name
        self.gmso_xml = gmso_xml
        if all([name, forcefield_files]):
            raise ValueError("Give only one of `name` or `forcefield_files`.")
        if self.gmso_xml is True:
            self.gmso_ff = ForceField(forcefield_files or name)

        else:
            self.gmso_ff = (
                ffutils.FoyerFFs().load(forcefield_files or name).to_gmso_ff()
            )


class BaseHOOMDForcefield:
    """Base HOOMD forcefield class."""

    def __init__(self, hoomd_forces):
        self.hoomd_forces = hoomd_forces
        if hoomd_forces is None:
            raise NotImplementedError(
                "`hoomd_forces` must be defined in the subclass."
            )
        if not isinstance(hoomd_forces, list):
            raise TypeError("`hoomd_forces` must be a list.")
