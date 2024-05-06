class MissingPotentialError(Exception):
    def __init__(self, connection="", potential_class=None):
        self.connection = connection
        self.potential_class = potential_class
        msg = self._generate_msg()
        super().__init__(msg)

    def _generate_msg(self):
        return (
            f"Missing {self.potential_class} potential for"
            f" {self.connection} {self.potential_type}"
        )

    @property
    def potential_type(self):
        return None


class MissingPairPotentialError(MissingPotentialError):
    @property
    def potential_type(self):
        return "pair"


class MissingBondPotentialError(MissingPotentialError):
    @property
    def potential_type(self):
        return "bond"


class MissingAnglePotentialError(MissingPotentialError):
    @property
    def potential_type(self):
        return "angle"


class MissingDihedralPotentialError(MissingPotentialError):
    @property
    def potential_type(self):
        return "dihedral"


class MissingCoulombPotentialError(MissingPotentialError):
    def _generate_msg(self):
        return (
            f"Missing Coulomb force {self.potential_class} "
            f"for electrostatic interactions."
        )


class MoleculeLoadError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class UnitError(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class ForceFieldError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
