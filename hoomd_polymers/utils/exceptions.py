class MissingPotentialError(Exception):
    def __init__(self, connection, potential_class):
        self.connection = connection
        self.potential_class = potential_class
        msg = f"Missing potential for {self.connection} {self.potential_type} in {self.potential_class}."
        super().__init__(msg)

    @property
    def potential_type(self):
        raise NotImplementedError


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

