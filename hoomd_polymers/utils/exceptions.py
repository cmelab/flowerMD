class MissingPotentialError(Exception):
    pass


class MissingPairPotentialError(MissingPotentialError):
    def __init__(self, pair, potential_type):
        msg = f"Missing pair potential for {pair} pair in {potential_type}."
        super().__init__(msg)


class MissingBondPotentialError(MissingPotentialError):
    def __init__(self, bond, potential_type):
        msg = f"Missing bond potential for {bond} bond in {potential_type}."
        super().__init__(msg)


class MissingAnglePotentialError(MissingPotentialError):
    def __init__(self, angle, potential_type):
        msg = f"Missing angle potential for {angle} angle in {potential_type}."
        super().__init__(msg)
