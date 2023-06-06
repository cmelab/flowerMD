class MissingPotentialError(Exception):
    pass


class MissingPairPotentialError(MissingPotentialError):
    def __init__(self, pair, type):
        msg = f"Missing pair potential for {pair} in {type}"
        super().__init__(msg)