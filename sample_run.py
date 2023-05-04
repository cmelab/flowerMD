
from hoomd_polymers.library import PPS

pps = PPS(lengths=4, n_mols=2, force_field="pps_opls")
print(pps)