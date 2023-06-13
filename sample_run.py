
import hoomd.md

from hoomd_polymers.library import PPS

cell = hoomd.md.nlist.Cell(buffer=0.4)
lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('C', 'C')] = dict(epsilon=0.35, sigma=0.29)
lj.r_cut[('C', 'C')] = 2.5
lj.params[('C', 'S')] = dict(epsilon=0.35, sigma=0.65)
lj.r_cut[('C', 'S')] = 2.5
lj.params[('S', 'S')] = dict(epsilon=0.35, sigma=0.65)
lj.r_cut[('S', 'S')] = 2.5
lj.params[('S', 'H')] = dict(epsilon=0.35, sigma=0.65)
lj.r_cut[('S', 'H')] = 2.5
lj.params[('H', 'H')] = dict(epsilon=0.35, sigma=0.65)
lj.r_cut[('H', 'H')] = 2.5
lj.params[('C', 'H')] = dict(epsilon=0.35, sigma=0.65)
lj.r_cut[('C', 'H')] = 2.5

bond = hoomd.md.bond.Harmonic()
bond.params['C-C'] = dict(k=3.0, r0=2.38)
bond.params['C-H'] = dict(k=3.0, r0=2.38)
bond.params['C-S'] = dict(k=3.0, r0=2.38)
bond.params['S-H'] = dict(k=3.0, r0=2.38)

angle = hoomd.md.angle.Harmonic()
angle.params['C-C-C'] = dict(k=3.0, t0=0.7851)
angle.params['C-S-C'] = dict(k=3.0, t0=0.7851)
angle.params['S-C-C'] = dict(k=3.0, t0=0.7851)
angle.params['S-C-H'] = dict(k=3.0, t0=0.7851)
angle.params['C-C-H'] = dict(k=3.0, t0=0.7851)
angle.params['H-S-C'] = dict(k=3.0, t0=0.7851)

harmonic = hoomd.md.dihedral.Periodic()
harmonic.params['C-C-C-C'] = dict(k=3.0, d=-1, n=3, phi0=0)
harmonic.params['C-C-C-H'] = dict(k=100.0, d=1, n=4, phi0=0)
harmonic.params['H-C-C-H'] = dict(k=3.0, d=-1, n=3, phi0=0)
harmonic.params['C-C-C-S'] = dict(k=100.0, d=1, n=4, phi0=0)
harmonic.params['C-C-S-C'] = dict(k=3.0, d=-1, n=3, phi0=0)
harmonic.params['S-C-C-H'] = dict(k=100.0, d=1, n=4, phi0=0)
harmonic.params['C-C-S-H'] = dict(k=100.0, d=1, n=4, phi0=0)

coulombic = hoomd.md.long_range.pppm.make_pppm_coulomb_forces(
    nlist=cell, resolution=(0, 0, 1), order=1, r_cut=2
)
forcefield = [lj, bond, angle, harmonic,coulombic[1]]
pps = PPS(lengths=3, n_mols=2, force_field=forcefield, remove_hydrogens=False)
# pps = PPS(lengths=3, n_mols=2, force_field="pps_opls")
# system = Pack(molecules=pps.molecules, density=0.01)
# forcefield = Forcefield(forcefield_files="/Users/Marjan/Documents/projects/hoomd-polymers/hoomd_polymers/library/forcefields/pps_opls.xml")
# system.apply_forcefield(forcefield=forcefield)
print(pps)