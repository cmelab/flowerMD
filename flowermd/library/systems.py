"""Examples for the Systems class."""

import mbuild as mb
import numpy as np
from scipy.spatial.distance import pdist

from flowermd.base.system import System


class SingleChainSystem(System):
    """Builds a vacuum box around a single chain.

    The box lengths are chosen so they are at least as long as the largest particle distance.
    The maximum distance of the chain is calculated using scipy.spatial.distance.pdist().
    This distance multiplied by a buffer defines the box dimensions. The chain is centered in the box.

    Parameters
    ----------
    buffer : float, default 1.05
        A factor to multiply box dimensions. Must be greater than 1 so that the particles are inside the box.

    """

    def __init__(self, molecules, base_units=dict(), buffer=1.05):
        self.buffer = buffer
        super(SingleChainSystem, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        if len(self.all_molecules) > 1:
            raise ValueError(
                "This system class only works for systems contianing a single molecule."
            )
        chain = self.all_molecules[0]
        eucl_dist = pdist(self.all_molecules[0].xyz)
        chain_length = np.max(eucl_dist)
        box = mb.Box(lengths=np.array([chain_length] * 3) * self.buffer)
        comp = mb.Compound()
        comp.add(chain)
        comp.box = box
        chain.translate_to((box.Lx / 2, box.Ly / 2, box.Lz / 2))
        return comp
'''
class RandomWalk(System):
    """Places chain segments in a consecutive random walk. Optional self-avoiding parameter or shape constraint using mbuild.

    Parameters
    ----------
    

    """

    def __init__(self, molecules, base_units=dict(),num_mon,num_pol,radius,bond_length,density):
        self.num_mon = num_mon
        self.num_pol = num_pol
        self.radius = radius
        self.bond_length = bond_length
        self.density = density
        super(RandomWalk, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        N = self.num_mon * self.num_pol
        L = np.cbrt(N / self.density)
        cube = CuboidConstraint(center=(0,0,0), Lx=L, Ly=L, Lz=L)
        last_path = None
        print("Starting walk")
        for walk_num in range(self.num_pol):
            walk_passed = False
            while not walk_passed:
                try:
                    if walk_num == 0:
                        initial_point = (0,0,0)
                    else:
                      initial_point = find_low_density_point(
                            points=last_path.coordinates,
                            box_min=cube.mins,
                            box_max=cube.maxs,
                            edge_buffer=self.radius,
                            n_candidates=500,
                        )[np.random.randint(10)]
                    
                    path = HardSphereRandomWalk(
                        N=self.num_mon,
                        bead_name=f"_A{walk_num}",
                        radius=self.radius,
                        volume_constraint=cube,
                        bond_length=self.bond_length,
                        min_angle=np.pi/2,
                        max_angle=np.pi,
                        max_attempts=1e4,
                        start_from_path=last_path,
                        attach_paths=False,
                        seed=None,
                        initial_point=initial_point,
                        trial_batch_size=100
                    )
    
                    last_path = path
                    walk_passed = True
                except Exception as e:
                    print(e)
        
        polymer_system = last_path.to_compound()
        positions = polymer_system.xyz
        print("Finished walk")
    
        return polymer_system

class DPDBuilder(System):
    """Runs a DPD system to relax an intial configuration of overlapping particles.

    Parameters
    ----------

    """

    def __init__(self, molecules, base_units=dict(),A=1000,gamma=1000,k=1000,num_pol=100,num_mon=10,kT=1.0,r_cut = 1.15,bond_length=1.0,dt=0.001,density=0.8,particle_spacing = 1.1,minimum_distance=0.95)
        self.gamma=gamma,
        self.k=k,
        self.num_pol=num_pol,
        self.num_mon=num_mon,
        self.kT=kT,
        self.r_cut = r_cut,
        self.bond_length=bond_length,
        self.dt=dt,
        self.density=density,
        self.particle_spacing = particle_spacing
        self.minimum_distance = minimum_distance
        super(DPDBuilder, self).__init__(
            molecules=molecules, base_units=base_units
        )

    def _build_system(self):
        frame = RandWalk(self.num_pol, self.num_mon, density=self.density) #change this to class function above
        harmonic = hoomd.md.bond.Harmonic()
        harmonic.params["b"] = dict(r0=self.bond_length, k=self.k)
        integrator = hoomd.md.Integrator(dt=self.dt)
        integrator.forces.append(harmonic)
        simulation = hoomd.Simulation(device=hoomd.device.auto_select(), seed=np.random.randint(65535))# TODO seed
        simulation.operations.integrator = integrator 
        simulation.create_state_from_snapshot(frame)
        const_vol = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        integrator.methods.append(const_vol)
        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        simulation.operations.nlist = nlist
        DPD = hoomd.md.pair.DPD(nlist, default_r_cut=self.r_cut, kT=self.kT)
        DPD.params[('A', 'A')] = dict(A=self.A, gamma=self.gamma)
        integrator.forces.append(DPD)
        
        simulation.run(0)
        simulation.run(1000)
        snap=simulation.state.get_snapshot()
        
        while not check_bond_length_equilibration(snap,self.num_mon, self.num_pol,max_bond_length=self.particle_spacing):
            simulation.run(1000)
            snap=simulation.state.get_snapshot()
            #add loop exit based on simulation wall time or number of steps
    
        while not check_inter_particle_distance(snap,minimum_distance=self.minimum_distance):
            simulation.run(1000)
            snap=simulation.state.get_snapshot()
            #add loop exit based on simulation wall time or number of steps
        
        return snap
'''
