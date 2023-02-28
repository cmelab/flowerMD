import hoomd
import numpy as np


class ScaleEpsilon(hoomd.custom.Action):
    def __init__(self, sim, scale_factor):
        self.scale_factor = scale_factor
        self.sim = sim

    def act(self, timestep):
        self.sim.scale_epsilon(shift_by=self.scale_factor)


class ScaleSigma(hoomd.custom.Action):
    def __init__(self, sim, scale_factor):
        self.scale_factor = scale_factor
        self.sim = sim

    def act(self, timestep):
        self.sim.scale_sigma(shift_by=self.scale_factor)
        lj_forces = self.sim._lj_force()
        for param in lj_forces.params.keys():
            print(lj_forces.params[param]["sigma"])


class PullParticles(hoomd.custom.Action):
    def __init__(self, shift_by, axis, neg_filter, pos_filter):
        self.shift_by = shift_by
        self.axis = axis
        self.neg_filter = neg_filter
        self.pos_filter = pos_filter

    def act(self, timestep):
        with self._state.cpu_local_snapshot as snap:
            neg_filter = snap.particles.rtag[self.neg_filter.tags]
            pos_filter = snap.particles.rtag[self.pos_filter.tags]
            snap.particles.position[neg_filter] -= (self.shift_by*self.axis)
            snap.particles.position[pos_filter] += (self.shift_by*self.axis)


class UpdateWalls(hoomd.custom.Action):
    def __init__(self, sim):
        self.sim = sim

    def act(self, timestep):
        self.update_walls()

    def update_walls(self):
        for wall_axis in self.sim._wall_forces:
            wall_force = self.sim._wall_forces[wall_axis][0]
            wall_kwargs = self.sim._wall_forces[wall_axis][1]
            self.sim.remove_force(wall_force)
            self.sim.add_walls(wall_axis, **wall_kwargs)
