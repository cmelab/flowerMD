import hoomd
import numpy as np


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
        box = self._state.box
        self.sim._update_walls()
