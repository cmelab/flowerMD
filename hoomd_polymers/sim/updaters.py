import hoomd

class PullParticles(hoomd.custom.Action):
    def __init__(self, shift_by, axis, neg_filter, pos_filter):
        self.shift_by = shift_by
        self.axis = axis
        self.neg_filter = neg_filter
        self.pos_filter = pos_filter

    def act(self, timestep):
        snap = self._state.get_snapshot()
        snap.particles.position[self.neg_filter.tags] -= (self.shift_by*self.axis)
        snap.particles.position[self.pos_filter.tags] += (self.shift_by*self.axis)
        self._state.set_snapshot(snap)
