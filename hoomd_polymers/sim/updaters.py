from hoomd.custom import Action

class PullParticles(Action):
    def __init__(self, shift_by, axis, neg_filter, pos_filter):
        self.shift_by = shift_by
        self.axis = axis
        self.neg_filter = neg_filter
        self.pos_filter = pos_filter

    def act(self, timestep):
        snap = self._state.get_snapshot()
        snap.particles.position[self.neg_filter.tags][self.axis] -= self.shift_by
        snap.particles.position[self.pos_filter.tags][self.axis] += self.shift_by
        self._state.set_snapshot(snap)
