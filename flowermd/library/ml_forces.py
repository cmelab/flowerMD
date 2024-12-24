import hoomd
import numpy as np
import torch

from flowermd.internal import find_neighbors


class IsotropicCustomForce(hoomd.md.force.Custom):
    """Custom force class for getting forces from a torch model."""

    def __init__(self, model, box_len, num_neighbors, device):
        super().__init__(aniso=False)
        # pytorch model
        self.model = model
        self.model.eval()
        self.device = device
        self.box_len = box_len
        self.num_neighbors = num_neighbors

    def _find_neighbors(self, positions):
        neighbor_list = find_neighbors(
            positions, self.box_len, self.num_neighbors
        )
        return neighbor_list

    def set_forces(self, timestep):
        # get positions and orientations
        with self._state.cpu_local_snapshot as snap:
            rtags = np.array(snap.particles.rtag, copy=False)
            positions = np.array(snap.particles.position[rtags], copy=True)

        # get neighbor list
        neighbor_list = self._find_neighbors(positions)
        positions_tensor = (
            torch.as_tensor(positions)
            .type(torch.FloatTensor)
            .to(self.device)
            .unsqueeze(0)
        )
        neighbor_list_tensor = (
            torch.from_numpy(neighbor_list.astype(np.int64))
            .to(self.device)
            .unsqueeze(0)
        )
        positions_tensor.requires_grad = True
        predicted_force = (
            self.model(positions_tensor, neighbor_list_tensor)
            .detach()
            .cpu()
            .numpy()
        )

        with self.cpu_local_force_arrays as arrays:
            with self._state.cpu_local_snapshot as snap:
                rtags = np.array(snap.particles.rtag, copy=False)
                arrays.force[rtags] = predicted_force
        del (
            predicted_force,
            neighbor_list_tensor,
            positions_tensor,
            rtags,
            positions,
        )
