import hoomd
import numpy as np
import rowan
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


class AnisotropicCustomForce(hoomd.md.force.Custom):
    def __init__(
        self, force_model, torque_model, box_len, num_neighbors, device
    ):
        super().__init__(aniso=True)
        self.device = device
        # load force model
        self.force_model = force_model
        self.force_model.to(self.device)
        self.force_model.eval()

        # load torque model
        self.torque_model = torque_model
        self.torque_model.to(self.device)
        self.torque_model.eval()

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
            orientations_q = np.array(
                snap.particles.orientation[rtags], copy=True
            )
        orientation_R = rowan.to_matrix(orientations_q)
        # get neighbor list
        neighbor_list = self._find_neighbors(positions)
        # preparing torch tensors
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
        orientation_R_tensor = (
            torch.from_numpy(orientation_R)
            .type(torch.FloatTensor)
            .to(self.device)
            .unsqueeze(0)
        )
        with torch.no_grad():
            predicted_force = (
                self.force_model(
                    positions_tensor, orientation_R_tensor, neighbor_list_tensor
                )
                .detach()
                .cpu()
                .numpy()
            )
            predicted_torque = (
                self.torque_model(
                    positions_tensor, orientation_R_tensor, neighbor_list_tensor
                )
                .detach()
                .cpu()
                .numpy()
            )

        with self.cpu_local_force_arrays as arrays:
            with self._state.cpu_local_snapshot as snap:
                rtags = np.array(snap.particles.rtag, copy=False)
                arrays.force[rtags] = predicted_force
                arrays.torque[rtags] = predicted_torque
