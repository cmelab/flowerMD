import freud
import numpy as np
import torch


def find_neighbors(positions, box_len, num_neighbors):
    box = freud.box.Box.cube(box_len)
    aq = freud.locality.AABBQuery(box, positions)
    query_points = positions
    query_result = aq.query(
        query_points, dict(num_neighbors=num_neighbors, exclude_ii=True)
    )
    nlist = query_result.toNeighborList()
    neighbor_list = np.asarray(
        list(zip(nlist.query_point_indices, nlist.point_indices))
    )
    return neighbor_list


def neighbors_dr(
    positions: torch.Tensor, neighbor_list: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the distance vector between particles and their neighbors.
    """
    B, N, _ = positions.shape
    N_neighbors = neighbor_list.shape[-1]

    NN_repeated = neighbor_list.unsqueeze(-1).expand((B, N, N_neighbors, 3))
    positions_repeated = positions[:, :, None, :].expand(
        (-1, -1, N_neighbors, -1)
    )
    neighbor_positions = torch.gather(
        positions_repeated, dim=1, index=NN_repeated
    )
    dr = positions_repeated - neighbor_positions  # (B, N, N_neighbors, 3)

    return dr


def adjust_periodic_boundary(dr: torch.Tensor, box_len: float) -> torch.Tensor:
    """
    Adjust the distance vector for periodic boundary conditions.
    """
    half_box_len = box_len / 2
    dr = torch.where(dr > half_box_len, dr - box_len, dr)
    dr = torch.where(dr < -half_box_len, dr + box_len, dr)
    return dr
