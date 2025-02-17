import torch

from flowermd.internal import adjust_periodic_boundary


def neighbors_distance_vector(positions, neighbor_list):
    """
    Calculate the distance vector between particles and their neighbors
    Parameters
    ----------
    positions: particle positions (B, N, 3)
    neighbor_list: list of neighbors for each particle (B, N, N_neighbors)

    Returns
    -------
    dr: distance vector between particles and their neighbors (B, N, N_neighbors, 3)
    """
    B = positions.shape[0]
    N = positions.shape[1]
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


def orientation_dot_product(particle_orientations, neighbors_orientations):
    """
    Calculate the dot product of the principal axis of the particle orientation
    with the principal axis of the neighbors orientations.

    Parameters
    ----------
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N, N_neighbors, 3, 3)

    Returns
    -------
    dot product (B, N, N_neighbors, 3, 3)

    """
    dot_prod = torch.einsum(
        "ijkhl, ijkhm -> ijklm", particle_orientations, neighbors_orientations
    )  # (B, N, N_neighbors, 3, 3)

    return dot_prod


def orientation_element_product(particle_orientations, neighbors_orientations):
    """
    Calculate the element wise product of the particle orientation columns with
    the neighbors orientations columns.

    Parameters
    ----------
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N, N_neighbors, 3, 3)

    Returns
    -------
    element wise product (B, N, N_neighbors, 3, 3, 3)

    """
    element_prod = torch.einsum(
        "ijkhl, ijkhm -> ijklmh", particle_orientations, neighbors_orientations
    )
    return element_prod


def orientation_principal_cross_product(
    particle_orientations, neighbors_orientations
):
    """
    Calculate the cross product of the principal axis of the particle orientation
    with the principal axis of the neighbors orientations.

    Parameters
    ----------
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N, N_neighbors, 3, 3)

    Returns
    -------
    cross product (B, N, N_neighbors, 3, 3)

    """
    cross_prod = torch.cross(
        torch.transpose(particle_orientations, -1, -2),
        torch.transpose(neighbors_orientations, -1, -2),
        dim=-1,
    )

    return cross_prod


def relative_orientation(particle_orientations, neighbors_orientations):
    """
    Calculate the relative orientation between the particle and its neighbors.

    Parameters
    ----------
    particle_orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)
    neighbors_orientations: neighbors orientation rotation matrix
     (B, N, N_neighbors, 3, 3)

    Returns
    -------
    relative orientation (B, N, N_neighbors, 3, 3)

    """
    relative_orientation = torch.matmul(
        particle_orientations, neighbors_orientations.transpose(-1, -2)
    )
    return relative_orientation


def RBF_dr_orientation(dr, orientations):
    """
    Calculate the RBF of the distance vector and the orientation vector.

    Parameters
    ----------
    dr: distance vector between particles and their neighbors (B, N, N_neighbors, 3)
    orientations: particle orientation rotation matrix repeated along
     the neighbor axis (B, N, N_neighbors, 3, 3)

    Returns
    -------
    RBF (B, N, N_neighbors, 3)

    """
    dr_broadcast = dr.unsqueeze(-1).expand(-1, -1, -1, 3, 3)
    cross = torch.cross(
        dr_broadcast, torch.transpose(orientations, -1, -2), dim=-1
    )
    norm_sq = torch.pow(torch.norm(cross, dim=-1), 2)
    rbf = torch.exp(-norm_sq)
    return rbf


def orientation_dependent_features(
    position, orientation_R, neighbor_list, box_len, device
):
    """

    Parameters
    ----------
    position: particle positions (B, N, 3)
    orientation_R: particle orientation rotation matrix (B, N, 3, 3)
    neighbor_list: list of neighbors for each particle (B, N * N_neighbors, 2)

    Returns
    -------

    """
    batch_size = position.shape[0]
    N_particles = position.shape[1]

    # change tuple based neighbor list to (B, N, neighbor_idx)
    neighbor_list = neighbor_list.reshape(
        batch_size, N_particles, -1, neighbor_list.shape[-1]
    )[:, :, :, 1].to(device)  # (B, N, N_neighbors)
    N_neighbors = neighbor_list.shape[-1]
    dr = neighbors_distance_vector(
        position, neighbor_list
    )  # (B, N, N_neighbors, 3)
    dr = adjust_periodic_boundary(dr, box_len)

    R = torch.norm(dr, dim=-1, keepdim=True)  # (B, N, N_neighbors, 1)

    inv_R = 1.0 / R  # (B, N, N_neighbors, 1)

    ################ orientation related features ################

    # repeat the neighbors idx to match the shape of orientation_R. This
    # is necessary to gather each particle's neighbors' orientation
    NN_expanded = neighbor_list[:, :, :, None, None].expand(
        (-1, -1, -1, 3, 3)
    )  # (B, N, N_neighbors, 3, 3)
    # repeart the orientation_R to match the shape of neighbor_list
    orientation_R_expanded = orientation_R[:, :, None, :, :].expand(
        (-1, -1, N_neighbors, -1, -1)
    )  # (B, N, N_neighbors, 3, 3)
    # get all neighbors' orientation
    neighbors_orient_R = torch.gather(
        orientation_R_expanded, dim=1, index=NN_expanded
    )  # (B, N, N_neighbors, 3, 3)

    # dot product: (B, N, N_neighbors, 3, 3)
    orient_dot_prod = orientation_dot_product(
        orientation_R_expanded, neighbors_orient_R
    )

    # element product: (B, N, N_neighbors, 3, 3, 3)
    orient_element_prod = orientation_element_product(
        orientation_R_expanded, neighbors_orient_R
    )
    # element product norm: (B, N, N_neighbors, 3, 3)
    element_prod_norm = torch.norm(orient_element_prod, dim=-1)

    # principal cross product: (B, N, N_neighbors, 3, 3)
    orient_cross_prod = orientation_principal_cross_product(
        orientation_R_expanded, neighbors_orient_R
    )
    # cross product norm: (B, N, N_neighbors, 3)
    cross_prod_norm = torch.norm(orient_cross_prod, dim=-1)

    # relative orientation: (B, N, N_neighbors, 3, 3)
    rel_orient = relative_orientation(
        orientation_R_expanded, neighbors_orient_R
    )

    ################ RBF features ################

    # RBF for particles:(B, N, N_neighbors, 3)
    rbf_particle = RBF_dr_orientation(dr, orientation_R_expanded)
    # RBF for neighbors: (B, N, N_neighbors, 3)
    rbf_neighbors = RBF_dr_orientation(dr, neighbors_orient_R)

    # concatenate all features (B, N, N_neighbors, 77)
    features = torch.cat(
        (
            R,
            dr / R,
            inv_R,
            orient_dot_prod.flatten(start_dim=-2),
            orient_element_prod.flatten(start_dim=-3),
            element_prod_norm.flatten(start_dim=-2),
            orient_cross_prod.flatten(start_dim=-2),
            cross_prod_norm,
            rel_orient.flatten(start_dim=-2),
            rbf_particle,
            rbf_neighbors,
        ),
        dim=-1,
    )

    return features.to(device), R
