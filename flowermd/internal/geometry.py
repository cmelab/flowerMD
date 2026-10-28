"""Small geometry helpers shared by the placement and stereochemistry code.

Every function broadcasts over leading axes, so one call handles many
candidate conformations at once.
"""

import numpy as np


def turn_about_bond(xyz_a, xyz_b, angle):
    """Turn by `angle` (radians) about the line from `xyz_a` to `xyz_b`.

    Returned as a pose (R, t), x -> R x + t, with t = a - R a, so that
    x -> R (x - a) + a: rotate about the bond direction, keeping a fixed.
    """
    k = (xyz_b - xyz_a) / np.linalg.norm(xyz_b - xyz_a)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    # Rodrigues' formula as a matrix (cross @ y == np.cross(k, y))
    rotation = (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross
        + (1 - np.cos(angle)) * np.outer(k, k)
    )
    return rotation, xyz_a - rotation @ xyz_a


def apply_pose(pose, xyz):
    """Move the points `xyz` (..., 3) by the pose (R, t): x -> R x + t."""
    rotation, shift = pose
    return xyz @ rotation.T + shift  # row vectors: x R^T = (R x)^T


def compose(outer, inner):
    """The pose that moves points by `inner` first, then by `outer`."""
    (r_out, t_out), (r_in, t_in) = outer, inner
    return r_out @ r_in, r_out @ t_in + t_out


def dihedrals(p0, p1, p2, p3):
    """Signed dihedral angles p0-p1-p2-p3 (radians, IUPAC sign convention).

    The angle between the plane of p0, p1, p2 and the plane of p1, p2, p3,
    seen along the bond p1-p2: 180 degrees for trans, positive for a
    clockwise turn of p3 when looking from p1 toward p2.
    """
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)  # normals of the two planes
    length = np.linalg.norm(b2, axis=-1)
    return np.arctan2(
        length * np.sum(n1 * b3, axis=-1), np.sum(n1 * n2, axis=-1)
    )


def signed_volumes(edges):
    """Signed volume spanned by three edge vectors, shape (..., 3, 3).

    The sign is the handedness of a stereocenter whose edges run from one
    substituent to the other three: a mirror image flips it.
    """
    return np.linalg.det(edges)


def crosses(p, q, center, u, v):
    """Whether segment p-q crosses triangle (center, u, v); broadcasts.

    The Moeller-Trumbore test: solve p + t (q - p) = center + x e1 + y e2
    for (t, x, y). The segment crosses the triangle where the point is
    inside it (x, y >= 0, x + y <= 1) and between p and q (0 <= t <= 1).
    """
    e1, e2, d, s = u - center, v - center, q - p, p - center
    h, w = np.cross(d, e2), np.cross(s, e1)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.sum(e1 * h, axis=-1)
        x, y, t = (
            np.sum(m * n, axis=-1) / a for m, n in ((s, h), (d, w), (e2, w))
        )
    return (x >= 0) & (y >= 0) & (x + y <= 1) & (t >= 0) & (t <= 1)
