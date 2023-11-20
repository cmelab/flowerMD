"""Utility functions for the surface wetting module."""

import copy

import hoomd
import numpy as np


def combine_forces(drop_forces, surface_forces, drop_ptypes, surface_ptypes):
    """Combine the surface and droplet forces."""
    combined_force_list = []

    # retrieve individual force objects from the droplet and surface
    # force lists
    drop_forces_dict = retrieve_forces(drop_forces)
    surface_forces_dict = retrieve_forces(surface_forces)

    # combine LJ forces
    combined_force_list.append(
        combine_lj_forces(
            drop_forces_dict["lj"],
            surface_forces_dict["lj"],
            drop_ptypes,
            surface_ptypes,
        )
    )

    # add Coulomb forces from droplet simulation if exists
    if drop_forces_dict.get("coulomb"):
        combined_force_list.append(drop_forces_dict["coulomb"])
    # combine bond forces
    combined_force_list.append(
        combine_bond_forces(
            drop_forces_dict["bond"], surface_forces_dict["bond"]
        )
    )
    # combine angle forces
    combined_force_list.append(
        combine_angle_forces(
            drop_forces_dict["angle"], surface_forces_dict["angle"]
        )
    )
    # combine dihedral forces
    combined_force_list.append(
        combine_dihedral_forces(
            drop_forces_dict["dihedral"], surface_forces_dict["dihedral"]
        )
    )
    return combined_force_list


def retrieve_forces(hoomd_force_list):
    """Retrieve individual force objects from the hoomd force list."""
    force_dict = {}
    for force in hoomd_force_list:
        if isinstance(force, hoomd.md.pair.LJ):
            force_dict["lj"] = force
        elif isinstance(force, hoomd.md.long_range.pppm.Coulomb):
            force_dict["coulomb"] = force
        elif isinstance(force, hoomd.md.bond.Harmonic):
            force_dict["bond"] = force
        elif isinstance(force, hoomd.md.angle.Harmonic):
            force_dict["angle"] = force
        elif isinstance(force, hoomd.md.dihedral.OPLS):
            force_dict["dihedral"] = force
    return force_dict


def combine_lj_forces(
    drop_lj, surface_lj, drop_ptypes, surface_ptypes, combining_rule="geometric"
):
    """Combine the droplet and surface LJ forces."""
    if combining_rule not in ["geometric", "lorentz"]:
        raise ValueError("combining_rule must be 'geometric' or 'lorentz'")

    lj = hoomd.md.pair.LJ(nlist=drop_lj.nlist)

    # add droplet LJ parameters
    for drop_pair, drop_params in drop_lj.params.items():
        lj.params[drop_pair] = drop_params
        lj.r_cut[drop_pair] = drop_lj.r_cut[drop_pair]

    # add the surface LJ parameters to the droplet LJ parameters
    for surface_pair, surface_params in surface_lj.params.items():
        if surface_pair not in list(lj.params.keys()):
            lj.params[surface_pair] = surface_params
            lj.r_cut[surface_pair] = surface_lj.r_cut[surface_pair]

    # find new pairs and add them to the droplet LJ pairs
    r_cut = list(drop_lj.r_cut.values())[0]
    for drop_ptype in drop_ptypes:
        for surface_ptype in surface_ptypes:
            if (drop_ptype, surface_ptype) not in list(lj.params.keys()) and (
                surface_ptype,
                drop_ptype,
            ) not in list(lj.params.keys()):
                epsilon = np.sqrt(
                    drop_lj.params[(drop_ptype, drop_ptype)]["epsilon"]
                    * surface_lj.params[(surface_ptype, surface_ptype)][
                        "epsilon"
                    ]
                )
                if combining_rule == "geometric":
                    sigma = np.sqrt(
                        drop_lj.params[(drop_ptype, drop_ptype)]["sigma"]
                        * surface_lj.params[(surface_ptype, surface_ptype)][
                            "sigma"
                        ]
                    )
                else:
                    # combining_rule == 'lorentz'
                    sigma = 0.5 * (
                        drop_lj.params[(drop_ptype, drop_ptype)]["sigma"]
                        + surface_lj.params[(surface_ptype, surface_ptype)][
                            "sigma"
                        ]
                    )

                lj.params[(drop_ptype, surface_ptype)] = {
                    "sigma": sigma,
                    "epsilon": epsilon,
                }
                lj.r_cut[(drop_ptype, surface_ptype)] = r_cut
    return lj


def combine_bond_forces(drop_bond, surface_bond):
    """Combine the droplet and surface bond forces."""
    bond = copy.deepcopy(drop_bond)
    # add the surface bond parameters to the droplet bond parameters
    for k, v in surface_bond.params.items():
        if k not in bond.params.keys():
            bond.params[k] = v
    return bond


def combine_angle_forces(drop_angle, surface_angle):
    """Combine the droplet and surface angle forces."""
    angle = copy.deepcopy(drop_angle)
    # add the surface angle parameters to the droplet angle parameters
    for k, v in surface_angle.params.items():
        if k not in angle.params.keys():
            angle.params[k] = v
    return angle


def combine_dihedral_forces(drop_dihedral, surface_dihedral):
    """Combine the droplet and surface dihedral forces."""
    dihedral = copy.deepcopy(drop_dihedral)
    # add the surface dihedral parameters to the droplet dihedral parameters
    for k, v in surface_dihedral.params.items():
        if k not in dihedral.params.keys():
            dihedral.params[k] = v
    return dihedral
