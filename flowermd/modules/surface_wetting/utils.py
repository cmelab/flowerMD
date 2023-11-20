"""Utility functions for the surface wetting module."""

import hoomd
import numpy as np


def combine_forces(drop_forces, surface_forces, drop_ptypes, surface_ptypes):
    """Combine the surface and droplet forces."""
    combined_force_list = []

    # retrieve individual force objects from the droplet and surface
    # force lists
    drop_forces_dict = _retrieve_all_forces(drop_forces)
    surface_forces_dict = _retrieve_all_forces(surface_forces)

    # combine LJ forces
    combined_force_list.append(
        _combine_lj_forces(
            drop_forces_dict["lj"],
            surface_forces_dict["lj"],
            drop_ptypes,
            surface_ptypes,
        )
    )

    # add Coulomb forces if exists
    if drop_forces_dict.get("coulomb"):
        combined_force_list.append(drop_forces_dict["coulomb"])
    elif surface_forces_dict.get("coulomb"):
        combined_force_list.append(surface_forces_dict["coulomb"])
    # combine bond forces
    combined_force_list.append(
        _merge_force_parameters(
            hoomd.md.bond.Harmonic,
            drop_forces_dict["bond"],
            surface_forces_dict["bond"],
        )
    )
    # combine angle forces
    combined_force_list.append(
        _merge_force_parameters(
            hoomd.md.angle.Harmonic,
            drop_forces_dict["angle"],
            surface_forces_dict["angle"],
        )
    )
    # combine dihedral forces
    combined_force_list.append(
        _merge_force_parameters(
            hoomd.md.dihedral.OPLS,
            drop_forces_dict["dihedral"],
            surface_forces_dict["dihedral"],
        )
    )
    return combined_force_list


def _retrieve_all_forces(hoomd_force_list):
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


def _combine_lj_forces(
    drop_lj, surface_lj, drop_ptypes, surface_ptypes, combining_rule="geometric"
):
    """Combine the droplet and surface LJ forces."""
    if combining_rule not in ["geometric", "lorentz"]:
        raise ValueError("combining_rule must be 'geometric' or 'lorentz'")

    lj = hoomd.md.pair.LJ(nlist=drop_lj.nlist)

    # add droplet LJ parameters
    for k, v in drop_lj.params.items():
        lj.params[k] = v
        lj.r_cut[k] = drop_lj.r_cut[k]

    # add the surface LJ parameters if they don't already exist
    for k, v in surface_lj.params.items():
        if k not in list(lj.params.keys()):
            lj.params[k] = v
            lj.r_cut[k] = surface_lj.r_cut[k]

    # add pair parameters for droplet-surface interactions
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


def _merge_force_parameters(new_force_type, drop_force_obj, surface_force_obj):
    """Merge the force parameters of the droplet and surface forces."""
    new_force_obj = new_force_type()
    # add droplet force parameters
    for k, v in drop_force_obj.params.items():
        new_force_obj.params[k] = v
    # add the surface force parameters if they don't already exist
    for k, v in surface_force_obj.params.items():
        if k not in list(new_force_obj.params.keys()):
            new_force_obj.params[k] = v
    return new_force_obj
