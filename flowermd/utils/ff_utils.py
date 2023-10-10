import os

import forcefield_utilities as ffutils
import hoomd
from gmso.parameterization import apply

from flowermd.assets import FF_DIR

from .exceptions import (
    MissingAnglePotentialError,
    MissingBondPotentialError,
    MissingCoulombPotentialError,
    MissingDihedralPotentialError,
    MissingPairPotentialError,
)


def ff_xml_directory():
    ff_xml_directory = dict()
    for dirpath, dirnames, filenames in os.walk(FF_DIR):
        for file in filenames:
            if file.endswith(".xml"):
                ff_xml_directory[file.split(".xml")[0]] = os.path.join(
                    dirpath, file
                )
    return ff_xml_directory


def find_xml_ff(ff_source):
    xml_directory = ff_xml_directory()
    if os.path.isfile(ff_source):
        if not ff_source.endswith(".xml"):
            raise ValueError("ForceField file type must be XML.")
        ff_xml_path = ff_source
    elif not xml_directory.get(ff_source.split(".xml")[0]):
        raise ValueError(
            "{} forcefield is not supported. Supported XML forcefields "
            "are {}".format(ff_source, list(xml_directory.keys()))
        )
    else:
        ff_key = ff_source.split(".xml")[0]
        ff_xml_path = xml_directory.get(ff_key)
    return ff_xml_path


def xml_to_gmso_ff(ff_xml):
    ff_xml_path = find_xml_ff(ff_xml)
    gmso_ff = ffutils.FoyerFFs().load(ff_xml_path).to_gmso_ff()
    return gmso_ff


def apply_xml_ff(ff_xml_path, gmso_mol):
    gmso_ff = ffutils.FoyerFFs().load(ff_xml_path).to_gmso_ff()
    apply(top=gmso_mol, forcefields=gmso_ff, identify_connections=True)
    # TODO: Warning if any parameter is missing?
    return gmso_mol


def _include_hydrogen(connections, hydrogen_types):
    return any(p in hydrogen_types for p in connections)


def _validate_hoomd_ff(forcefields, topology_information, ignore_hydrogen=True):
    pair_forces = []
    bond_forces = []
    angle_forces = []
    dihedral_forces = []
    coulomb_forces = []

    for force in forcefields:
        if isinstance(force, hoomd.md.pair.Pair):
            pair_forces.append(force)
        elif isinstance(force, hoomd.md.bond.Bond):
            bond_forces.append(force)
        elif isinstance(force, hoomd.md.angle.Angle):
            angle_forces.append(force)
        elif isinstance(force, hoomd.md.dihedral.Dihedral):
            dihedral_forces.append(force)
        elif isinstance(force, hoomd.md.long_range.pppm.Coulomb):
            coulomb_forces.append(force)
    # check if all the required forcefields are present
    if topology_information["pair_types"] and not pair_forces:
        raise MissingPairPotentialError(
            connection=topology_information["pair_types"],
            potential_class=str(hoomd.md.pair.LJ),
        )
    if topology_information["bond_types"] and not bond_forces:
        raise MissingBondPotentialError(
            connection=topology_information["bond_types"],
            potential_class=str(hoomd.md.bond.Bond),
        )
    if topology_information["angle_types"] and not angle_forces:
        raise MissingAnglePotentialError(
            connection=topology_information["angle_types"],
            potential_class=str(hoomd.md.angle.Angle),
        )
    if topology_information["dihedral_types"] and not dihedral_forces:
        raise MissingDihedralPotentialError(
            connection=topology_information["dihedral_types"],
            potential_class=str(hoomd.md.dihedral.Dihedral),
        )
    if any(topology_information["particle_charge"]) and not coulomb_forces:
        raise MissingCoulombPotentialError(
            potential_class=str(hoomd.md.long_range.pppm.Coulomb)
        )

    for f in pair_forces:
        params = list(map(list, f.params.keys()))
        for pair in topology_information["pair_types"]:
            pair = list(pair)
            if ignore_hydrogen and _include_hydrogen(
                pair, topology_information["hydrogen_types"]
            ):
                # ignore pair interactions that include hydrogen atoms
                continue
            if not (pair in params or pair[::-1] in params):
                raise MissingPairPotentialError(
                    connection=tuple(pair), potential_class=type(f)
                )

    for f in bond_forces:
        params = list(f.params.keys())
        for bond in topology_information["bond_types"]:
            bond_dir1 = "-".join(bond)
            bond_dir2 = "-".join(bond[::-1])
            if ignore_hydrogen and _include_hydrogen(
                bond, topology_information["hydrogen_types"]
            ):
                # ignore bonds that include hydrogen atoms
                continue
            if not (bond_dir1 in params or bond_dir2 in params):
                raise MissingBondPotentialError(
                    connection=bond_dir1, potential_class=type(f)
                )

    for f in angle_forces:
        params = list(f.params.keys())
        for angle in topology_information["angle_types"]:
            angle_dir1 = "-".join(angle)
            angle_dir2 = "-".join(angle[::-1])
            if ignore_hydrogen and _include_hydrogen(
                angle, topology_information["hydrogen_types"]
            ):
                # ignore angles that include hydrogen atoms
                continue
            if not (angle_dir1 in params or angle_dir2 in params):
                raise MissingAnglePotentialError(
                    connection=angle_dir1, potential_class=type(f)
                )

    for f in dihedral_forces:
        params = list(f.params.keys())
        for dihedral in topology_information["dihedral_types"]:
            dihedral_dir1 = "-".join(dihedral)
            dihedral_dir2 = "-".join(dihedral[::-1])
            if ignore_hydrogen and _include_hydrogen(
                dihedral, topology_information["hydrogen_types"]
            ):
                # ignore dihedrals that include hydrogen atoms
                continue
            if not (dihedral_dir1 in params or dihedral_dir2 in params):
                raise MissingDihedralPotentialError(
                    connection=dihedral_dir1, potential_class=type(f)
                )
