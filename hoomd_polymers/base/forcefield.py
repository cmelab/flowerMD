import os

import forcefield_utilities as ffutils
from gmso.parameterization import apply

from hoomd_polymers.base.base_types import FF_Types
# from hoomd_polymers.library import FF_DIR


def ff_xml_directory():
    ff_xml_directory = dict()
    for (dirpath, dirnames, filenames) in os.walk(FF_DIR):
        for file in filenames:
            if file.endswith('.xml'):
                ff_xml_directory[file.split('.xml')[0]] = os.path.join(dirpath, filenames)
    return ff_xml_directory


def _find_xml_ff(ff_source):
    xml_directory = ff_xml_directory()
    if os.path.isfile(ff_source):
        if not ff_source.endswith(".xml"):
            raise ValueError("ForceField file type must be XML.")
        ff_xml_path = ff_source
        ff_type = FF_Types.CUSTOM
    elif not xml_directory.get(ff_source.split('.xml')[0]):
        raise ValueError("{} forcefield is not supported. Supported XML forcefields are {}".
                         format(ff_source, list(xml_directory.keys())))
    else:
        ff_key = ff_source.split('.xml')[0]
        ff_xml_path = xml_directory.get(ff_key)
        ff_type = getattr(FF_Types, ff_key)
    return ff_xml_path, ff_type


def _apply_xml_ff(ff_xml_path, gmso_mol):
    gmso_ff = ffutils.FoyerFFs().load(ff_xml_path).to_gmso_ff()
    apply(top=gmso_mol,
          forcefields=gmso_ff,
          identify_connections=True)
    #TODO: Warning if any parameter is missing?
    return gmso_mol


def _validate_hoomd_ff(ff_list, gmso_mol):
    #TODO: Check if a force exsits for every bonded and non-bonded interaction types.
    return True
