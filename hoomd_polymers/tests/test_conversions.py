from hoomd_polymers.utils import base_units, unit_conversions

def test_kelvin_from_reduced():
    kT = 4
    ref_energy = .2104 # ref_energy for PEEK/PEKK kcal/mol
    kelvin = unit_conversions.kelvin_from_reduced(kT, ref_energy)
    assert kelvin == 424

def test_reduced_from_kT():
    kelvin = 424
    ref_energy = .2104
    kT = unit_conversions.reduce_from_kelvin(kelvin, ref_energy)
    assert kT == 4

def test_convert_to_real_time():
    ref_energy = .2104
    ref_mass = 15.99943 # ref_mass for PEEK/PEKK amu
    ref_distance = 3.3996695084235347 #ref_dist for PEEK/PEKK ang
    dt = 0.001
    real_time = unit_conversions.convert_to_real_time(
                    dt,
                    ref_energy,
                    ref_distance,
                    ref_mass
    )                               
    assert real_time == 1.449

def test_base_units():# Test the conversion factors used in unit_conversions.py
    assert base_units.base_units()["kcal_to_j"] == 4184
    assert base_units.base_units()["amu_to_kg"] == 1.6605e-27
    assert base_units.base_units()["ang_to_m"] == 1e-10
    assert base_units.base_units()["boltzmann"] == 1.38064852e-23
    assert base_units.base_units()["avogadro"] == 6.022140857e23
