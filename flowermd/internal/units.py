"""FlowerMD Unit class."""

import unyt as u


class Units:
    """FlowerMD Unit class.

    Example usage:
    --------------

    ::
        length = 1.0 * flowermd.internal.Units.angstrom
        energy = 1.0 * flowermd.internal.Units.kcal_mol
        mass = 1.0 * flowermd.internal.Units.amu
        time = 1.0 * flowermd.internal.Units.ps
        temperature = 1.0 * flowermd.internal.Units.K


    """

    # length units
    m = u.m
    meter = u.m
    cm = u.cm
    centimeter = u.cm
    nm = u.nm
    nanometer = u.nm
    pm = u.pm
    picometer = u.pm
    ang = u.angstrom
    angstrom = u.angstrom

    # energy units
    J = u.J
    Joule = u.J
    kJ = u.kJ
    cal = u.cal
    kcal = u.kcal
    kcal_mol = u.kcal / u.mol
    kJ_mol = u.kJ / u.mol

    # mass units
    g = u.g
    gram = u.g
    amu = u.amu
    kg = u.kg

    mol = u.mol

    # time units
    s = u.s
    second = u.s
    ns = u.ns
    nanosecond = u.ns
    ps = u.ps
    picosecond = u.ps
    fs = u.fs
    femtosecond = u.fs

    # temperature units
    K = u.K
    Kelvin = u.K
    C = u.degC
    Celsius = u.degC
    F = u.degF
    Fahrenheit = u.degF

    # mass density units
    g_cm3 = u.g / u.cm**3
    kg_m3 = u.kg / u.m**3
    amu_A3 = u.amu / u.angstrom**3
    # number density units
    n_m3 = u.Unit("m**-3")
    n_cm3 = u.Unit("cm**-3")
    n_A3 = u.Unit("angstrom**-3")
    n_nm3 = u.Unit("nm**-3")

    # pressure units
    atm = u.atm
    bar = u.bar
    Pa = u.Pa
    kPa = u.kPa
    MPa = u.MPa
    GPa = u.GPa
    psi = u.psi
