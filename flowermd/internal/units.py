import unyt as u


class Units:
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

    # mass units
    g = u.g
    gram = u.g
    amu = u.amu
    kg = u.kg

    # Mol units
    mol = u.mol
    kcal_mol = u.kcal / u.mol
    kJ_mol = u.kJ / u.mol

    # time units
    s = u.s
    second = u.s
    ns = u.ns
    nanosecond = u.ns
    ps = u.ps
    picosecond = u.ps

    # temperature units
    K = u.K
    Kelvin = u.K
    C = u.degC
    Celsius = u.degC
    F = u.degF
    Fahrenheit = u.degF
