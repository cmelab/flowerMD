import sys
import time

import mbuild as mb
import numpy as np
import pytest
import unyt as u

from flowermd.base import Molecule, Polymer
from flowermd.internal.local_boltzmann import _plan
from flowermd.internal.placement import (
    chain_junctions,
    random_chain_conformation,
    repeat_units,
)
from flowermd.library import (
    OPLS_AA,
    AtomisticRandomWalk,
    EllipsoidChainRand,
    LJChain,
    LocalBoltzmannSampler,
    PolyEthylene,
    RandomWalk,
    SingleChainSystem,
    UniformTorsion,
    mbuildSystem,
)
from flowermd.tests import BaseTest


class TestSystems(BaseTest):
    def test_single_chain_buffer(self):
        chain = PolyEthylene(lengths=10, num_mols=1)
        system = SingleChainSystem(molecules=chain, buffer=1.05)
        chain2 = PolyEthylene(lengths=10, num_mols=1)
        system2 = SingleChainSystem(molecules=chain2, buffer=2.10)

        assert np.allclose(system.box.Lx * 2, system2.box.Lx, atol=1e-3)
        assert np.allclose(system.box.Ly * 2, system2.box.Ly, atol=1e-3)
        assert np.allclose(system.box.Lz * 2, system2.box.Lz, atol=1e-3)

    def test_single_chain_lengths(self):
        lj_chain = LJChain(lengths=10, num_mols=1)
        system = SingleChainSystem(molecules=lj_chain, buffer=1.05)
        chain = lj_chain._molecules[0]
        chain_length = np.linalg.norm(
            chain.children[-1].pos - chain.children[0].pos
        )
        assert np.allclose(chain_length * 1.05, system.box.Lx, atol=1e-3)

    def test_multiple_molecules(self):
        lj_chain = LJChain(lengths=10, num_mols=2)
        with pytest.raises(ValueError):
            SingleChainSystem(molecules=lj_chain, buffer=1.05)

    def test_ell_rand_walk(self):
        chains = EllipsoidChainRand(
            lengths=10, num_mols=10, lpar=0.5, bead_mass=1.0, density=0.85
        )
        system = mbuildSystem(molecules=chains)
        assert system.box
        assert system.n_particles == 400  # 10 x 10 x 4

    def test_RandomWalk(self):
        lj_chain = LJChain(
            lengths=[10],
            num_mols=[20],
            bead_sequence=["_A"],
            bead_mass={"_A": 1.0},
            bond_lengths={"_A-_A": 1.0},
            identify_connections=False,
        )
        bond_length = 1.0
        ref_length = 1.0 * u.Unit("nm")
        ref_mass = 1.0 * u.Unit("g/mol")
        ref_energy = 1.0 * u.Unit("kcal / mol")
        ref_values_dict = {
            "length": ref_length,
            "mass": ref_mass,
            "energy": ref_energy,
        }
        system = RandomWalk(
            molecules=lj_chain,
            buffer=0.2,
            bond_length=bond_length,
            density=1.0 * u.Unit("nm**-3"),
            base_units=ref_values_dict,
        )
        snapshot = system.hoomd_snapshot
        positions = np.asarray(snapshot.particles.position)
        box = np.array(snapshot.configuration.box[:3])
        bond_group = np.asarray(snapshot.bonds.group)

        r_ij = positions[bond_group[:, 0]] - positions[bond_group[:, 1]]
        r_ij -= box * np.round(r_ij / box)
        bond_lengths = np.linalg.norm(r_ij, axis=1)

        np.testing.assert_allclose(
            bond_lengths,
            bond_length,
            atol=1e-3,
            err_msg=(
                f"Bond length deviates from target {bond_length}. "
                f"min={bond_lengths.min():.6f}, max={bond_lengths.max():.6f}, "
                f"n_bad={np.sum(~np.isclose(bond_lengths, bond_length, atol=1e-3))}"
            ),
        )

    def test_RandomWalk_different_seeds(self):
        lj_chain1 = LJChain(lengths=10, num_mols=10)
        lj_chain2 = LJChain(lengths=10, num_mols=10)
        system1 = RandomWalk(
            molecules=lj_chain1,
            bond_length=1.0,
            density=1.1,
            seed=1234,
        )
        system2 = RandomWalk(
            molecules=lj_chain2,
            bond_length=1.0,
            density=1.1,
            seed=4567,
        )
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                system1.gmso_system.positions, system2.gmso_system.positions
            )


def _geometry(chain, xyz):
    """Bond lengths, bond-angle cosines and signed volumes at 4-fold atoms."""
    index = {p: i for i, p in enumerate(chain.particles())}
    bonds = np.array([(index[a], index[b]) for a, b in chain.bonds()])
    nbr = [[] for _ in xyz]
    for a, b in bonds:
        nbr[a].append(b)
        nbr[b].append(a)
    unit = [(xyz[n] - xyz[j]) for j, n in enumerate(nbr)]
    unit = [v / np.linalg.norm(v, axis=1)[:, None] for v in unit]
    cosines = np.concatenate(
        [(v @ v.T)[np.triu_indices(len(v), 1)] for v in unit]
    )
    volumes = [np.linalg.det(v[:3]) for v in unit if len(v) == 4]
    lengths = np.linalg.norm(xyz[bonds[:, 0]] - xyz[bonds[:, 1]], axis=1)
    return lengths, cosines, np.array(volumes)


def _ladder():
    """Three two-atom units: a two-bond (ring) junction, then a single bond."""
    chain = mb.Compound()
    units = []
    for x in range(3):
        unit = mb.Compound()
        unit.add(mb.Particle(name="C", pos=[0.15 * x, 0.0, 0.0]))
        unit.add(mb.Particle(name="C", pos=[0.15 * x, 0.15, 0.0]))
        unit.add_bond(list(unit.particles()))
        chain.add(unit)
        units.append(list(unit.particles()))
    chain.add_bond([units[0][0], units[1][0]])
    chain.add_bond([units[0][1], units[1][1]])
    chain.add_bond([units[1][0], units[2][0]])
    return chain


class TestAtomisticRandomWalk:
    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        tmpdir.chdir()

    def test_keeps_geometry_and_handedness(self):
        chain = PolyEthylene(lengths=8, num_mols=1).molecules[0]
        new = random_chain_conformation(
            chain, np.zeros(3), np.random.default_rng(0)
        )
        for before, after in zip(
            _geometry(chain, chain.xyz), _geometry(chain, new)
        ):
            np.testing.assert_allclose(after, before, atol=1e-6)
        assert not np.allclose(new, chain.xyz)

    def test_repeat_units_and_junctions(self):
        chain = PolyEthylene(lengths=5, num_mols=1).molecules[0]
        units = repeat_units(chain)
        assert len(units) == 5
        assert sorted(sum(units, [])) == list(range(chain.n_particles))
        index = {p: i for i, p in enumerate(chain.particles())}
        bonds = [(index[a], index[b]) for a, b in chain.bonds()]
        junctions = chain_junctions(units, bonds, chain.n_particles)
        assert len(junctions) == 4 and all(j.torsion for j in junctions)

    def test_ladder_junction_keeps_built_pose(self):
        chain = _ladder()
        junctions = chain_junctions(
            [[0, 1], [2, 3], [4, 5]],
            [(0, 1), (2, 3), (4, 5), (0, 2), (1, 3), (2, 4)],
            6,
        )
        assert [j.torsion for j in junctions] == [False, True]
        new = random_chain_conformation(
            chain, np.zeros(3), np.random.default_rng(1)
        )
        built = np.linalg.norm(
            chain.xyz[:4, None] - chain.xyz[None, :4], axis=2
        )
        placed = np.linalg.norm(new[:4, None] - new[None, :4], axis=2)
        np.testing.assert_allclose(placed, built, atol=1e-9)

    def test_torsion_sampler(self):
        calls = []

        class NoTurn(UniformTorsion):
            def choose(self, index, placed, propose, rng):
                calls.append(propose(np.zeros(1)))
                return 0.0

        chain = PolyEthylene(lengths=6, num_mols=1).molecules[0]
        new = random_chain_conformation(
            chain, np.zeros(3), np.random.default_rng(2), NoTurn()
        )
        assert len(calls) == 5
        distances = np.linalg.norm(chain.xyz[:, None] - chain.xyz, axis=2)
        np.testing.assert_allclose(
            np.linalg.norm(new[:, None] - new, axis=2), distances, atol=1e-6
        )

    def test_system(self):
        chains = PolyEthylene(lengths=6, num_mols=4)
        uniform = UniformTorsion()
        system = AtomisticRandomWalk(
            chains, density=0.85, seed=3, torsion_sampler=uniform
        )
        again = AtomisticRandomWalk(
            PolyEthylene(lengths=6, num_mols=4),
            density=0.85,
            seed=3,
            torsion_sampler=uniform,
        )
        np.testing.assert_allclose(system.system.xyz, again.system.xyz)
        assert system.n_particles == 4 * 38
        index = {p: i for i, p in enumerate(system.system.particles())}
        xyz = system.system.xyz
        lengths = [
            np.linalg.norm(xyz[index[a]] - xyz[index[b]])
            for a, b in system.system.bonds()
        ]
        assert max(lengths) < 0.16


class TestLocalBoltzmannSampler:
    @pytest.fixture(scope="class")
    def sampler(self):
        pytest.importorskip("openff.interchange")
        return LocalBoltzmannSampler()

    def test_rigid_units(self, sampler):
        # Every C-C bond with a carbon beyond each end turns, so PE5 has
        # 8 rigid units: the two ethyl ends and six CH2 groups.
        chain = PolyEthylene(lengths=5, num_mols=1).molecules[0]
        xyz, units = sampler.template(chain)
        assert len(units) == 8
        # In 2,2'-bithiophene only the bond between the rings turns.
        chain = Polymer(
            lengths=2,
            num_mols=1,
            smiles="c1ccsc1",
            bond_indices=[7, 8],
            bond_length=0.145,
            bond_orientation=[None, None],
        ).molecules[0]
        assert len(sampler.template(chain)[1]) == 2

    def test_keeps_template_geometry_and_handedness(self, sampler):
        ps = "CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C"
        chain = Molecule(num_mols=1, smiles=ps).molecules[0]
        xyz, units = sampler.template(chain)
        assert len(units) == 8  # 5 backbone pieces and 3 phenyls
        new = random_chain_conformation(
            chain, np.zeros(3), np.random.default_rng(0), sampler
        )
        for before, after in zip(_geometry(chain, xyz), _geometry(chain, new)):
            np.testing.assert_allclose(after, before, atol=1e-6)
        built = _geometry(chain, chain.xyz)
        assert np.all(np.sign(built[2]) == np.sign(_geometry(chain, new)[2]))
        # the template has the force field's bond lengths
        bonds = sampler._topology(chain)[1]["bonds"]
        r = [
            np.linalg.norm(xyz[i] - xyz[j]) - r0
            for (i, j), (_, r0) in bonds.items()
        ]
        assert np.max(np.abs(r)) < 2e-3

    def test_statistics(self, sampler):
        chain = PolyEthylene(lengths=10, num_mols=1).molecules[0]
        carbons = [i for i, p in enumerate(chain.particles()) if p.name == "C"]
        rng = np.random.default_rng(1)
        ree2 = [
            np.sum(np.diff(x[carbons][[0, -1]], axis=0) ** 2)
            for x in (
                random_chain_conformation(chain, np.zeros(3), rng, sampler)
                for _ in range(100)
            )
        ]
        # C_n of a C20 chain at 413 K is about 5 (rotational isomeric state
        # theory); a uniform walk over every torsion would give about 2.
        assert 4.3 < np.mean(ree2) / (19 * 0.154**2) < 6.0

    def test_restart_cap(self):
        pytest.importorskip("openff.interchange")
        sampler = LocalBoltzmannSampler(r_min=0.6, max_restarts=3)
        chain = PolyEthylene(lengths=10, num_mols=1).molecules[0]
        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="4 attempts"):
            random_chain_conformation(
                chain, np.zeros(3), np.random.default_rng(0), sampler
            )
        assert time.perf_counter() - start < 60


class TestLocalBoltzmannSamplerOPLS:
    @pytest.fixture
    def no_openff(self, monkeypatch):
        """Make every openff import fail."""
        for name in [m for m in sys.modules if m.startswith("openff")]:
            monkeypatch.setitem(sys.modules, name, None)
        for name in ("openff", "openff.toolkit", "openff.interchange"):
            monkeypatch.setitem(sys.modules, name, None)

    def test_without_openff(self, no_openff):
        with pytest.raises(ImportError, match="OPLS_AA"):
            LocalBoltzmannSampler()
        sampler = LocalBoltzmannSampler(force_field=OPLS_AA())
        assert LocalBoltzmannSampler(force_field="oplsaa").force_field.name
        ps = "CC(c1ccccc1)CC(c1ccccc1)CC(c1ccccc1)C"
        chain = Molecule(num_mols=1, smiles=ps).molecules[0]
        xyz, units = sampler.template(chain)
        assert len(units) == 8
        new = random_chain_conformation(
            chain, np.zeros(3), np.random.default_rng(0), sampler
        )
        for before, after in zip(_geometry(chain, xyz), _geometry(chain, new)):
            np.testing.assert_allclose(after, before, atol=1e-6)
        built = _geometry(chain, chain.xyz)
        assert np.all(np.sign(built[2]) == np.sign(_geometry(chain, new)[2]))
        chain = PolyEthylene(lengths=5, num_mols=1).molecules[0]
        x = random_chain_conformation(
            chain, np.zeros(3), np.random.default_rng(1), sampler
        )
        assert np.all(np.isfinite(x))

    def test_torsion_and_lj_terms(self):
        sampler = LocalBoltzmannSampler(force_field=OPLS_AA())
        chain = PolyEthylene(lengths=3, num_mols=1).molecules[0]  # hexane
        terms = sampler._topology(chain)[1]
        carbon = [p.element.symbol == "C" for p in chain.particles()]
        # OPLS-AA Ryckaert-Bellemans coefficients (kJ/mol), psi = phi - pi
        rb = {True: (2.9288, -1.4644, 0.2092, -1.6736)}  # CT-CT-CT-CT
        rb[False] = (0.6276, 1.8828, 0.0, -2.5104)  # with an HC end
        phi = np.linspace(-np.pi, np.pi, 37)
        for quad, values in terms["dihedrals"].items():
            mine = sum(k * (1 + np.cos(n * phi - p)) for k, n, p in values)
            c = rb[all(carbon[i] for i in quad)]
            ref = sum(c[m] * np.cos(phi - np.pi) ** m for m in range(4))
            np.testing.assert_allclose(mine - mine[0], ref - ref[0], atol=1e-9)
        assert terms["scale14"] == 0.5 and terms["combining"] == "geometric"
        sigma = {True: 0.35, False: 0.25}
        eps = {True: 0.276144, False: 0.125520}
        junctions = chain_junctions(
            terms["units"], list(terms["bonds"]), len(carbon)
        )
        for j, step in zip(junctions, _plan(terms, junctions, 4)):
            rows, cols = j.moved[step.pr], step.pc
            si = np.array([sigma[carbon[i]] for i in rows])
            sj = np.array([sigma[carbon[i]] for i in cols])
            np.testing.assert_allclose(step.sigma2, si * sj)
            ei = np.array([eps[carbon[i]] for i in rows])
            ej = np.array([eps[carbon[i]] for i in cols])
            assert np.all(
                np.isclose(step.eps, np.sqrt(ei * ej))
                | np.isclose(step.eps, 0.5 * np.sqrt(ei * ej))
            )
