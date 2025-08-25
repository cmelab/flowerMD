import numpy as np
import pytest

from flowermd.library import LJChain, PolyEthylene, SingleChainSystem


class TestSingleChainSystem:
    def test_buffer(self):
        chain = PolyEthylene(lengths=10, num_mols=1)
        system = SingleChainSystem(molecules=chain, buffer=1.05)
        chain2 = PolyEthylene(lengths=10, num_mols=1)
        system2 = SingleChainSystem(molecules=chain2, buffer=2.10)

        assert np.allclose(system.box.Lx * 2, system2.box.Lx, atol=1e-3)
        assert np.allclose(system.box.Ly * 2, system2.box.Ly, atol=1e-3)
        assert np.allclose(system.box.Lz * 2, system2.box.Lz, atol=1e-3)

    def test_lengths(self):
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
