import os

import gmso
import hoomd
import numpy as np
import pytest

from flowermd import SingleChainSystem
from flowermd.internal.exceptions import ForceFieldError, ReferenceUnitError
from flowermd.library import (
    OPLS_AA,
    LJChain,
)
from flowermd.tests import BaseTest


Class TestSystems(BaseTest):
'''
#test long chain,short chain,AA
	def test_single_chain(self, polyethylene):
        	polyethylene_short = polyethylene(lengths=3, num_mols=1)
        	short_system = SingleChainSystem(
            		molecules=[polyethylene]
		)
        	short_system.apply_forcefield(
            		r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        	)

        
		polyethylene_long = polyethylene(lengths=150, num_mols=1)
        	long_system = SingleChainSystem(
            		molecules=[polyethylene]
		)
        	long_system.apply_forcefield(
            		r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        	)	
		
		#add assert for success in system being built
             
#test CG chain
	def test_single_chain_cg(self, polyethylene):
        	cg_chain = LJChain(lengths=15, num_mols=1)
        	system = SingleChainSystem(
            		molecules=[cg_chain]
		)
        	system.apply_forcefield(
            		r_cut=2.5, force_field=[OPLS_AA()], auto_scale=True
        	)	

        	assert system.n_mol_types == 1
        	assert len(system.all_molecules) == len(polyethylene.molecules)
        	assert len(system.hoomd_forcefield) > 0
        	assert system.n_particles == system.hoomd_snapshot.particles.N
       
#test buffer <= 1.0 breaks
	def test_single_chain_buffer(self):
        	cg_chain = LJChain(lengths=15, num_mols=1)
        	system = SingleChainSystem(
            		molecules=[cg_chain], buffer=0.8
		)
		#add assert for break
 
#test num_mols > 1 breaks
	def test_single_chain_mols(self):
        	cg_chain = LJChain(lengths=15, num_mols=3)
        	system = SingleChainSystem(
            		molecules=[cg_chain]
		)
		#add assert for break
'''
