#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:40:55 2018

@author: brian
"""
import numpy as np

def make_single_site_coupling(h, O, L):
    """ Adds a uniform term h sum_i O_i to the hamiltonian, where O is a local op 
    Returns: site list, op list"""
    sites = [[i] for i in range(L)]
    ops = [(h * O.copy()).tolist() for _ in range(L)]
    return sites, ops

def make_nn_coupling(J, O1, O2, L, bc='periodic'):
    """ make a uniform nearest-neighbor coupling list assuming periodic bcs"""
    sites = [(i, (i+1)%L) for i in range(L)]
    ops = [ (J*np.kron(O1, O2)).tolist() for _ in range(L)]
    return sites,ops


def sigmax():
    return np.array([[0, 1],[1,0]])
def sigmaz():
    return np.array([[1,0], [0, -1]])

def make_tfi_hamiltonian(g, J, L):
    """ returns site and op lists"""
    sites, ops = make_single_site_coupling(-g, sigmax(), L)
    sites2, ops2 = make_nn_coupling(-J, sigmaz(), sigmaz(), L)
    sites += sites2
    ops += ops2
    return sites, ops

def pbc_chain_dict(L):
    return dict(Name='Hypercube',L=L,Dimension=1,Pbc=True)

def get_loc_obs_dict(i, O, name=''):
    sites =[ [i]]
    ops = [O.tolist()]
    return dict(ActingOn=sites,Operators=ops,Name=name+str(i))

def get_2site_obs_dict(i, j,O, name=''):
    sites = [[i, j]]
    ops = [O.tolist()]
    return dict(ActingOn=sites, Operators=ops,Name=name+str(i)+str(j))

def get_zz():
    return np.kron(sigmaz(), sigmaz())


def make_tfi_pdict(g, J, L, outfile = 'netket_output'):
    """ param dict for netket to construct ising hamiltonian"""
    sites, ops = make_tfi_hamiltonian(g,J,L)
    params=dict()
    params['Graph'] = pbc_chain_dict(L)
    params['Hilbert'] = dict(QuantumNumbers=[-1,1],Size=L)
    params['Hamiltonian'] = dict(ActingOn=sites, Operators=ops)
       
    # request a list of all nearest neighbor correlators
    params['Observables'] = [get_2site_obs_dict(i, (i+1)%L, get_zz(), name='zz') for i in range(L)]
    
    #rbm details
    params['Machine'] = dict(Name='RbmSpin',Alpha=2.0)
    params['Sampler'] = dict(Name='MetropolisLocal')
    
    params['Learning']={
        'Method'         : 'Sr',
        'Nsamples'       : 1.0e3,
        'NiterOpt'       : 500,
        'Diagshift'      : 0.1,
        'UseIterative'   : False,
        'OutputFile'     : outfile,
        'StepperType'    : 'Sgd',
        'LearningRate'   : 0.1,
    }

    return params







