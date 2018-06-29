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

def distPBC(i,j,L):
    return min(abs(i-j), abs(i+L-j))

def make_r6_hamiltonian(Delta, Omega, V,L, ktrunc=1):
    sites, ops = make_single_site_coupling(-Omega/2, sigmax(), L)
    Vbar=0
    for i in range(L):
        for j in range(i+1, L):
            d= distPBC(i,j,L)
            if d <=ktrunc:
                sites += [(i,j)]
                ops += [((V/(4* d**6)) * np.kron(sigmaz(), sigmaz())).tolist()]
                
                if i==0:
                    Vbar += V/(2 *(d**6))
    zcoup = -.5 * (Delta - Vbar)
    sitesz, opsz = make_single_site_coupling(zcoup, sigmaz(), L)
    sites += sitesz
    ops += opsz
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

def constr_learning_dict(Method='Sr',Nsamples=1E3,NiterOpt=500,Diagshift=.1,UseIterative=False, OutputFile='<your_filename_here>.json',StepperType='Sgd',LearningRate=.1):
    return dict(Method=Method, Nsamples=Nsamples, NiterOpt=NiterOpt,Diagshift=Diagshift,UseIterative=UseIterative,OutputFile=OutputFile,StepperType=StepperType,LearningRate=LearningRate)

def make_tfi_pdict(g, J, L, obslist=[], alpha=1.0 ,learning_dict= constr_learning_dict()):
    """ param dict for netket to construct ising hamiltonian.
        g: the transverse field strength
        J: the strength of the ZZ coupling
        L: the length of the chain (PBC)
        
        observables_list: a list of dictionaries which specify observables. A single dict can also be used.
        learning_dict: specifies learning method, rate, and output file
        
        Returns: params dict which completely specifies a netket run"""
        
    sites, ops = make_tfi_hamiltonian(g,J,L)
    return make_pdict(L, sites, ops, obslist = obslist, alpha=alpha,learning_dict=learning_dict)

def make_r6_pdict(Delta, Omega, V, ktrunc, L, obslist=[],alpha=1.0,learning_dict=constr_learning_dict()):
    sites,ops = make_r6_hamiltonian(Delta,Omega,V,L,ktrunc=ktrunc)
    return make_pdict(L,sites,ops,obslist=obslist,alpha=alpha,learning_dict=learning_dict)

def make_pdict(L, sites, ops, obslist=[], alpha=1.0 ,learning_dict= constr_learning_dict()):
    """ returns pdict for PBC chain L"""
    params=dict()
    params['Graph'] = pbc_chain_dict(L)
    params['Hilbert'] = dict(QuantumNumbers=[-1,1],Size=L)
    params['Hamiltonian'] = dict(ActingOn=sites, Operators=ops)
       
    if len(obslist)>0:
        params['Observables'] = obslist
    
    #rbm details
    params['Machine'] = dict(Name='RbmSpin',Alpha=alpha)
    params['Sampler'] = dict(Name='MetropolisLocal')
    
    params['Learning']=learning_dict
    return params



