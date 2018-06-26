#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:46:07 2018

@author: btimar
"""

import numpy as np
import json

from netket_constructor import make_tfi_hamiltonian, pbc_chain_dict, get_2site_obs_dict, get_zz

params = dict()

#physics related
L=2
g=1
J=0
sites, ops = make_tfi_hamiltonian(g,J,L)
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
    'OutputFile'     : "isingtest",
    'StepperType'    : 'Sgd',
    'LearningRate'   : 0.1,
}

outfile = "isingtest_params.json"
with open(outfile, 'w') as of:
    json.dump(params, of)







