#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:40:18 2018

@author: brian
"""
from netket_utils import get_weights, normalize, constr_op, matrix_el, gen_basis_strings, eval_wf
from netket_constructor import sigmaz, sigmax

import numpy as np
import json
import matplotlib.pyplot as plt
#ising parameters
L = 2
J=1
gmin=0.0
gmax = 10.0
#how many different hamiltonians you want to try
Nsamp = 100
gvals = np.linspace(gmin, gmax, Nsamp)

def make_param_file_name(i):
    return "wavefunction_test/ising_wftest_{0}.json".format(i)
def make_output_file_name(i):
    return "wavefunction_test/ising_wftest_output_{0}".format(i)

# generate the parameter files

def gen_param_files():
    from netket_constructor import make_tfi_pdict
    for i in range(Nsamp):
        pfile = make_param_file_name(i)
        outfile = make_output_file_name(i)
        params = make_tfi_pdict(gvals[i], J, L, outfile=outfile)
        with open(pfile, 'w') as pf:
            json.dump(params, pf)

#gen_param_files()

# run netket on the param files

def run_netket():
    from subprocess import call
    for i in range(Nsamp):
        retcode=call(["mpirun", "-n", "5", "netket", make_param_file_name(i) ])

# load the wavefunctions and the observable outputs

def get_wf(i, qnums = [-1, 1]):
    """ Return the output state of sample i.
    Make sure to use qnums that agree with what you put into the hamiltonian constructor"""
    from netket_utils import get_weights
    a,b,W = get_weights(make_output_file_name(i) + ".wf")
    s =gen_basis_strings(L, qnums=qnums)
    return normalize(eval_wf(a, b, W, s))

def get_netket_obs(i, opname):
    """ returns mean val and std dev from last iteration"""
    of = make_output_file_name(i)+".log"
    with open(of) as of:
        data=json.load(of)["Output"]
        return data[-1][opname]["Mean"], data[-1][opname]["Sigma"]
    

def process_netket_output():
    states = [get_wf(i) for i in range(Nsamp)]
    
    oplist_base = [np.identity(2) for _ in range(L)]
    z1z2 = oplist_base.copy()
    z1z2[0], z1z2[1] = sigmaz(), sigmaz()
    z1z2 = constr_op(z1z2)
    
    zzcorr_from_wf = [matrix_el(z1z2, psi) for psi in states]
        
    zz_netket = [get_netket_obs(i, "zz01") for i in range(Nsamp)]
    zz_mean_from_netket = [t[0] for t in zz_netket]
    zz_sigma_from_netket = [t[1] for t in zz_netket]
    
    return zzcorr_from_wf, zz_mean_from_netket, zz_sigma_from_netket, states

zzwf, zz_nk, zz_sigma_nk, states = process_netket_output()

plt.plot(gvals, zzwf, label='from wf')
plt.plot(gvals, zz_nk, 'x', label='netket obs')
plt.legend()

i=99
a,b,W = get_weights(make_output_file_name(i) + ".wf")
#    
#    
#
#
#
#
#
#
#a,b,W = get_weights("isingtest.wf")
#Nh=len(a)
#

#
#states=gen_basis_strings(Nh, qnums=[-1,1])
#psi = normalize(eval_wf(a, b, W, states))
#
#zzvals = matrix_el(z1z2, psi)
