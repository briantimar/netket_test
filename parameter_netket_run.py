#This code is meant to run a netket code a few times
#Reconstruct the wave function for each run
#Calculate the expectation value of some observables
#Compare with the Netket output file.log for each oberservable and for each run

from wavefunction_utils import wave_function_normalized, constr_op, expect_val
from wavefunction_utils import construct_basis, fourier_coefficient, load_parameters
from wavefunction_utils import entanglement_entropy
from ising1d_test import sigmaz, sigmax, n


import numpy as np
import json
import matplotlib.pyplot as plt

def make_param_file_name(i):
#    return 'ising1d_test_{0}.json'.format(i)
     return 'rydberg1d_test_{0}.json'.format(i)
def make_output_file_name(i):
#    return 'ising1d_test_output_{0}'.format(i)
     return 'rydberg1d_test_output_{0}'.format(i)
#Runs Netket on the files
def run_netket(Nsamples):
    import subprocess
    for i in range(Nsamples):
        retcode=subprocess.call(['netket', make_param_file_name(i)] )
#Loads the wave function in each file.wf 
def load_wf(i):
    basis_dic, basis = construct_basis(L)
    a, b, W = load_parameters(make_output_file_name(i)+'.wf')
    wf, wf_dic = wave_function_normalized(L, a, b, W)
    return wf
#Gets the data for an observable from the last iteration
def get_netket_obs(i, opname):
    """ returns mean val and std dev from last iteration"""
    of = make_output_file_name(i)+".log"
    with open(of) as of:
        data=json.load(of)["Output"]
        return data[-1][opname]["Mean"], data[-1][opname]["Sigma"]
#Makes lists of matrices for certain operators
def zz(L, i,j):
    oplist_base = [np.identity(2) for _ in range(L)]
    oplist_base[i], oplist_base[j] = sigmaz(), sigmaz()
    op = constr_op(oplist_base)
    return op
def z(L, i):
    oplist_base = [np.identity(2) for _ in range(L)]
    oplist_base[i] = sigmaz()
    op = constr_op(oplist_base)
    return op
def xx(L, i,j):
    oplist_base = [np.identity(2) for _ in range(L)]
    oplist_base[i], oplist_base[j] = sigmax(), sigmax()
    op = constr_op(oplist_base)
    return op
def x(L, i):
    oplist_base = [np.identity(2) for _ in range(L)]
    oplist_base[i] = sigmax()
    op = constr_op(oplist_base)
    return op
def nn(L, i,j):
    oplist_base = [np.identity(2) for _ in range(L)]
    oplist_base[i], oplist_base[j] = n(), n()
    op = constr_op(oplist_base)
    return op
def n_op(L, i):
    oplist_base = [np.identity(2) for _ in range(L)]
    oplist_base[i] = n()
    op = constr_op(oplist_base)
    return op
#makes the observables to compare 
def process_netket_output(states, op, name_in_netket):
    op_from_wf = [expect_val(op, psi) for psi in states]
    
    op_netket = [get_netket_obs(i, name_in_netket) for i in range(Nsamples)]
    op_mean_from_netket = [t[0] for t in op_netket]
    op_sigma_from_netket = [t[1] for t in op_netket]
    
    return op_mean_from_netket, op_sigma_from_netket     
def entanglement_entropy_list(states):
    enentropy_list = []
    for psi in states:
        enentropy_list.appen(entanglement_entropy(psi))
    return enentropy_list
###############################################################################
#Let us try this on an 1D Ising transverse field Hamiltonian
#L=2
#J=-1
#hmin = 0
#hmax = 10
#Nsamples = 100
#hsamples = np.linspace(hmin, hmax, Nsamples)
# This function generates the parameter files to be run by Netket
#def gen_param_files():
#    from ising1d_test import make_tfi_pars
#    for i in range(Nsamples):
#        pfile = make_param_file_name(i)
#        outfile = make_output_file_name(i)
#        pars = make_tfi_pars(hsamples[i], J, L, outfile)
#        with open(pfile, 'w') as pf:
#            json.dump(pars, pf)

#gen_param_files()
#run_netket(Nsamples)
###############################################################################
#RYDBERG RYDBERG RYDBERG RYDBERG RYDBERG RYDBERG RYDBERG RYDBERG RYDBERG RYDBERG
###############################################################################
#Let us try this on a 1D Rydberg atoms chain
L = 2
Delta = 0.1
V = 1
Omegamin = 0
Omegamax = 0.5
Nsamples = 50
Omegasamples = np.linspace(Omegamin, Omegamax, Nsamples)

def gen_param_files():
    from Rydberg_spinbasis_test import make_Ryd_pars
    for i in range(Nsamples):
        pfile = make_param_file_name(i)
        outfile = make_output_file_name(i)
        pars = make_Ryd_pars(Omegasamples[i], V, Delta, L, outfile)
        with open(pfile, 'w') as pf:
            json.dump(pars, pf)
#gen_param_files()
#run_netket(Nsamples)
###############################################################################
states = [load_wf(i) for i in range(Nsamples)]
zzwf, zz_nk, zz_sigma_nk = process_netket_output(states, zz(L, 0,1), 'zz01')
z1wf, z1_nk, z1_sigma_nk = process_netket_output(states, z(L, 0), 'z1')
z2wf, z2_nk, z2_sigma_nk = process_netket_output(states, z(L,1), 'z2')
enentropy = entanglement_entropy_list(states) 

plt.plot(Omegasamples, zzwf, label='from wf')
plt.plot(Omegasamples, zz_nk, 'x', label='netket obs')
plt.legend()
plt.ylabel('<z1z2>', fontsize=18)
plt.xlabel('Omega', fontsize=16)
plt.savefig('spinprodcuts_minevsNetkets_Ryd_10moresamples.png')
plt.show()

plt.plot(Omegasamples, enentropy)
plt.ylabel('Entanglement Entropy', fontsize=18)
plt.xlabel('Omega', fontsize=16)
plt.savefig('enentropy_Ryd_10moresamples.png')
plt.show()

plt.plot(Omegasamples, z1wf, label='from wf')
plt.plot(Omegasamples, z1_nk, 'x', label='netket obs')
plt.legend()
plt.ylabel('<z1>', fontsize=18)
plt.xlabel('Omega', fontsize=16)
plt.savefig('spin1_minevsNetkets_Ryd_10moresamples.png')
plt.show()

plt.plot(Omegasamples, z2wf, label='from wf')
plt.plot(Omegasamples, z2_nk, 'x', label='netket obs')
plt.legend()
plt.ylabel('<z2>', fontsize=18)
plt.xlabel('Omega', fontsize=16)
plt.savefig('spin2_minevsNetkets_Ryd_10moresamples.png')
plt.show()

plt.plot(Omegasamples, z1wf, label = 'z1 from wf')
plt.plot(Omegasamples, z2wf, label = 'z2 from wf')
plt.plot(Omegasamples, z1_nk, 'x', label = 'z1 from netket')
plt.plot(Omegasamples, z2_nk, '+', label = 'z2 from netket')
plt.ylabel('<z1> and <z2>', fontsize=18)
plt.xlabel('Omega', fontsize=16)
plt.legend()
plt.savefig('spin1_spin2_minevsNetkets_Ryd_10moresamples.png')
plt.show()

###############################################################################
