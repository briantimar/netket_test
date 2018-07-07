#This code is meant to run a netket code a few times
#Reconstruct the wave function for each run
#Calculate the expectation value of some observables
#Compare with the Netket output file.log for each oberservable and for each run

from wavefunction_utils import wave_function_normalized, constr_op, expect_val
from wavefunction_utils import load_parameters
from wavefunction_utils import ent_entropy, load_parameters_symm, wave_function_normalized_symm
from ising1d_test import sigmaz, sigmax, n


import numpy as np
import json
import matplotlib.pyplot as plt

def make_param_file_name(i):
#    return 'ising1d_test_{0}.json'.format(i)
     return 'ising1d_test_{0}.json'.format(i)
def make_output_file_name(i):
#    return 'ising1d_test_output_{0}'.format(i)
     return 'ising1d_test_output_{0}'.format(i)
#Runs Netket on the files
def run_netket(Nsamples):
    import subprocess
    for i in range(Nsamples):
        retcode=subprocess.call([ 'netket', make_param_file_name(i)] )
#Loads the wave function in each file.wf 
def load_wf(i):
    a, b, W = load_parameters(make_output_file_name(i)+'.wf')
    wf, wf_dic = wave_function_normalized(L, a, b, W)
    return wf
#Loads symmetrized wave function
def load_wf_symm(i,L):
    a_symm, b_symm, W_symm = load_parameters_symm(make_output_file_name(i)+'.wf', L)
    wf, wf_dic = wave_function_normalized_symm(L, a_symm, b_symm, W_symm)
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
def get_energies():
    energies, energyvariance = np.zeros(Nsamples), np.zeros(Nsamples)
    for i in range(Nsamples):
        energies[i], energyvariance[i] = get_netket_obs(i, 'Energy')
    return energies, energyvariance
def process_netket_output(states, op, name_in_netket):
    op_from_wf = [expect_val(op, psi) for psi in states]
    
    op_netket = [get_netket_obs(i, name_in_netket) for i in range(Nsamples)]
    op_mean_from_netket = [t[0] for t in op_netket]
    op_sigma_from_netket = [t[1] for t in op_netket]
    
    return op_from_wf, op_mean_from_netket, op_sigma_from_netket  

def entanglement_entropy_list():
    enentropy_list = [0]
    for i in range(Nsamples-1):
        enentropy_list.append(ent_entropy(L, 10000, 'isng1d_test_output_{0}.wf'.format(i+1), 2000, 10))
    return enentropy_list
###############################################################################
#Let us try this on an 1D Ising transverse plus lonfitudinal field Hamiltonian
L=4
J=1
I = 0
hmin = 0
hmax = 1
Nsamples = 30
hsamples = np.linspace(hmin, hmax, Nsamples)

# This function generates the parameter files to be run by Netket
def gen_param_files():
    from ising1d_test import make_tfi_pars
    for i in range(Nsamples):
        pfile = make_param_file_name(i)
        outfile = make_output_file_name(i)
        pars = make_tfi_pars(hsamples[i], I, J, L, outfile)
        with open(pfile, 'w') as pf:
            json.dump(pars, pf)

gen_param_files()
run_netket(Nsamples)

#------------------------------------------------------------------------------
#                      PLOTTING                   
#------------------------------------------------------------------------------
#Loads the states and the energies
states = [load_wf_symm(i, L) for i in range(Nsamples)]
for i in range(Nsamples): states[i] = np.ndarray.flatten(states[i])
energies, energy_var = get_energies()
energies = energies.tolist()

#Initializes list to be prepared to plot and output the files 
xxwf, xx_nk, xx_nk_sigma = np.zeros(Nsamples), np.zeros(Nsamples), np.zeros(Nsamples)
xwf, x_nk, x_nk_sigma    = np.zeros(Nsamples), np.zeros(Nsamples), np.zeros(Nsamples)
zzwf, zz_nk, zz_nk_sigma = np.zeros(Nsamples), np.zeros(Nsamples), np.zeros(Nsamples)
zwf, z_nk, z_nk_sigma    = np.zeros(Nsamples), np.zeros(Nsamples), np.zeros(Nsamples)

for i in range(L):
    xxwf[i], xx_nk[i], xx_nk_sigma[i] = process_netket_output(states, xx(L, i,(i+1)%L), 'xx%d%d' %(i, (i+1)%L))
    xwf[i], x_nk[i], x_nk_sigma[i] = process_netket_output(states, x(L, i), 'x%d' %i)
    zzwf[i], zz_nk[i], zz_nk_sigma[i] = process_netket_output(states, zz(L, i,(i+1)%L), 'zz%d%d' %(i, (i+1)%L))
    zwf[i], z_nk[i], z_nk_sigma[i] = process_netket_output(states, z(L, i), 'z%d' %i)


#Loading the exact diagonalization information
data = json.load(open('exact_expectationvalues.json'))
exact_energy = data['GE']
exact_zz12 = data['z1z2']
exact_z1 = data['z1']
exact_z2 = data['z2']
exact_xx12 = data['x1x2']
exact_x1 = data['x1']
exact_x2 = data['x2']
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
ax = axs[0,0]

plt.plot(hsamples, energies, label = 'Netket energy', linewidth=4.0)
plt.plot(hsamples, exact_energy, label = 'ED energy')
plt.legend()
plt.ylabel('Energy')
plt.xlabel('h')
plt.savefig('Ising_energy.png')
plt.show()

plt.plot(hsamples, zzwf, label='from wf', linewidth=4.0)
plt.plot(hsamples, zz_nk, 'x', label='netket obs')
ax.error(hsamples, zz_nk, zz_nk_sigma)
plt.plot(hsamples, exact_zz12, '-',color = 'red',  label='ED', linewidth = 1.0)
plt.legend()
plt.ylabel('E(z1z2)', fontsize=18)
plt.xlabel('Omega', fontsize=16)
plt.savefig('Ising1d_corrz1z2.png')
plt.show()

plt.plot(hsamples, xxwf, label='from wf', linewidth=4.0)
plt.plot(hsamples, xx_nk, 'x', label='netket obs')
ax.error(hsamples, zz_nk, xx_nk_sigma)
plt.plot(hsamples, exact_xx12, '-',color = 'red',  label='ED', linewidth = 1.0)
plt.legend()
plt.ylabel('E(z1z2)', fontsize=18)
plt.xlabel('Omega', fontsize=16)
plt.savefig('Ising1d_corrx1x2.png')
plt.show()

plt.plot(hsamples, zwf, label = 'zi from wf')
plt.plot(hsamples, z_nk, 'x', label = 'zi from netket')
ax.plot(hsamples, z_nk, z_nk_sigma)
plt.plot(hsamples, exact_z1, '-',  label='ED', linewidth = 1.0)
plt.ylabel('E(zi) and E(z2)', fontsize=18)
plt.xlabel('h', fontsize=16)
plt.legend()
plt.savefig('Ising1d_zi.png')
plt.show()

plt.plot(hsamples, xwf, label = 'xi from wf')
plt.plot(hsamples, x_nk, 'x', label = 'zi from netket')
ax.plot(hsamples, x_nk, x_nk_sigma)
plt.plot(hsamples, exact_z1, '-',  label='ED', linewidth = 1.0)
plt.ylabel('E(zi) and E(z2)', fontsize=18)
plt.xlabel('h', fontsize=16)
plt.legend()
plt.savefig('Ising1d_xi.png')
plt.show()

#-----------------------SAVE THE RESULTS IN A FILE-----------------------------
results = {}

results['h'] = hsamples.tolist()
results['GE_netket'] = energies
results['zz_netket'] = zzwf
results['zz_netket_sigma'] = zz_nk_sigma
results['zi_netket'] = zwf
results['zi_netket_sigma'] = z_nk_sigma
results['xx_netket'] = zzwf
results['xx_netket_sigma'] = xx_nk_sigma
results['xi_netket'] = zwf
results['xi_netket_sigma'] = x_nk_sigma


            
json_file="results.json"
with open(json_file, 'w') as outfile:
    json.dump(results, outfile)