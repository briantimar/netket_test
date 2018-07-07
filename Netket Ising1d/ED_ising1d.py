from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d, boson_basis_1d
import numpy as np 
import json
import matplotlib.pyplot as plt
import scipy as sc

def z_op(i,basis,check_symm=True,dtype=np.float64):
    actson = [[1,i]] 
    static = [["z", actson ]]
    dynamic=[]
    return hamiltonian(static,dynamic,basis=basis,dtype=dtype,check_symm=check_symm)

def zz_op(i,j, basis, check_symm=True, dtype=np.float64):
    actson = [[1,i,j]]
    static = [["zz", actson]]
    dynamic = []
    return hamiltonian(static, dynamic, basis=basis, dtype = dtype, check_symm = check_symm)

def x_op(i,basis,check_symm=True,dtype=np.float64):
    actson = [[1,i]] 
    static = [["x", actson ]]
    dynamic=[]
    return hamiltonian(static,dynamic,basis=basis,dtype=dtype,check_symm=check_symm)

def get_sent2(psi, basis, subsys=None, return_rdm=None):
    """Return the entanglement entropy  S_2 of psi, living in basis <basis>, 
    computed in the reduced subsystem specified by subsys
    subsys = list of site labels [0, 1, ..., k] specifying the subsystem. 
    If subsys=None,  defaults to 0....N/2 -1
    
    return_rdm can be specified as 'A' (the subsystem of interest), 
    'B', or both; if so a dictionary is returned
    """
    if subsys is None:
        #the default quspin block
        subsys=tuple(range(basis.N//2))
    
    sdict= basis.ent_entropy(psi, sub_sys_A=subsys,return_rdm=return_rdm, alpha=2.0)
    # the quspin value is normalized by the subsystem size
    SA= sdict['Sent_A'] * len(subsys) 
    if return_rdm is not None:
        sdict['Sent_A']=SA        
        return sdict
    return SA

def make_tlfi(L, J, I, h, basis):
#    J, I, h  = 4*J, 2*I, 2*h
    #Constructing the Hamiltonian
    J_zz=[[-J,i,(i+1)%L] for i in range(L)] # PBC
    h_field=[[-h,i] for i in range(L)]
    I_field=[[I,i] for i in range(L)]
    static = [['zz', J_zz], ['x', h_field], ['z', I_field]]
    dynamic = []
    H=hamiltonian(static,dynamic,basis=basis,check_symm = True, dtype=np.float64)
    energy, psi = H.eigh()

    return energy[0], psi[:,0], energy[1], psi[:,1]

#------------------------------Creating the Cut--------------------------------
'''The following Code is meant to give us plots for energy, entanglement
   entropy, Z1, Z2, Z1Z1 for a cut in parameter space of the Ising1d model'''
#Defining the parameters for our Hamiltonian
L = 4
J = 1  
I = 0
h_min, h_max = 0, 1.5
Nsamples = 50
hsamples = np.linspace(h_min, h_max, Nsamples)

#Defining a bsis and list to store the values of the obsevables' exp vals
basis = spin_basis_1d(L)
groundenergy = []
firstexenergy = []
en_entropy = []
exp_val_z1 = []
exp_val_z2 = []
exp_val_x1 = []
exp_val_x2 = []
exp_val_z1z2 = []
exp_val_z2z1 = []

#Defining a subsystem for the entanglement entropy and the operators
subsystem=[i for i in range(L//2)]
z1, z2, z1z2, z2z1 = z_op(0, basis), z_op(1, basis), zz_op(0,1, basis), zz_op(1,0, basis)
x1,x2 = x_op(0, basis), x_op(1, basis)

#Mining the information for the cut
for h in hsamples:
    energy0, psi0, energy1, psi1 = make_tlfi(L, J, I, h, basis)
    groundenergy.append(energy0)
    firstexenergy.append(energy1)
    exp_val_z1.append(z1.expt_value(psi0))
    exp_val_z2.append(z2.expt_value(psi0))
    exp_val_x1.append(x1.expt_value(psi0))
    exp_val_x2.append(x2.expt_value(psi0))
    exp_val_z1z2.append(z1z2.expt_value(psi0))
    exp_val_z2z1.append(z2z1.expt_value(psi0))
    Sent=get_sent2(psi0, basis, subsystem)
    en_entropy.append(Sent)

#Plotting time
plt.plot(hsamples, groundenergy)
plt.plot(hsamples, firstexenergy)
plt.xlabel('h')
plt.ylabel('Ground and First excited Energies')
plt.savefig('Exact_ge_fe.png')
plt.show()

plt.plot(hsamples, exp_val_z1, label = 'E(Z1)')
plt.plot(hsamples, exp_val_z2, label = 'E(Z2)')
plt.ylabel('E(Z1) and E(Z2)')
plt.legend()
plt.xlabel('h')
plt.savefig('Exact_exp_val_z1_z2.png')
plt.show()

plt.plot(hsamples, exp_val_x1, label = 'E(Z1)')
plt.plot(hsamples, exp_val_x2, label = 'E(Z2)')
plt.ylabel('E(x1) and E(x2)')
plt.legend()
plt.xlabel('h')
plt.savefig('Exact_exp_val_x1_x2.png')
plt.show()


plt.plot(hsamples, exp_val_z1z2, label = 'E(z1z2)')
plt.plot(hsamples, exp_val_z2z1, label = 'E(z2z1)')
plt.ylabel('E(Z1Z2) and E(Z2Z1)')
plt.legend()
plt.xlabel('h')
plt.savefig('Exact_exp_val_z1z2_z2z1.png')
plt.show()

plt.plot(hsamples, en_entropy)
plt.xlabel('h')
plt.ylabel('Entanglement Entropy')
plt.savefig('Exact_ent_entropy.png')
plt.show()

#-----------------------SAVE THE RESULTS IN A FILE-----------------------------
exactresults = {}

exactresults['h'] = hsamples.tolist()
exactresults['GE'] = groundenergy
exactresults['EE'] = firstexenergy
exactresults['z1'] = exp_val_z1
exactresults['z2'] = exp_val_z2
exactresults['x1'] = exp_val_x1
exactresults['x2'] = exp_val_x2
exactresults['z1z2'] = exp_val_z1z2
exactresults['z2z1'] = exp_val_z2z1
exactresults['entanglement_entropy'] = en_entropy
            
json_file="exact_expectationvalues.json"
with open(json_file, 'w') as outfile:
    json.dump(exactresults, outfile)