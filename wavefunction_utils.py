import numpy as np
import json
#Here are a few functions to reconstruct tha Wave Function given a .wf file
def get_bin_sites(L,i):
    sites = []
    while i > 0:
        if i % 2 == 1: sites.append(1)
        else: sites.append(-1)
        i = int(i/2)
    sites = sites + [-1]*(L-len(sites))
    sites = np.array(sites)[::-1]
    return sites


#Constructs a basis of 2**L strings of 1s and zeroes.
#Returns a ductionary with the basis vectors and how they are numbered and an array of them
def construct_basis(L):
    basis_dic = {}
    basis = []
    for i in range(2**L):
        sites = get_bin_sites(L,i)
        basis_dic[tuple(sites)] = i
        basis.append(np.asarray(sites))
    
    return basis_dic, basis

#calculates the Fourier coefficients based on the machine parameters W, a and b
def fourier_coefficient(basis_vec, a, b, W):
    prefactor = np.exp(np.dot(a, basis_vec))
    theta = np.dot(np.transpose(W), basis_vec) + b
    return prefactor*np.prod(2*np.cosh(theta), axis= 0)

#This function loads the parameters from the file prefix.wf
def load_parameters(file):
    data = json.load(open(file))
    rows = len(data["Machine"]["a"])
    cols = len(data["Machine"]["b"])
    a, b = np.zeros(rows, dtype=complex), np.zeros(cols, dtype=complex)
    W    = np.zeros((rows, cols), dtype=complex)
    
    for i in range(rows):
        xa,ya = data["Machine"]["a"][i][0], data["Machine"]["a"][i][1]
        a[i] = complex(xa,ya)
    for j in range(cols):
        xb,yb = data["Machine"]["b"][j][0], data["Machine"]["b"][j][1]
        b[j] = complex(xb,yb)
    
    for i in range(rows):
        for j in range(cols):
            xw,yw = data["Machine"]["W"][i][j][0], data["Machine"]["W"][i][j][1]
            W[i][j] = complex(xw, yw)
    
    return a, b, W
    
#This function constructs the normalized wave function  
def wave_function_normalized(L, a, b, W):
    wavefunc_dic = {}
    wavefunc = []
    
    #We construct the wave function unnormalized
    basis_dic, basis = construct_basis(L)
    for basis_vec in basis:
        wavefunc.append(fourier_coefficient(basis_vec, a, b, W))
    
    #Normalize the array    
    wavefunc = wavefunc/np.sqrt(np.sum(np.abs(np.array(wavefunc))**2))
    #Checking
    if np.sum(np.abs(wavefunc)**2) == 1: print("Wave function normalized!")
    else: print("Norm is %f" %np.sum(np.abs(wavefunc)**2))
    
    #Gives is the normalized dictionary
    for basis_vec in basis_dic.keys():
        wavefunc_dic[basis_vec] = wavefunc[basis_dic[basis_vec]].tolist()
        
    return wavefunc, wavefunc_dic
    
#This function measures the intanglement entropy of a given state using the swap
#operator trick
def entanglement_entropy(wavefunc):
    x = 0
    for state in wavefunc:
        for otherstate in wavefunc:
            x += abs(state)**2*abs(otherstate)**2
    
    return -np.Log(x)
#Constructs the tensor product given a list of operators
def constr_op(oplist):
    """ given a list of local ops, return the tensor product """
    o = oplist[0]
    for i in range(1, len(oplist)):
        o = np.kron(o, oplist[i])
    return o
#Returns the expectation value of an observable op in state psi
def expect_val(op, psi):
    bra = np.conj(np.transpose(psi))
    ket = op.dot(psi)
    return bra.dot(ket)
