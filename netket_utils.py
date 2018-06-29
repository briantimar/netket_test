#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:00:56 2018

@author: brian

"""
import numpy as np
import json

def to_complex(arr):
    return arr[...,0] + 1j * arr[..., 1]

def parse_netket_arr(arr,dtype=np.float64):
    """ returns complex weights from netket lists"""
    return to_complex(np.asarray(arr, dtype=dtype))

def gen_basis_strings(L, qnums=[0,1]):
    """Returns all basis strings of length L where the values are drawn from qnums"""
    sps =len(qnums)
    if L ==1:
        return np.asarray(qnums, dtype=int).reshape((1,sps))
    else:
        sub_basis_str = gen_basis_strings(L-1,qnums=qnums)
        Nsub = sub_basis_str.shape[1]
        basis_str = np.empty((L, sps**L),dtype=int)
        for d in range(sps):
            darr = qnums[d] * np.ones((1,Nsub))
            basis_str[:, d*Nsub:(d+1)*Nsub] = np.concatenate((sub_basis_str, darr),axis=0)
        return basis_str

def get_weights(wf_file):
    """ returns list of a, b, W -- the three RBM weight arrays
        shape of a: (nvis,)
        shape of b: (nhid,)
        shape of W: (nvis,nhid)
        all three are complex valued"""
    with open(wf_file) as f:
        pdict = json.load(f)["Machine"]
        a = parse_netket_arr(pdict["a"])
        b = parse_netket_arr(pdict["b"])
        W = parse_netket_arr(pdict["W"])
        return a, b, W
    
def eval_wf(a, b, W, s):
    """ evaluate rbm wf with the specified weight arrays. 
        s is a vector of values for the visible nodes, ie physical spins.
        an array of s values can be supplied -- in that case the first dimension should be number of visible nodes."""
    try:
        Nv, Ns = s.shape
    except ValueError:
        Nv = len(s)
        Ns = 1
    Nh = len(b)
    if len(a)!=Nv:
        raise ValueError
    s=np.reshape(s, (Nv, Ns))
    if W.shape != (Nv, Nh):
        raise ValueError
    b =b.reshape((Nh, 1))
    b = np.repeat(b, Ns, axis=1)
    pref=np.exp(np.dot(a,s))   #shape (Ns,)
    theta = np.dot(np.transpose(W), s) + b  #shape (Nhid, Ns)
    return np.prod( 2 * np.cosh(theta), axis=0) * pref

def normalize(psi):
    return psi /np.sqrt( np.sum( np.abs(psi)**2))

def constr_op(oplist):
    """ given a list of local ops, return the tensor product """
    o = oplist[0]
    for i in range(1, len(oplist)):
        o = np.kron(o, oplist[i])
    return o

def matrix_el(op, psi):
    """ returns <psi| op |psi>"""
    psidag = np.conj(np.transpose(psi))
    return np.dot(psidag, np.dot(op, psi))  

def run_netket(filegen, nrun=1,parallel=False,nproc=1):
    """ Calls netket to run on the json file which is returned by filegen().
        filegen(i): function which returns the json param file to be used for run i.
        nrun: how many runs. filegen will be called on i=0, ..., nrun-1
        
        parallel: if true passes to mpi with number of process nproc 
        Returns: list of retcodes from the calls"""
    if parallel:
        base_call = ["mpirun", "-n", str(nproc), "netket"]
    else:
        base_call = ["netket"]
    from subprocess import call
    retcodes = []
    for i in range(nrun):
        retcodes.append(call(base_call + [filegen(i)]))
    return retcodes
        
        
def get_netket_obs(i, ofilegen, opname,iternum=-1):
    """ returns mean val and std dev from last iteration
         i: integer lableling the outputfile
         ofilegen: returns, when called on i, the OutputFile that was fed to netket (not including .log)
            
         opname: name of the observable
         iternum: which iteration to draw from"""
    of = ofilegen(i)+".log"
    with open(of) as of:
        data=json.load(of)["Output"]
        return data[iternum][opname]["Mean"], data[iternum][opname]["Sigma"]
    
    
    
def get_trace(i, ofilegen, opname):
    """ returns mean value and std dev of opname for all iterations.
        i and ofilegen are as in get_netket_obs.
        Returns lists: mean, sigma"""
        
    of = ofilegen(i)+".log"
    with open(of) as of:
        data=json.load(of)["Output"]
        niter=len(data)
        return [ data[i][opname]["Mean"] for i in range(niter)], [data[i][opname]["Sigma"] for i in range(niter)]






