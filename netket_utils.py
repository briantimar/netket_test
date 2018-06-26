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
    return to_complex(np.asarray(arr, dtype=dtype))

def gen_basis_strings(L, qnums=[0,1]):
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
    psidag = np.conj(np.transpose(psi))
    return np.dot(psidag, np.dot(op, psi))  





