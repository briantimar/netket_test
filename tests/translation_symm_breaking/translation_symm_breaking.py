#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 08:40:53 2018

@author: brian

Disagreement among on-site ops in some parameter regimes

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../..')
sys.path.append('/home/brian/Documents/ryd-theory-code/ryd-theory/python_code')
savepath = "/home/brian/Dropbox/docs/netket/t_symm_breaking/"

from netket_constructor import make_tfi_pdict, constr_learning_dict, make_r6_hamiltonian, make_r6_pdict
import json
Delta=0.1
V=1
ktrunc=1
L=2
alpha=1.0

Niter=500

pfile_base = "tsb_params"
ofile_base = "tsb_output"

def ofilegen(i):
    return ofile_base + "_{0}".format(i)
def pfilegen(i):
    return pfile_base + "_{0}.json".format(i)

Nsamp =20
gmin = 0
gmax = 1
gvals = np.linspace(gmin, gmax, Nsamp)

from netket_constructor import get_loc_obs_dict, sigmaz
observables = [get_loc_obs_dict(ii, sigmaz(), name='z') for ii in range(L)]
obsnames = ['z'+str(i) for i in range(L)]

z_by_lr = dict()

def get_zvals(lr):
    
    for i in range(Nsamp):
        ofile = ofilegen(i)
        Omega = gvals[i]
        learning_dict = constr_learning_dict(OutputFile=ofile, LearningRate=lr,NiterOpt=Niter)
        pdict = make_r6_pdict(Delta,Omega,V,ktrunc, L, obslist = observables, learning_dict=learning_dict, alpha=alpha)
    
        with open(pfilegen(i),'w') as pf:
            json.dump(pdict,pf)
        
    
    nproc = 5
    from netket_utils import run_netket, get_netket_obs, get_trace
    retcodes=run_netket(pfilegen, nrun=Nsamp,nproc=nproc,parallel=True)
    
    def get_all(obsname):
        o = [ get_netket_obs(i, ofilegen, obsname) for i in range(Nsamp)]
        omean, osigma = [t[0] for t in o], [t[1] for t in o]
        return omean, osigma
    
    Etrace, sigmaEtrace =get_trace(5, ofilegen, "Energy")
    fig,ax=plt.subplots()
    for i in range(Nsamp):
        E,s=get_trace(i, ofilegen, "Energy")
        plt.plot(E,label="trial {0}".format(i))
    
    
    E, sigmaE = get_all('Energy')
    zvals = dict([ (nm,  get_all(nm)) for nm in obsnames])
    return zvals
    

zvals_ed = []
e_ed=[]
import tools
from ryd_base import make_r6_1d

basis=tools.get_hcb_basis(L)
from ryd_base import n_op
n0 = n_op(0, basis)

for g in gvals:
    h = make_r6_1d(Delta,g,V,ktrunc,basis,bc='periodic')
    e,s = h.eigsh(k=1,which='SA')
    psi=s[:,0]
    e_ed.append(e)
    zvals_ed.append(2*n0.expt_value(psi)-1)


lr_samp = [1, .1, .01]
for lr in lr_samp:
    z_by_lr[lr] = get_zvals(lr)

#
#fig,ax=plt.subplots()
#plt.plot(gvals, zvals_ed, label='ed')
fig,ax=plt.subplots()
for lr in lr_samp:
    zvals=z_by_lr[lr]
    plt.plot(gvals, np.asarray(zvals['z1'][0]) - np.asarray(zvals['z0'][0]), '-x', label="lr="+str(lr))
    #for nm in ['z0', 'z1']:
        #plt.errorbar(gvals, zvals[nm][0], yerr=zvals[nm][1], marker='x', ls='none',label="lr={0:.2e}".format(lr) +nm)
plt.legend()
plt.xlabel('Omega')
plt.ylabel('z diff')
plt.title('z1 - z0; ryd model, various learning rates')
plt.title('ryd L={0}_niter={2}_alpha={3}'.format(L,lr,Niter,alpha))
fig.savefig(savepath +"ryd_L={0}_niter={1}_alpha={2}.png".format(L,Niter,alpha))

