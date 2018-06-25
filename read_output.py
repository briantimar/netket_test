#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 00:18:12 2018

@author: btimar
"""
import json 
import numpy as np
import matplotlib.pyplot as plt

outfile = "isingtest.log"
iters = []
energy=[]
sigma=[]
evar = []
evarsig = []
L=2

obskeys = ['zz{0}{1}'.format(i, (i+1)%L) for i in range(L)]
obsdict = dict([(k, []) for k in obskeys])

with open(outfile) as f:
    data=json.load(f)
    
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Energy"]["Mean"])
        sigma.append(iteration["Energy"]["Sigma"])
        evar.append(iteration["EnergyVariance"]["Mean"])
        evarsig.append(iteration["EnergyVariance"]["Sigma"])
        for k in obskeys:
            obsdict[k].append(iteration[k]["Mean"])
plt.plot(iters, energy)

fig,ax=plt.subplots()
for k in obskeys:
    plt.plot(iters, obsdict[k],label=k)
plt.legend()