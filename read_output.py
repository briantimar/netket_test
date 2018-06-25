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
L=12
obskeys = dict([('z'+str(i), []) for i in range(L)])


with open(outfile) as f:
    data=json.load(f)
    for iteration in data["Output"]:
        iters.append(iteration["Iteration"])
        energy.append(iteration["Energy"]["Mean"])
        sigma.append(iteration["Energy"]["Sigma"])
        evar.append(iteration["EnergyVariance"]["Mean"])
        evarsig.append(iteration["EnergyVariance"]["Sigma"])
        for i in range(L):
            obskeys['z'+str(i)].append(iteration['z'+str(i)]['Mean'])
plt.plot(iters, energy)

fig,ax=plt.subplots()
for i in range(L):
    plt.plot(iters, obskeys['z'+str(i)],label=i)
plt.legend()