#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Wed Nov 22 11:16:00 2023

@author: jakobp
'''

# Loading modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib import rc
import matplotlib
import os
import json
import pandas as pd


# constants from model
rho_l               = 920       # density of liquid [kg/m³]
rho_s               = 1028      # density of solid ice [kg/m³]
psi_s_min           = 0.05      # minimal solid fraction for a layer to count as sea ice


# importing data from runs
path_snow = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/run_MOSAiC_FYI_realsnow/output'
path_double_snow = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/run_MOSAiC_FYI_double_realsnow/output'
path_half_snow = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/run_MOSAiC_FYI_half_realsnow/output'

paths = [path_snow, path_double_snow, path_half_snow]
runs = ['real','double','half']

dat = {}
for run, path in zip(runs,paths):
    dat[run] = {var: np.loadtxt(path + '/dat_'+var+'.dat') for var in ['T', 'S_bu' ,'psi_l', 'psi_s', 'thick', 'freeboard', 'snow', 'vital_signs']}
    
#%% plot snow and ice thicknesses:
# plot of snow thickness
plt.plot(dat['real']['snow'][:,0], label = 'real precip_l')
plt.plot(dat['double']['snow'][:,0], label = 'double precip_l')
plt.plot(dat['half']['snow'][:,0], label = 'half precip_l')
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('snow thickness [m]')
plt.title('Snow thicknesses for different snow precipitations')
plt.show()

# plot of ice thickness
plt.plot(dat['real']['vital_signs'][:,3], label = 'real precip_l')
plt.plot(dat['double']['vital_signs'][:,3], label = 'double precip_l')
plt.plot(dat['half']['vital_signs'][:,3], label = 'half precip_l')
plt.legend()
plt.xlabel('time [day]')
plt.ylabel('total ice thickness [m]')
plt.title('total ice thickness for different precipitations')
plt.show()


#%% get Nactive, number of active layer to deal with bottom layer:
'''
Nactive = {}
for run in runs: 
    Nactive[run] = np.sum(dat[run]['thick'] != 0., axis = 1) - 1 # -1 to account for python indexing
psi_s_bound = np.zeros_like(Nactive['real'], dtype = 'float')
psi_s_bound_plus1 = np.zeros_like(Nactive['real'], dtype = 'float')
psi_s_bound_min1 = np.zeros_like(Nactive['real'], dtype = 'float')
for i in np.arange(Nactive['real'].shape[0]):
    psi_s_bound[i] = dat['real']['psi_s'][i,Nactive['real'][i]]
    psi_s_bound_plus1[i] = dat['real']['psi_s'][i,np.min([Nactive['real'][i]+1,79])]
    psi_s_bound_min1[i] = dat['real']['psi_s'][i,Nactive['real'][i]-1]

plt.plot(psi_s_bound)
plt.show()
plt.plot(psi_s_bound_plus1)
plt.show()
plt.plot(psi_s_bound_min1)
plt.show()
plt.plot(Nactive['real'])
plt.show()
plt.plot(Nactive['double'])
plt.show()
plt.plot(Nactive['half'])
plt.show()
'''

#%% calculate mean bulk salinity weighted by layer thickness:
S_bu_mean_1 = {}
for run in runs:
    S_bu_mean_1[run] = np.sum(dat[run]['S_bu'] * dat[run]['thick'], axis = 1) / np.sum(dat[run]['thick'], axis = 1)
    
# plot of mean bulk salinity averaged weighted by layer thicknesses
plt.plot(S_bu_mean_1['real'], label = 'real precip_l')
plt.plot(S_bu_mean_1['double'], label = 'double precip_l')
plt.plot(S_bu_mean_1['half'], label = 'half precip_l')
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('Mean bulk salinities, weighted by layer thickness')
plt.show()

#%% calculate weight for mean bulk salinity
# for layers within ice: weight = layer thickness
# for ocean-ice boundary layer : weight = (psi_s(boundary)/psi_s_min) * layer thickness
Nactive = {}
layer_weight = {}
S_bu_mean = {}
for run in runs:
    layer_weight[run] = dat[run]['thick'].copy()
    Nactive[run] = np.sum(dat[run]['thick'] != 0., axis = 1) - 1 # -1 to account for python indexing
    for i in np.arange(Nactive[run].shape[0]):
        layer_weight[run][i,Nactive[run][i]] = (dat[run]['psi_s'][i,Nactive[run][i]] / psi_s_min) * dat[run]['thick'][i,Nactive[run][i]]
    S_bu_mean[run] = np.sum(dat[run]['S_bu'] * layer_weight[run], axis = 1) / dat[run]['vital_signs'][:,3]
    
# plot weighted mean bulksalinity
plt.plot(S_bu_mean['real'], label = 'real precip_l')
plt.plot(S_bu_mean['double'], label = 'double precip_l')
plt.plot(S_bu_mean['half'], label = 'half precip_l')
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('Mean bulk salinities, weighted by layer thickness')
plt.show()

# difference between corrected and not corrected weighted mean bulk salinities
plt.plot(S_bu_mean_1['real'] - S_bu_mean['real'], label = 'real precip_l')
plt.plot(S_bu_mean_1['double'] - S_bu_mean['double'], label = 'double precip_l')
plt.plot(S_bu_mean_1['half'] - S_bu_mean['half'], label = 'half precip_l')
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('differences between corrected and not corrected mean bulk salinities')
plt.show()



#%% determine point in time, where thickness reaches 1m:
t_1m = {}
for run in runs:
    t_1m[run] = np.where(dat[run]['vital_signs'][:,3] >= 1.)[0][0]

# plot weighted mean bulksalinity
plt.plot(S_bu_mean['real'][:t_1m['real'],], label = 'real precip_l')
plt.plot(S_bu_mean['double'][:t_1m['double'],], label = 'double precip_l')
plt.plot(S_bu_mean['half'][:t_1m['half'],], label = 'half precip_l')
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('Mean bulk salinities, weighted by layer thickness')
plt.show()


#%% get mass of every layer and calculate total amount shed salt

m = {}              # mass of layer per area [kg/m²]
S_bu_diff = {}      # difference to sea ice bulk salinity -> 'shed' bulk salinity
m_salt = {}
M_salt = {}
for run in runs:
    m[run]              = ((rho_s * dat[run]['psi_s'] + rho_l * dat[run]['psi_l']) * dat[run]['thick'])[:t_1m[run],:]
    S_1m                = dat[run]['S_bu'][:t_1m[run],:]    
    S_bu_diff[run]      = np.full_like(S_1m, 34) - S_1m
    m_salt[run]         = np.sum(m[run] * S_bu_diff[run], axis = 1)
    M_salt[run]         = np.sum(m_salt[run])
    
plt.plot(m_salt['real'], label = 'real')    
plt.plot(m_salt['double'], label = 'double')
plt.plot(m_salt['half'], label = 'half')
plt.legend()

print(M_salt)
#%%
S_bu_1m = 
S_bu_seaw = 
S_bu_diff = S_bu_1m - S_bu_seaw



plt.pcolor(S_bu_diff)
plt.colorbar()

