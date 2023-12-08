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
rho_snow            = 330       # density of (new) snow [kg/m³]
psi_s_min           = 0.05      # minimal solid fraction for a layer to count as sea ice


# importing data from runs
path_runs = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/'

path_k_0_01 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_01/output'
path_k_0_05 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_05/output'
path_k_0_1 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_1/output'
path_k_0_25 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_25/output'
path_k_0_5 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_5/output'
path_k_1_0 = path_runs + 'k/run_MOSAiC_FYI_ksnow_1_0/output'
path_k_1_5 = path_runs + 'k/run_MOSAiC_FYI_ksnow_1_5/output'

paths = [path_k_0_01, path_k_0_05, path_k_0_1, path_k_0_25, path_k_0_5, path_k_1_0, path_k_1_5]
runs = ['k_0_01','k_0_05','k_0_1','k_0_25','k_0_5', 'k_1_0', 'k_1_5']

dat = {}
for run, path in zip(runs,paths):
    dat[run] = {var: np.loadtxt(path + '/dat_'+var+'.dat') for var in ['T', 'S_bu' ,'psi_l', 'psi_s', 'thick', 
                                                                       'freeboard', 'snow', 'vital_signs', 'T2m_T_top']}
    
plot_labels = {'k_0_01':'k_snow = 0.01', 'k_0_05':'k_snow = 0.05', 'k_0_1': 'k_snow = 0.1', 
               'k_0_25': 'k_snow = 0.25', 'k_0_5': 'k_snow = 0.5', 'k_1_0': 'k_snow = 1.0', 'k_1_5': 'k_snow = 1.5'}    

#%% plot snow and ice thicknesses:
# plot of snow thickness
for run in runs: plt.plot(dat[run]['snow'][:,0], label = plot_labels[run])
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('snow thickness [m]')
plt.title('Snow thicknesses for different snow precipitations')
plt.show()

# plot of ice thickness
for run in runs: plt.plot(dat[run]['vital_signs'][:,3], label = plot_labels[run])
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
for run in runs: plt.plot(S_bu_mean_1[run], label = plot_labels[run])
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
for run in runs: plt.plot(S_bu_mean[run], label = plot_labels[run])
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('Mean bulk salinities, weighted by layer thickness')
plt.show()

# difference between corrected and not corrected weighted mean bulk salinities
for run in runs: plt.plot(S_bu_mean_1[run] - S_bu_mean[run], label = plot_labels[run])
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('differences between corrected and not corrected mean bulk salinities')
plt.show()



#%% determine point in time, where thickness reaches 1m:
#t_1m = {}
#for run in runs:
#    t_1m[run] = np.where(dat[run]['vital_signs'][:,3] >= 1.0)[0][0]
    
    
# take time of maximal thickness:
t_thick_max = {}
for run in runs:
    t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])

# plot weighted mean bulksalinity
for run in runs: plt.plot(S_bu_mean[run][:t_thick_max[run],], label = plot_labels[run])
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('Mean bulk salinities, weighted by layer thickness')
plt.show()


#%% get mass of every layer and calculate total amount shed salt

m = {}              # mass of layer per area [kg/m²]
S_bu_diff = {}      # difference to sea ice bulk salinity -> 'shed' bulk salinity
m_salt = {}
#M_salt = {}
for run in runs:
    m[run]              = ((rho_s * dat[run]['psi_s'] + rho_l * dat[run]['psi_l']) * dat[run]['thick'])[:t_thick_max[run],:]
    S_1m                = dat[run]['S_bu'][:t_thick_max[run],:]    
    S_bu_diff[run]      = np.full_like(S_1m, 34) - S_1m
    m_salt[run]         = np.sum(m[run] * S_bu_diff[run], axis = 1) #
    #M_salt[run]         = np.sum(m_salt[run])
    
for run in runs: plt.plot(m_salt[run], label = plot_labels[run])
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mass of salt [g/m²]')
plt.title('Accumulated total mass of shed salt until d_ice = 1m')
plt.show()

#%% plot mean bulk salinity against thickness:

for run in runs: plt.plot(dat[run]['vital_signs'][:t_thick_max[run],3]/dat[run]['vital_signs'][t_thick_max[run],3],
                          S_bu_mean[run][:t_thick_max[run],], label = plot_labels[run])
plt.legend()
plt.xlabel('relative ice thickness (to max thickness)')
plt.ylabel('mean bulk salinity [g/kg]')
plt.title('Mean bulk salinites at different snow precipitations')
plt.show()


#%% calculate corresponding snow thicknesses for heat conductivities:
def func_k(psi_l, psi_s):
    c0 = 0.138
    c1 = -1.01e-3
    c2 = 3.233e-6
    
    rho = psi_l * rho_l + psi_s * rho_s
    k = c0 + c1*rho + c2*rho**2
    return k

print('heat conductivity of snow in SAMSIM: k_snow = ', func_k(dat['k_1_0']['snow'][:,2],dat['k_1_0']['snow'][:,3]), 'W/(m*K)')


#%% plot T_top
for run in runs: plt.plot(dat[run]['T2m_T_top'][:,1], label = plot_labels[run])
#for run in runs: plt.plot(dat[run]['T2m_T_top'][:,0], label = plot_labels[run]) # Tq2m always the same -> prescribed from input
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('temperature [°C]')
plt.title('')

