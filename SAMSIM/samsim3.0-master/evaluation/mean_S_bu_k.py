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
import datetime
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from functions import load_data_grid


# constants from model
rho_l               = 920.      # density of liquid [kg/m³]
rho_s               = 1028.     # density of solid ice [kg/m³]
rho_snow            = 330.      # density of (new) snow [kg/m³]
psi_s_min           = 0.05      # minimal solid fraction for a layer to count as sea ice


# collect data: 
path_runs = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/'
path_runs_fbs1 = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/k/fbs_1/'
path_runs_ideal = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/k_idealized_forcing/'

#fbs = 1
path_k_0_01_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_0_01_fbs1'
path_k_0_1_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_0_1_fbs1'
path_k_0_15_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_0_15_fbs1'
path_k_0_2_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_0_2_fbs1'
path_k_0_3_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_0_3_fbs1'
path_k_0_4_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_0_4_fbs1'
path_k_0_5_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_0_5_fbs1'
path_k_1_0_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_1_0_fbs1'
path_k_1_5_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_1_5_fbs1'
path_k_2_0_fbs1 = path_runs_fbs1 +  'run_MOSAiC_FYI_ksnow_2_0_fbs1'

paths_fbs1 = {'k_0_01_fbs1': path_k_0_01_fbs1, 'k_0_1_fbs1': path_k_0_1_fbs1, 'k_0_15_fbs1': path_k_0_15_fbs1, 
              'k_0_2_fbs1': path_k_0_2_fbs1, 'k_0_3_fbs1': path_k_0_3_fbs1, 'k_0_4_fbs1': path_k_0_4_fbs1, 'k_0_5_fbs1': path_k_0_5_fbs1, 'k_1_0_fbs1': path_k_1_0_fbs1, 
              'k_1_5_fbs1': path_k_1_5_fbs1, 'k_2_0_fbs1': path_k_2_0_fbs1}

runs_fbs1 = list(paths_fbs1.keys())

#fbs = 0
path_k_0_005 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_005'
path_k_0_01 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_01'
path_k_0_05 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_05'
path_k_0_1 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_1'
path_k_0_175 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_175'
path_k_0_25 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_25'
path_k_0_5 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_5'
path_k_1_0 = path_runs + 'k/run_MOSAiC_FYI_ksnow_1_0'
path_k_1_5 = path_runs + 'k/run_MOSAiC_FYI_ksnow_1_5'


paths_fbs0 = {'k_0_005': path_k_0_005, 'k_0_01': path_k_0_01, 'k_0_05': path_k_0_05, 'k_0_1': path_k_0_1, 'k_0_175': path_k_0_175, 
         'k_0_25': path_k_0_25, 'k_0_5': path_k_0_5, 'k_1_0': path_k_1_0, 'k_1_5': path_k_1_5}

runs_fbs0 = list(paths_fbs0.keys())

#fbs=1, idealized forcing:
path_k_0_1_id = path_runs_ideal + 'run_MOSAiC_FYI_ksnow_0_1_fbs1_ideal'
path_k_0_2_id = path_runs_ideal + 'run_MOSAiC_FYI_ksnow_0_2_fbs1_ideal'
path_k_0_5_id = path_runs_ideal + 'run_MOSAiC_FYI_ksnow_0_5_fbs1_ideal'
path_k_1_0_id = path_runs_ideal + 'run_MOSAiC_FYI_ksnow_1_0_fbs1_ideal'
path_k_2_0_id = path_runs_ideal + 'run_MOSAiC_FYI_ksnow_2_0_fbs1_ideal'

paths_ideal = {'k_0_1_id': path_k_0_1_id, 'k_0_2_id': path_k_0_2_id, 'k_0_5_id': path_k_0_5_id, 
               'k_1_0_id': path_k_1_0_id, 'k_2_0_id': path_k_2_0_id}

runs_ideal = list(paths_ideal.keys())

paths = {**paths_fbs0, ** paths_fbs1, **paths_ideal}

runs = runs_fbs0 + runs_fbs1 + runs_ideal


plot_labels = {'k_0_005':'k_snow = 0.005', 'k_0_01':'k_snow = 0.01', 'k_0_05':'k_snow = 0.05', 'k_0_1': 'k_snow = 0.1', 'k_0_175': 'k_snow = 0.175', 
               'k_0_25': 'k_snow = 0.25', 'k_0_5': 'k_snow = 0.5', 'k_1_0': 'k_snow = 1.0', 'k_1_5': 'k_snow = 1.5', 
               'k_0_01_fbs1': 'k_snow = 0.01, m_snow = 0', 'k_0_1_fbs1': 'k_snow = 0.1, m_snow = 0', 'k_0_15_fbs1': 'k_snow = 0.15, m_snow = 0', 
               'k_0_2_fbs1': 'k_snow = 0.2, m_snow = 0', 'k_0_3_fbs1': 'k_snow = 0.3, m_snow = 0', 'k_0_4_fbs1': 'k_snow = 0.4, m_snow = 0', 'k_0_5_fbs1': 'k_snow = 0.5, m_snow = 0', 
               'k_1_0_fbs1': 'k_snow = 1.0, m_snow = 0', 'k_1_5_fbs1': 'k_snow = 1.5, m_snow = 0', 
               'k_2_0_fbs1': 'k_snow = 2.0, m_snow = 0',
               'k_0_1_id': 'k = 0.1, idealized forcing', 'k_0_2_id': 'k = 0.2, idealized forcing', 'k_0_5_id': 'k = 0.5, idealized forcing', 
               'k_1_0_id': 'k = 1.0, idealized forcing', 'k_2_0_id': 'k = 2.0, idealized forcing'}

k_values = {'k_0_005': 0.005, 'k_0_01': 0.01, 'k_0_05': 0.05, 'k_0_1': 0.1, 'k_0_175': 0.175, 
               'k_0_25': 0.25, 'k_0_5': 0.5, 'k_1_0': 1.0, 'k_1_5': 1.5, 
               'k_0_01_fbs1': 0.01, 'k_0_1_fbs1': 0.1, 'k_0_15_fbs1': 0.15, 'k_0_2_fbs1': 0.2, 'k_0_3_fbs1': 0.3, 'k_0_4_fbs1': 0.4, 'k_0_5_fbs1': 0.5, 
               'k_1_0_fbs1': 1.0, 'k_1_5_fbs1': 1.5, 'k_2_0_fbs1': 2.0, 
               'k_0_1_id': 0.1, 'k_0_2_id': 0.2, 'k_0_5_id': 0.5, 
               'k_1_0_id': 1.0, 'k_2_0_id': 2.0}


dat = {}
for run in runs:
    dat[run] = load_data_grid(paths[run], free_flag = 1)
    dat[run]['k'] = k_values[run]
    

# Load config file
with open(path_k_1_0_fbs1 + '/config.json') as json_file:
    config = json.load(json_file)

# build time axis
offset = pd.DateOffset(seconds=config['time_out'])
time = pd.date_range(config['start_time'], freq=offset, periods=config['time_total']/config['time_out'] + 1).to_series()
dx           = config['time_out']/(60*60*24)  # get dx in days
timeunit     = '[days]'

#%% colormap for k:
colormap = plt.cm.viridis
norm = matplotlib.colors.Normalize(vmin = min(k_values.values()), vmax = max(k_values.values()))
sm = matplotlib.cm.ScalarMappable(norm = norm, cmap = colormap)   

#%% plot snow and ice thicknesses:
# plot of snow thickness
figsize = (10,6)
matplotlib.rcParams.update({'font.size': 16})


fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1: ax.plot(dat[run]['snow'][:,0], label = plot_labels[run],
                              color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('snow thickness [m]')
ax.set_title('Snow thicknesses for different snow precipitations')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()


# plot of ice thickness
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]: ax.plot(dat[run]['vital_signs'][:,3], label = plot_labels[run],
                              color = colormap(norm(dat[run]['k'])))
ax.invert_yaxis()
ax.set_xlabel('time [day]')
ax.set_ylabel('total ice thickness [m]')
ax.set_title('total ice thickness')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/ice_thickness_fbs1.png')
plt.show()


#plot ice thickness for ideal forcings:
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ideal: ax.plot(dat[run]['vital_signs'][:,3], label = plot_labels[run],
                              color = colormap(norm(dat[run]['k'])))
ax.invert_yaxis()
ax.set_xlabel('time [day]')
ax.set_ylabel('total ice thickness [m]')
ax.set_title('total ice thickness for idealized forcings')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/ice_thickness_fbs1_ideal.png')
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
    S_bu_mean_1[run] = np.sum(dat[run]['S'] * dat[run]['thick'], axis = 1) / np.sum(dat[run]['thick'], axis = 1)
    
# plot of mean bulk salinity averaged weighted by layer thicknesses
fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]: ax.plot(S_bu_mean_1[run], label = plot_labels[run], 
                              color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('mean salinity [psu]')
ax.set_title('Mean bulk salinities, weighted by layer thickness')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()

#%% calculate weight for mean bulk salinity
# for layers within ice: weight = layer thickness
# for ocean-ice boundary layer : weight = (psi_s(boundary)/psi_s_min) * layer thickness
plt.legend()

thick_ice = {}
psi_s_ice = {}
S_ice = {}
for run in runs:
    thick_ice[run] = dat[run]['thick'][:,1:]
    psi_s_ice[run] = dat[run]['psi_s'][:,1:]
    S_ice[run] = dat[run]['S'][:,1:]
    


Nactive = {}
layer_weight = {}
S_bu_mean = {}
for run in runs:
    layer_weight[run] = thick_ice[run].copy()
    Nactive[run] = np.sum(thick_ice[run] != 0., axis = 1) - 1 # -1 to account for python indexing
    for i in np.arange(Nactive[run].shape[0]):
        layer_weight[run][i,Nactive[run][i]] = (psi_s_ice[run][i,Nactive[run][i]] / psi_s_min) * thick_ice[run][i,Nactive[run][i]]
    S_bu_mean[run] = np.sum(S_ice[run] * layer_weight[run], axis = 1) / dat[run]['vital_signs'][:,3]
 


'''
Nactive = {}
layer_weight = {}
S_bu_mean = {}
for run in runs_fbs1:
    layer_weight[run] = dat[run]['thick'].copy()
    Nactive[run] = np.sum(dat[run]['thick'] != 0., axis = 1) - 1 # -1 to account for python indexing
    for i in np.arange(Nactive[run].shape[0]):
        layer_weight[run][i,Nactive[run][i]] = (dat[run]['psi_s'][i,Nactive[run][i]] / psi_s_min) * dat[run]['thick'][i,Nactive[run][i]]
    S_bu_mean[run] = np.sum(dat[run]['S'] * layer_weight[run], axis = 1) / dat[run]['vital_signs'][:,3]

'''
# plot weighted mean bulksalinity
for run in runs_fbs1: plt.plot(S_bu_mean[run], label = plot_labels[run])
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('Mean bulk salinities, weighted by layer thickness')
plt.show()

# difference between corrected and not corrected weighted mean bulk salinities
for run in runs_fbs1: plt.plot(S_bu_mean_1[run] - S_bu_mean[run], label = plot_labels[run])
plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('differences between corrected and not corrected mean bulk salinities')
plt.show()



#%% determine point in time, where thickness reaches 1m:
#t_1m = {}colormap = plt.cm.viridis
norm = matplotlib.colors.Normalize(vmin = min(k_values.values()), vmax = max(k_values.values()))
sm = matplotlib.cm.ScalarMappable(norm = norm, cmap = colormap)   
#for run in runs:
#    t_1m[run] = np.where(dat[run]['vital_signs'][:,3] >= 1.0)[0][0]
    
    
# take time of maximal thickness:
t_thick_max = {}
for run in runs:
    t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])

# plot weighted mean bulksalinity
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]: ax.plot(S_bu_mean[run][:t_thick_max[run],], label = plot_labels[run],
                              color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('mean salinity [psu]')
ax.set_title('Mean bulk salinities')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/MBSovertime_fbs1.png')
plt.show()

# plot weighted mean bulksalinity
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ideal: ax.plot(S_bu_mean[run][:t_thick_max[run],], label = plot_labels[run],
                              color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('mean salinity [psu]')
ax.set_title('Mean bulk salinities with idealized forcing')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/MBSovertime_ideal.png')
plt.show()

#%% get mass of every layer and calculate total amount shed salt

m = {}              # mass of layer per area [kg/m²]
S_bu_diff = {}      # difference to sea ice bulk salinity -> 'shed' bulk salinity
m_salt = {}
#M_salt = {}
for run in runs:
    m[run]              = ((rho_s * dat[run]['psi_s'] + rho_l * dat[run]['psi_l']) * dat[run]['thick'])[:t_thick_max[run],:]
    S_1m                = dat[run]['S'][:t_thick_max[run],:]    
    S_bu_diff[run]      = np.full_like(S_1m, 34) - S_1m
    m_salt[run]         = np.sum(m[run] * S_bu_diff[run], axis = 1) #
    #M_salt[run]         = np.sum(m_salt[run])

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]: ax.plot(m_salt[run], label = plot_labels[run],
                                   color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('mass of salt [g/m²]')
ax.set_title('Accumulated total mass of shed salt until d_ice = 1m')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()


#%% plot mean bulk salinity against relative thickness:
t_plot = 75

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]: 
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3]/dat[run]['vital_signs'][t_thick_max[run],3],
             S_bu_mean[run][:t_thick_max[run],], label = plot_labels[run],
             color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('relative ice thickness (to max thickness)')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ideal: 
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3]/dat[run]['vital_signs'][t_thick_max[run],3],
             S_bu_mean[run][:t_thick_max[run],], label = plot_labels[run],
             color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('relative ice thickness (to max thickness)')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites for idealized forcings')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()


# plot mean bulk salinity against absolute thickness:
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]:
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3], 
             S_bu_mean[run][:t_thick_max[run],], label = plot_labels[run],
             color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites until maximum thickness')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/MBSoverthick_fbs1.png')
plt.show()  

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ideal:
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3], 
             S_bu_mean[run][:t_thick_max[run],], label = plot_labels[run],
             color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites until maximum thickness for idealized forcings')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/MBSoverthick_ideal.png')
plt.show()  
    
#%% plot mean bulk salinity against absolute thickness:
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05) 
for run in runs_fbs1[1:]:
    ax.plot(dat[run]['vital_signs'][:t_plot,3], 
             S_bu_mean[run][:t_plot,], label = plot_labels[run], 
             color = colormap(norm(dat[run]['k'])))
#plt.legend()
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites until mid-November')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/MBSoverthick_fbs1_midNov.png')
plt.show()


fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05) 
for run in runs_ideal:
    ax.plot(dat[run]['vital_signs'][:250,3], 
             S_bu_mean[run][:250,], label = plot_labels[run], 
             color = colormap(norm(dat[run]['k'])))
#plt.legend()
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites for idealized forcing')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()

#%% fitting a parameterisation S(thickness):

t_plot = 75    

def exp(x,a,b,c):
    f = a * np.exp(-b * x) + c
    return f

xdata = dat['k_0_1_fbs1']['vital_signs'][:t_plot,3][2:]
for run in runs_fbs1[2:]: xdata = np.append(xdata, dat[run]['vital_signs'][:t_plot,3][2:])
ydata = S_bu_mean['k_0_1_fbs1'][:t_plot,][2:]
for run in runs_fbs1[2:]: ydata = np.append(ydata, S_bu_mean[run][:t_plot,][2:])

X = np.linspace(0.05,1.05,50)

popt, pcov = curve_fit(exp, xdata, ydata)

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]:
    ax.plot(dat[run]['vital_signs'][:t_plot,3], 
             S_bu_mean[run][:t_plot,] - popt[2], #label = plot_labels[run], 
             color = colormap(norm(dat[run]['k'])))
ax.plot(X, exp(X, *popt) - popt[2], 
         color = 'red', linestyle = '--', 
         label = 'exponential fit: f(x) = ' + str(round(popt[0],2)) + 
                 '*exp(-' + str(round(popt[1],2)) + '*x) + ' + str(round(popt[2],2)))
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg] - 8.62 W/mK')
ax.set_title('Mean bulk salinites until mid-November')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/MBSoverthick_fbs1_midNov_fit_log.png')
plt.show()

#%% fitting parameterization on idealized forcings data:
xdata = dat['k_0_1_id']['vital_signs'][:t_plot,3][2:]
for run in runs_fbs1[2:]: xdata = np.append(xdata, dat[run]['vital_signs'][:t_plot,3][2:])
ydata = S_bu_mean['k_0_1_fbs1'][:t_plot,][2:]
for run in runs_fbs1[2:]: ydata = np.append(ydata, S_bu_mean[run][:t_plot,][2:])



#%% calculate corresponding snow thicknesses for heat conductivities:
def func_k(psi_l, psi_s):
    c0 = 0.138
    c1 = -1.01e-3
    c2 = 3.233e-6
    
    rho = psi_l * rho_l + psi_s * rho_s
    k = c0 + c1*rho + c2*rho**2
    return k

#print('heat conductivity of snow in SAMSIM: k_snow = ', func_k(dat['k_1_0']['snow'][:,2],dat['k_1_0']['snow'][:,3]), 'W/(m*K)')




#%% fitting idealized forcings
t_del = (31*5 + 30*2 + 28) * 24*60. # time from 01/01/2019 - 31/08/2019 in minutes


input_data = ['fl_lw', 'fl_sw', 'fl_sen', 'fl_lat', 'T2m', 'precip_l']
dat_input = {}
for i in input_data:
    dat_input[i] = np.loadtxt(path_k_1_0_fbs1 + '/input/' + i + '.txt')[int(t_del):]

y_labels = ['longwave heatflux [W/m²]', 'shortwave heatflux [W/m²]', 'sensible heatflux [W/m²]', 
            'latent heatflux [W/m²]', 'temperature [°C]', 'precipitation [m]']
j = 0
for i in input_data: 
    plt.plot(dat_input[i])
    plt.xlabel('time [min] since 01.01.2019')
    plt.ylabel(y_labels[j])
    plt.title(i)
    plt.show()
    j += 1



time_input = np.linspace(0,len(dat_input['T2m']), len(dat_input['T2m']))
def seasonalforcing(x,a,b,c,d):
    f = a * np.sin(b*x + c) + d
    return f
popt1, pcov1 =  curve_fit(seasonalforcing, time_input, dat_input['T2m'], p0 = (10, 1e-5,200000, -20))
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
ax.plot(time_input, dat_input['T2m'])
#cax = divider.append_axes('right', size='5%', pad=0.05)
ax.plot(time_input, seasonalforcing(time_input, *popt1), 
         color = 'red', linestyle = '--', label = 'idealized forcing')
ax.legend()
ax.set_xlabel('time [days]')
ax.set_ylabel('temperature [°C]')
ax.set_title('temperature forcing')
plt.savefig('PLOTS/forcing_temperature.png')
plt.show()

popt2, pcov2 =  curve_fit(seasonalforcing, time_input, dat_input['fl_lw'], p0 = (60, 1e-5, 200000, 200))
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
#cax = divider.append_axes('right', size='5%', pad=0.05)
ax.plot(time_input, dat_input['fl_lw'])
ax.plot(time_input, seasonalforcing(time_input, *popt2), 
         color = 'red', linestyle = '--', label = 'idealized forcing')
ax.legend()
ax.set_xlabel('time [days]')
ax.set_ylabel('longwave heatflux [W/m²]')
ax.set_title('longwave heatflux fl_lw')
plt.savefig('PLOTS/forcing_longwave.png')
plt.show()

def shortwave_forcing(x,a,b,c,d):
    f = a * np.sin(b*x + c) + d
    f[f<0]=0
    return f
popt3, pcov3 =  curve_fit(shortwave_forcing, time_input, dat_input['fl_sw'], p0 = (200, 5e-6, 200000, -70)) 
                          #bounds= ([-10,-100,1e-5,1.5e5,-100], [800,0,1.3e-5,2e5,100]))
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
#cax = divider.append_axes('right', size='5%', pad=0.05)
ax.plot(time_input, dat_input['fl_sw'])
ax.plot(time_input, shortwave_forcing(time_input, *popt3), 
         color = 'red', linestyle = '--', label = 'idealized forcing')
ax.legend()
ax.set_xlabel('time [days]')
ax.set_ylabel('shortwave heatflux [W/m²]')
ax.set_title('shortwave heatflux fl_sw')
plt.savefig('PLOTS/forcing_shortwave.png')
plt.show()

print('Fitting parameters:')
print('T2m: ', popt1)
print('longwave heatflux: ', popt2)
print('shortwave heatflux: ', popt3)

#%% comparing salinity profiles

# take time of maximal thickness:
t_thick_max = {}    
date_max_thick = {}
for run in runs:
    t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])
    #print('time of maximal thickness for ', plot_labels[run], ': ', 
    #      datetime.date(year = 2020, month = 9, day = 1) + datetime.timedelta(days=int(t_thick_max[run])))
    date_max_thick[run] = datetime.date(year = 2020, month = 9, day = 1) + datetime.timedelta(days=int(t_thick_max[run]))

    
for run in runs_fbs1:    
    plt.plot(dat[run]['S'][t_thick_max[run],1:],
             dat[run]['depth'][t_thick_max[run],1:]/(dat[run]['vital_signs'][t_thick_max[run],3]), 
             label = plot_labels[run] + ' at ' + str(date_max_thick[run]))
plt.legend()
plt.xlabel('Salinity [g/kg]')
plt.ylabel('depth realtive to maximal ice thickness')
plt.title('Salinity profiles at point of maximal ice thickness')
plt.show()

for run in runs_fbs1:    
    plt.plot(dat[run]['T'][t_thick_max[run],1:],
             dat[run]['depth'][t_thick_max[run],1:]/-dat[run]['depth'][t_thick_max[run],-1], 
             label = plot_labels[run])
plt.legend()
plt.xlabel('temperature [°C]')
plt.ylabel('depth realtive to maximal ice thickness')
plt.title('Temperature profiles at point of maximal ice thickness')
plt.show()


for run in runs_fbs1:    
    plt.plot(dat[run]['S'][228,1:],
             #dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             -dat[run]['depth'][228,1:]/dat[run]['depth'][228,-1], 
             label = plot_labels[run])
plt.legend()
plt.xlabel('Salinity [g/kg]')
plt.ylabel('depth realtive to maximal ice thickness')
plt.title('Salinity profiles in mid-april')
plt.show()



#%% profiles at same thickness:
    
t_90cm = {}
for run in runs_fbs1[1:]:
    t_90cm[run] = np.where(dat[run]['vital_signs'][:,3] >= 0.9)[0][0]


fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]:
    ax.plot(dat[run]['S'][t_90cm[run],1:],
             #dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             dat[run]['depth'][t_90cm[run],1:],#/dat[run]['depth'][t_1m[run],-1], 
             color = colormap(norm(k_values[run])),
             label = 'at ' + str(time[t_90cm[run]])[:-9])
ax.legend()
#ax.set_xlim(3.5,15)
ax.set_xlabel('Salinity [g/kg]')
ax.set_ylabel('depth [m]')
ax.set_title('Salinity profiles at 0.9m ice thickness')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()


t_90cm_id = {}
for run in runs_ideal:
    t_90cm_id[run] = np.where(dat[run]['vital_signs'][:,3] >= 0.9)[0][0]
    
    
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ideal:
    ax.plot(dat[run]['S'][t_90cm_id[run],1:],
             #dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             dat[run]['depth'][t_90cm_id[run],1:],#/dat[run]['depth'][t_90cm_id[run],-1], 
             color = colormap(norm(k_values[run])),
             label = 'at ' + str(time[t_90cm_id[run]])[:-9])
#ax.legend()
ax.set_ylim(-0.9,0.13)
for run in runs_ideal:
    ax.vlines(S_bu_mean[run][t_90cm_id[run]], -1,0.2,color = colormap(norm(k_values[run])), linestyle = 'dashed', alpha = 0.5)
ax.set_xlabel('Salinity [g/kg]')
ax.set_ylabel('depth [m]')
ax.set_title('Salinity profiles at 0.9m ice thickness with idealized forcings')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/salprofile_ideal_90cm.png')
plt.show()


#%% salinity profiles with mean bulk salinities at 90cm ice thickness
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]:    
    ax.plot(dat[run]['S'][t_90cm[run],1:],
             #dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             dat[run]['depth'][t_90cm[run],1:],#/dat[run]['depth'][t_1m[run],-1], 
             color = colormap(norm(k_values[run])),
             label = 'at ' + str(time[t_90cm[run]])[:-9])
ax.legend()
ax.set_xlim(4.25,10.5)
ax.set_ylim(-0.85,0.15)
for run in runs_fbs1[1:]:
    ax.vlines(S_bu_mean[run][t_90cm[run]], -1,0.2,color = colormap(norm(k_values[run])), linestyle = 'dashed', alpha = 0.8)
ax.set_xlabel('Salinity [g/kg]')
ax.set_ylabel('depth realtive to maximal ice thickness')
ax.set_title('Salinity profiles at 0.9m ice thickness, with mean bulk Salinity (dashed)')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/salproflie_fbs1_90cm.png')
plt.show()


#%% salinity profiles with mean bulk salinities at the smae time before warming event
t_plot = 75

fig, ax = plt.subplots()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_fbs1[1:]:    
    ax.plot(dat[run]['S'][t_plot,1:],
             #dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             dat[run]['depth'][t_plot,1:],#/dat[run]['depth'][t_1m[run],-1], 
             color = colormap(norm(k_values[run])))
             #label = 'at ' + str(time[t_plot])[:-9])
#ax.legend()
ax.set_xlim(4.25,13.5)
ax.set_ylim(-1,0.15)
for run in runs_fbs1[1:]:
    ax.vlines(S_bu_mean[run][t_plot], -1,0.2,color = colormap(norm(k_values[run])), linestyle = 'dashed', alpha = 0.8)
ax.set_xlabel('Salinity [g/kg]')
ax.set_ylabel('depth realtive to maximal ice thickness')
ax.set_title('Salinity profiles at ' + str(time[t_plot])[:-9] + ', with mean bulk Salinity (dashed)')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()


#%% calculate mean bulk salinities at t_90cm:

S_mean_90cm = {}
for run in runs_fbs1[1:]:
    S_mean_90cm[run] = np.sum(dat[run]['S'][t_90cm[run],1:] * layer_weight[run][t_90cm[run],:]) / dat[run]['vital_signs'][t_90cm[run],3]

S_mean_90cm


















