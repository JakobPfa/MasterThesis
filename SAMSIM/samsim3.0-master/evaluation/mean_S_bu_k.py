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



#%% constants from model
rho_l               = 920.      # density of liquid [kg/m³]
rho_s               = 1028.     # density of solid ice [kg/m³]
rho_snow            = 330.      # density of (new) snow [kg/m³]
psi_s_min           = 0.05      # minimal solid fraction for a layer to count as sea ice


#%% collect data: 
path_runs = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/'
path_runs_ksnow_fbs0 = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/ksnow_fbs_0/'
path_runs_ksnow_fbs1 = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/ksnow_fbs_1/'
path_runs_ksnow_idf = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/ksnow_idf/'
path_runs_oh_nosnow = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/OceanicHeatflux_nosnow/'
path_runs_oh_fbs1 = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/OceanicHeatflux_fbs1/'
path_runs_oh_idf = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/OceanicHeatflux_idf/'
path_runs_const_T = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/constant_T/'

# oceanic heating forcing with no snow at all
paths_oh_nosnow = {}
for i in [name for name in os.listdir(path_runs_oh_nosnow)]:
    paths_oh_nosnow[i[15:]] = (path_runs_oh_nosnow + i)
runs_oh_nosnow = list(paths_oh_nosnow.keys())
runs_oh_nosnow = ['oh0_nosnow', 'oh10_nosnow', 'oh20_nosnow', 'oh50_nosnow', 'oh100_nosnow']

# oceanic heating forcing with massless snow (fbs = 1)
paths_oh_fbs1 = {}
for i in [name for name in os.listdir(path_runs_oh_fbs1)]:
    paths_oh_fbs1[i[15:]] = (path_runs_oh_fbs1 + i)
runs_oh_fbs1 = list(paths_oh_fbs1.keys())
#runs_oh_nosnow = ['oh0_nosnow', 'oh10_nosnow', 'oh20_nosnow', 'oh50_nosnow', 'oh100_nosnow']

# oceanic heating forcing with massless snow (fbs = 1)
paths_oh_idf = {}
for i in [name for name in os.listdir(path_runs_oh_idf)]:
    paths_oh_idf[i[15:]] = (path_runs_oh_idf + i)
runs_oh_idf = list(paths_oh_idf.keys())

# heat conductivity forcing with fbs = 1 (massless snow)
paths_ksnow_fbs1 = {}
for i in [name for name in os.listdir(path_runs_ksnow_fbs1)]:
    paths_ksnow_fbs1[i[15:]] = (path_runs_ksnow_fbs1 + i)
runs_ksnow_fbs1 = list(paths_ksnow_fbs1.keys())
runs_ksnow_fbs1.remove('ksnow_0_01_fbs1') # lowest heat conductivity, no good output

# heat conductivity forcing with fbs = 0 (snow has mass)
paths_ksnow_fbs0 = {}
for i in [name for name in os.listdir(path_runs_ksnow_fbs0)]:
    paths_ksnow_fbs0[i[15:]] = (path_runs_ksnow_fbs0 + i)
runs_ksnow_fbs0 = list(paths_ksnow_fbs0.keys())

# heat conductivity forcing with idealized forcing (seasonal sine curve forcing) and fbs=1 (massless snow)
paths_ksnow_idf = {}
for i in [name for name in os.listdir(path_runs_ksnow_idf)]:
    paths_ksnow_idf[i[15:]] = (path_runs_ksnow_idf + i)
runs_ksnow_idf = list(paths_ksnow_idf.keys())

# constant temperature forcing (heating plate experiment) 
paths_const_T = {}
for i in [name for name in os.listdir(path_runs_const_T)]:
    paths_const_T[i[4:]] = (path_runs_const_T + i)
runs_const_T = list(paths_const_T.keys())


paths = {**paths_ksnow_fbs0, ** paths_ksnow_fbs1, **paths_ksnow_idf, **paths_oh_nosnow, 
         **paths_oh_fbs1, **paths_oh_idf, **paths_const_T}

runs = runs_ksnow_fbs0 + runs_ksnow_fbs1 + runs_ksnow_idf + runs_oh_nosnow + \
        runs_oh_fbs1 + runs_oh_idf + runs_const_T




plot_labels = {'k_0_005':'k_snow = 0.005', 'k_0_01':'k_snow = 0.01', 'k_0_05':'k_snow = 0.05', 'k_0_1': 'k_snow = 0.1', 'k_0_175': 'k_snow = 0.175', 
               'k_0_25': 'k_snow = 0.25', 'k_0_5': 'k_snow = 0.5', 'k_1_0': 'k_snow = 1.0', 'k_1_5': 'k_snow = 1.5', 
               'k_0_01_fbs1': 'k_snow = 0.01, m_snow = 0', 'k_0_1_fbs1': 'k_snow = 0.1, m_snow = 0', 'k_0_15_fbs1': 'k_snow = 0.15, m_snow = 0', 
               'k_0_2_fbs1': 'k_snow = 0.2, m_snow = 0', 'k_0_3_fbs1': 'k_snow = 0.3, m_snow = 0', 'k_0_4_fbs1': 'k_snow = 0.4, m_snow = 0', 'k_0_5_fbs1': 'k_snow = 0.5, m_snow = 0', 
               'k_1_0_fbs1': 'k_snow = 1.0, m_snow = 0', 'k_1_5_fbs1': 'k_snow = 1.5, m_snow = 0', 
               'k_2_0_fbs1': 'k_snow = 2.0, m_snow = 0',
               'k_0_1_id': 'k = 0.1, idealized forcing', 'k_0_2_id': 'k = 0.2, idealized forcing', 'k_0_5_id': 'k = 0.5, idealized forcing', 
               'k_1_0_id': 'k = 1.0, idealized forcing', 'k_2_0_id': 'k = 2.0, idealized forcing',
               'oh0_nosnow': 'oh = 0 W', 'oh0_5_nosnow': 'oh = 0.5 W', 'oh1_nosnow': 'oh = 1 W', 'oh10_nosnow': 'oh = 10 W', 
               'oh20_nosnow': 'oh = 20 W', 'oh50_nosnow': 'oh = 50 W', 'oh100_nosnow': 'oh = 100 W', 
               'oh0_5': '0.5 W', 'oh1': '1 W', 'oh10': '10 W', 'oh20': '20 W', 
               'oh1_idf_nosnow': '1 W', 'oh10_idf_nosnow': '10 W', 'oh20_idf_nosnow': '20 W', 
               'const_T_neg5C': '-5°C', 'const_T_neg10C': '-10°C', 'const_T_neg15C': '-15°C', 'const_T_neg20C': '-20°C'}


k_values = {'ksnow_0_005_fbs0': 0.005, 'ksnow_0_01_fbs0': 0.01, 'ksnow_0_05_fbs0': 0.05, 'ksnow_0_1_fbs0': 0.1, 'ksnow_0_175_fbs0': 0.175, 
               'ksnow_0_25_fbs0': 0.25, 'ksnow_0_5_fbs0': 0.5, 'ksnow_1_0_fbs0': 1.0, 'ksnow_1_5_fbs0': 1.5, 
               'ksnow_0_01_fbs1': 0.01, 'ksnow_0_1_fbs1': 0.1, 'ksnow_0_15_fbs1': 0.15, 'ksnow_0_2_fbs1': 0.2, 'ksnow_0_3_fbs1': 0.3, 'ksnow_0_4_fbs1': 0.4, 'ksnow_0_5_fbs1': 0.5, 
               'ksnow_1_0_fbs1': 1.0, 'ksnow_1_5_fbs1': 1.5, 'ksnow_2_0_fbs1': 2.0, 
               'ksnow_0_1_idf': 0.1, 'ksnow_0_2_idf': 0.2, 'ksnow_0_5_idf': 0.5, 
               'ksnow_1_0_idf': 1.0, 'ksnow_2_0_idf': 2.0}

oh_values = {'oh0_nosnow': 0, 'oh0_5_nosnow': 0.5, 'oh1_nosnow': 1, 'oh10_nosnow': 10, 
             'oh20_nosnow': 20, 'oh50_nosnow': 50, 'oh100_nosnow': 100,}
#oh_plotlabels = {'oh0_nosnow': 'oh = 0 W', 'oh0_5_nosnow': 'oh = 0.5 W', 'oh1_nosnow': 'oh = 1 W', 'oh10_nosnow': 'oh = 10 W', 
#                 'oh20_nosnow': 'oh = 20 W', 'oh50_nosnow': 'oh = 50 W', 'oh100_nosnow': 'oh = 100 W', 
#                 'oh0_5': '0.5 W', 'oh1': '1 W', 'oh10': '10 W', 'oh20': '20 W', 
#                 'oh1_idf_nosnow': '1 W', 'oh10_idf_nosnow': '10 W', 'oh20_idf_nosnow': '20 W'}

dat = {}
for run in runs:
    dat[run] = load_data_grid(paths[run], free_flag = 1)
    if run in k_values.keys():
        dat[run]['k'] = k_values[run]
    else: 
        dat[run]['k'] = np.nan
    if run in plot_labels.keys():
        dat[run]['label'] = plot_labels[run]
#    elif run in oh_plotlabels:
#        dat[run]['label'] = oh_plotlabels[run]

# Load config file
with open(paths_ksnow_fbs1['ksnow_1_0_fbs1'] + '/config.json') as json_file:
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

''' #snow thicknesses, not relevant cause forcing goes via heat conductivity
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1: ax.plot(dat[run]['snow'][:,0], #label = plot_labels[run],
                              color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('snow thickness [m]')
ax.set_title('Snow thicknesses for different snow precipitations')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()
'''

def plot_ice_thickness_kcolorbar(dat, runs, description):
    colormap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin = min(k_values.values()), vmax = max(k_values.values()))
    sm = matplotlib.cm.ScalarMappable(norm = norm, cmap = colormap)
    fig, ax = plt.subplots(figsize = (10,6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    for run in runs: ax.plot(dat[run]['vital_signs'][:,3],# label = plot_labels[run],
                                            color = colormap(norm(dat[run]['k'])))
    ax.invert_yaxis()
    ax.set_xlabel('time [day]')
    ax.set_ylabel('total ice thickness [m]')
    ax.set_title('Total Ice Thickness for' + description)
    plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
    return fig


def plot_ice_thickness(dat, runs, description):
    fig, ax = plt.subplots(figsize = (10,6))
    for run in runs:
        ax.plot(dat[run]['vital_signs'][:,3], label = dat[run]['label'])
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlabel('time [day]')
    ax.set_ylabel('total ice thickness [m]')
    ax.set_title('Total Ice Thickness for ' + description)
    plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')

fig_ice_thick_ksnow_fbs1 = plot_ice_thickness_kcolorbar(dat, runs_ksnow_fbs1, 'snow heat conductivity, smassless snow')

fig_ice_thick_ohnosnow = plot_ice_thickness(dat, runs_oh_nosnow, 'oceanic heatflux forcing, no snow')

#fig_ksnow_fbs1 = plot_ice_thickness_kcolorbar(dat, runs_ksnow_fbs1)

#%% plot of ice thickness
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1: ax.plot(dat[run]['vital_signs'][:,3],# label = plot_labels[run],
                                        color = colormap(norm(dat[run]['k'])))
ax.invert_yaxis()
ax.set_xlabel('time [day]')
ax.set_ylabel('total ice thickness [m]')
ax.set_title('total ice thickness')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/ice_thickness_fbs1.png')
plt.show()


#plot ice thickness for ideal forcings:
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_idf: ax.plot(dat[run]['vital_signs'][:,3],# label = plot_labels[run],
                              color = colormap(norm(dat[run]['k'])))
ax.invert_yaxis()
ax.set_xlabel('time [day]')
ax.set_ylabel('total ice thickness [m]')
ax.set_title('total ice thickness for idealized forcings')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/ice_thickness_fbs1_ideal.png')
plt.show()

fig, ax = plt.subplots(figsize = figsize)
for run in runs_oh_nosnow:
    ax.plot(dat[run]['vital_signs'][:,3], label = dat[run]['label'])
ax.invert_yaxis()
ax.legend()
ax.set_xlabel('time [day]')
ax.set_ylabel('total ice thickness [m]')
ax.set_title('total ice thickness for oceanic heatflux forcings')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/ice_thickness_fbs1_ideal.png')
plt.show()


fig, ax = plt.subplots(figsize = figsize)
for run in runs_oh_fbs1:
    ax.plot(dat[run]['vital_signs'][:,3], label = dat[run]['label'])
ax.invert_yaxis()
ax.legend()
ax.set_xlabel('time [day]')
ax.set_ylabel('total ice thickness [m]')
ax.set_title('total ice thickness for oceanic heatflux forcings')
#plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/ice_thickness_fbs1_ideal.png')
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
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1: ax.plot(S_bu_mean_1[run], #label = plot_labels[run], 
                              color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('mean salinity [psu]')
ax.set_title('Mean bulk salinities, weighted by layer thickness')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()

#%% calculate weight for mean bulk salinity
# for layers within ice: weight = layer thickness
# for ocean-ice boundary layer : weight = (psi_s(boundary)/psi_s_min) * layer thickness
#plt.legend()
'''
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
    S_bu_mean[run] = np.sum(S_ice[run] * layer_weight[run], axis = 1) / dat[run]['vital_signs'][:,3]#np.sum(layer_weight[run][i,1:])#dat[run]['vital_signs'][:,3]

    
 
plt.plot(np.sum(layer_weight[run], axis = 1),dat[run]['vital_signs'][:,3])
plt.plot(np.linspace(0,2), np.linspace(0,2))
plt.show()


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

def weighted_mean_bulk_salinity(dat):
    runs = dat.keys()
    Nactive, layer_weight, S_bu_mean = {},{},{} 
    for run in runs:
        layer_weight[run] = dat[run]['thick'][:,1:].copy()
        Nactive[run] = np.sum(dat[run]['thick'][:,1:] != 0., axis = 1) - 1 # -1 to account for python indexing
        for i in np.arange(Nactive[run].shape[0]):
            layer_weight[run][i,Nactive[run][i]] = (dat[run]['psi_s'][i,Nactive[run][i]+1] / psi_s_min) * dat[run]['thick'][i,Nactive[run][i]+1] #+1 to account for first layer being snow
        S_bu_mean[run] = np.sum(dat[run]['S'][:,1:] * layer_weight[run], axis = 1) / dat[run]['vital_signs'][:,3]#np.sum(layer_weight[run][i,1:])#dat[run]['vital_signs'][:,3]
    return S_bu_mean

S_bu_mean = weighted_mean_bulk_salinity(dat)

# plot weighted mean bulksalinity
for run in runs_ksnow_fbs1: plt.plot(S_bu_mean[run])#, label = plot_labels[run])
#plt.legend()
plt.xlabel('time [days]')
plt.ylabel('mean salinity [psu]')
plt.title('Mean bulk salinities, weighted by layer thickness')
plt.show()

# difference between corrected and not corrected weighted mean bulk salinities
for run in runs_ksnow_fbs1: plt.plot(S_bu_mean_1[run] - S_bu_mean[run])#, label = plot_labels[run])
#plt.legend()
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

def plot_sal_vs_t_kcolorbar(dat, runs, description):
    t_thick_max = {}
    for run in runs:
        t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])
        
    fig, ax = plt.subplots(figsize = (10,6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    for run in runs: ax.plot(S_bu_mean[run][:t_thick_max[run],],
                                  color = colormap(norm(dat[run]['k'])))
    ax.set_xlabel('time [days]')
    ax.set_ylabel('mean salinity [psu]')
    ax.set_title('Mean bulk salinities for ' + description)
    plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
    return fig
    
    
def plot_sal_vs_t(dat, runs, description):
    t_thick_max = {}
    for run in runs:
        t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])
        
    fig, ax = plt.subplots(figsize = (10,6))
    for run in runs: ax.plot(S_bu_mean[run][:t_thick_max[run],], label = dat[run]['label'])
    ax.legend()
    ax.set_xlabel('time [days]')
    ax.set_ylabel('mean salinity [psu]')
    ax.set_title('Mean bulk salinities for ' + description)

fig = plot_sal_vs_t_kcolorbar(dat, runs_ksnow_idf, 'heat cond. variations, idealized forcing')


fig2 = plot_sal_vs_t(dat, runs_oh_nosnow, 'oceanic heatflux forcing (nosnow)')

fig_const_T = plot_sal_vs_t(dat, runs_const_T, 'constant temperature forcing')



#%% plot weighted mean bulksalinity
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1: ax.plot(S_bu_mean[run][:t_thick_max[run],],# label = plot_labels[run],
                              color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('mean salinity [psu]')
ax.set_title('Mean bulk salinities')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/MBSovertime_fbs1.png')
plt.show()

# plot weighted mean bulksalinity for idealized forcing
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_idf: ax.plot(S_bu_mean[run][:t_thick_max[run],], #label = plot_labels[run],
                              color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('mean salinity [psu]')
ax.set_title('Mean bulk salinities with idealized forcing')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/MBSovertime_ideal.png')
plt.show()

# plot weighted mean bulksalinity for oceanic heating with no snow
fig, ax = plt.subplots(figsize = figsize)
for run in runs_oh_nosnow: ax.plot(S_bu_mean[run][:t_thick_max[run],], label = dat[run]['label'])
ax.legend()
ax.set_xlabel('time [days]')
ax.set_ylabel('mean salinity [psu]')
ax.set_title('Mean bulk salinities with oceanic heatflux forcing (nosnow)')
#plt.savefig('PLOTS/MBSovertime_ideal.png')
plt.show()


# plot weighted mean bulksalinity for oceanic heating with massless snow (fbs = 1)
fig, ax = plt.subplots(figsize = figsize)
for run in runs_oh_fbs1: ax.plot(S_bu_mean[run][:t_thick_max[run],], label = dat[run]['label'])
ax.legend()
ax.set_xlabel('time [days]')
ax.set_ylabel('mean salinity [psu]')
ax.set_title('Mean bulk salinities with oceanic heatflux forcing (fbs=1)')
#plt.savefig('PLOTS/MBSovertime_ideal.png')
plt.show()

# plot weighted mean bulksalinity for oceanic heating with idealized forcing and massless snow (fbs = 1)
fig, ax = plt.subplots(figsize = figsize)
for run in runs_oh_idf: ax.plot(S_bu_mean[run][:t_thick_max[run],], label = dat[run]['label'])
ax.legend()
ax.set_xlabel('time [days]')
ax.set_ylabel('mean salinity [psu]')
ax.set_title('Mean bulk salinities with oceanic heatflux forcing (fbs=1, and idealized forcing)')
#plt.savefig('PLOTS/MBSovertime_ideal.png')
plt.show()

#%% get mass of every layer and calculate total amount shed salt

def released_salt_mass(dat):
    runs = dat.keys()
    m, S_bu_diff, m_salt_diff, m_salt = {},{},{},{}              # mass of layer per area [kg/m²]
    for run in runs:
        m[run]              = ((rho_s * dat[run]['psi_s'] + rho_l * dat[run]['psi_l']) * dat[run]['thick'])#[:t_thick_max[run],:]
        S_1m                = dat[run]['S']#[:t_thick_max[run],:]    
        S_bu_diff[run]      = np.full_like(S_1m, 34) - S_1m
        m_salt_diff[run]    = np.diff(m[run] * S_bu_diff[run], axis = 1)
        m_salt[run]         = np.sum(m_salt_diff[run], axis = 1)
    return m_salt

m_salt = released_salt_mass(dat)

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1: ax.plot(m_salt[run],# label = plot_labels[run],
                                   color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('time [days]')
ax.set_ylabel('mass of salt [g/m²]')
ax.set_title('Accumulated total mass of shed salt until d_ice = 1m')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1: ax.plot(dat[run]['vital_signs'][:,3],m_salt[run],
                                   color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mass of salt [g/m²]')
ax.set_title('Accumulated total mass of shed salt until d_ice = 1m')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()


#%% plot mean bulk salinity against relative thickness:
t_plot = 75

def plot_sal_vs_thickness_kcolobar(dat, mean_salinity, runs, description):
    t_thick_max = {}
    for run in runs:
        t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])
    
    colormap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin = 0.1, vmax = 2.)
    sm = matplotlib.cm.ScalarMappable(norm = norm, cmap = colormap)
    
    fig, ax = plt.subplots(figsize = (10,6))                                          
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    for run in runs: 
        ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3],
                 mean_salinity[run][:t_thick_max[run],],# label = plot_labels[run],
                 color = colormap(norm(dat[run]['k'])))
    ax.set_xlabel('ice thickness [m]')
    ax.set_ylabel('mean bulk salinity [g/kg]')
    ax.set_title('Mean bulk salinites for ' + description)
    plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
    return fig

def plot_sal_vs_thickness(dat, mean_salinity, runs, description):
    t_thick_max = {}
    for run in runs:
        t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])
    
    fig, ax = plt.subplots(figsize = (10,6))                                          
    for run in runs: 
        ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3],
                 mean_salinity[run][:t_thick_max[run],], label = dat[run]['label'])
    ax.legend()
    ax.set_xlabel('ice thickness [m]')
    ax.set_ylabel('mean bulk salinity [g/kg]')
    ax.set_title('Mean bulk salinites for ' + description)
    return fig

fig_meansal_thick_ksnow_fbs1 = plot_sal_vs_thickness_kcolobar(dat, S_bu_mean, runs_ksnow_fbs1, 'snow heat conductivity, massless snow')

fig_meansal_thick_ohnosnow = plot_sal_vs_thickness(dat, S_bu_mean, runs_oh_nosnow, 'oceanic forcing, no snow')

fig_meansal_thick_constT = plot_sal_vs_thickness(dat, S_bu_mean, runs_const_T, 'constant T-forcing')

#%%
fig, ax = plt.subplots(figsize = figsize)                                          
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1: 
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3]/dat[run]['vital_signs'][t_thick_max[run],3],
             S_bu_mean[run][:t_thick_max[run],],# label = plot_labels[run],
             color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('relative ice thickness (to max thickness)')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_idf: 
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3]/dat[run]['vital_signs'][t_thick_max[run],3],
             S_bu_mean[run][:t_thick_max[run],],# label = plot_labels[run],
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
for run in runs_ksnow_fbs1:
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3], 
             S_bu_mean[run][:t_thick_max[run],], #label = plot_labels[run],
             color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites until maximum thickness')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/MBSoverthick_fbs1.png')
plt.show()  

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_idf:
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3], 
             S_bu_mean[run][:t_thick_max[run],], #label = plot_labels[run],
             color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites until maximum thickness for idealized forcings')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/MBSoverthick_ideal.png')
plt.show()  




fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
#cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_oh_nosnow:
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3], 
             S_bu_mean[run][:t_thick_max[run],])#, label = plot_labels[run])
             #color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites oceanic heatflux forcing, no snow')
#plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/MBSoverthick_ideal.png')
plt.show() 


fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
#cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_oh_fbs1:
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3], 
             S_bu_mean[run][:t_thick_max[run],])#, label = plot_labels[run])
             #color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites oceanic heatflux forcing, massless snow (fbs = 1)')
#plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/MBSoverthick_ideal.png')
plt.show()   


fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
#cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_oh_idf:
    ax.plot(dat[run]['vital_signs'][:t_thick_max[run],3], 
             S_bu_mean[run][:t_thick_max[run],])#, label = plot_labels[run])
             #color = colormap(norm(dat[run]['k'])))
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites oceanic heatflux forcing, massless snow (fbs = 1), idealized forcing')
#plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
#plt.savefig('PLOTS/MBSoverthick_ideal.png')
plt.show()  
    
#%% plot mean bulk salinity against absolute thickness:
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05) 
for run in runs_ksnow_fbs1:
    ax.plot(dat[run]['vital_signs'][:t_plot,3], 
             S_bu_mean[run][:t_plot,], #label = plot_labels[run], 
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
for run in runs_ksnow_idf:
    ax.plot(dat[run]['vital_signs'][:250,3], 
             S_bu_mean[run][:250,], #label = plot_labels[run], 
             color = colormap(norm(dat[run]['k'])))
#plt.legend()
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites for idealized forcing')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()

#%% fitting a parameterisation S(thickness):

t_fit = 75
t_plot = 75    

def exp(x,a,b,c):
    f = a * np.exp(-b * x) + c
    return f

xdata = np.empty((0,))
for run in runs_ksnow_fbs1: xdata = np.append(xdata, dat[run]['vital_signs'][:t_fit,3][2:])
ydata = np.empty((0,))
for run in runs_ksnow_fbs1: ydata = np.append(ydata, S_bu_mean[run][:t_fit,][2:])

X = np.linspace(0.05,1.05,50)

popt, pcov = curve_fit(exp, xdata, ydata)

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1:
    ax.plot(dat[run]['vital_signs'][:t_plot,3], 
             S_bu_mean[run][:t_plot,],# - popt[2], #label = plot_labels[run], 
             color = colormap(norm(dat[run]['k'])))
ax.plot(X, exp(X, *popt),# - popt[2], 
         color = 'red', linestyle = '--', 
         label = 'exponential fit: f(x) = ' + str(round(popt[0],1)) + 
                 '*exp(-' + str(round(popt[1],1)) + '*x) + ' + str(round(popt[2],1)))
ax.legend()
#ax.set_yscale('log')
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites until mid-November')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/MBSoverthick_fbs1_midNov_fit.png')
plt.show()

#%% fitting parameterization on idealized forcings data:
#xdata = np.empty(dat['k_0_1_id']['vital_signs'][:t_plot,3][2:].shape)
#for run in runs_ksnow_fbs1: xdata = np.append(xdata, dat[run]['vital_signs'][:t_plot,3][2:])
#ydata = np.empty(S_bu_mean['k_0_1_fbs1'][:t_plot,][2:].shape)
#for run in runs_ksnow_fbs1: ydata = np.append(ydata, S_bu_mean[run][:t_plot,][2:])
#runs_fbs1


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
    dat_input[i] = np.loadtxt(paths_ksnow_fbs1['ksnow_1_0_fbs1'] + '/input/' + i + '.txt')[int(t_del):]

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
ax.set_xlabel('time [min]')
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
ax.set_xlabel('time [min]')
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
ax.set_xlabel('time [min]')
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

    
for run in runs_ksnow_fbs1:    
    plt.plot(dat[run]['S'][t_thick_max[run],1:],
             dat[run]['depth'][t_thick_max[run],1:]/(dat[run]['vital_signs'][t_thick_max[run],3]))#, 
             #label = plot_labels[run] + ' at ' + str(date_max_thick[run]))
#plt.legend()
plt.xlabel('Salinity [g/kg]')
plt.ylabel('depth realtive to maximal ice thickness')
plt.title('Salinity profiles at point of maximal ice thickness')
plt.show()

for run in runs_ksnow_fbs1:    
    plt.plot(dat[run]['T'][t_thick_max[run],1:],
             dat[run]['depth'][t_thick_max[run],1:]/-dat[run]['depth'][t_thick_max[run],-1])#, 
             #label = plot_labels[run])
#plt.legend()
plt.xlabel('temperature [°C]')
plt.ylabel('depth realtive to maximal ice thickness')
plt.title('Temperature profiles at point of maximal ice thickness')
plt.show()


for run in runs_ksnow_fbs1:    
    plt.plot(dat[run]['S'][228,1:],
             #dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             -dat[run]['depth'][228,1:]/dat[run]['depth'][228,-1])#, 
             #label = plot_labels[run])
#plt.legend()
plt.xlabel('Salinity [g/kg]')
plt.ylabel('depth realtive to maximal ice thickness')
plt.title('Salinity profiles in mid-april')
plt.show()



#%% profiles at same thickness:
    
t_90cm = {}
for run in runs_ksnow_fbs1:
    t_90cm[run] = np.where(dat[run]['vital_signs'][:,3] >= 0.9)[0][0]


fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1:
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
for run in runs_ksnow_idf:
    t_90cm_id[run] = np.where(dat[run]['vital_signs'][:,3] >= 0.9)[0][0]
    
    
fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_idf:
    ax.plot(dat[run]['S'][t_90cm_id[run],1:],
             #dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             dat[run]['depth'][t_90cm_id[run],1:],#/dat[run]['depth'][t_90cm_id[run],-1], 
             color = colormap(norm(k_values[run])),
             label = 'at ' + str(time[t_90cm_id[run]])[:-9])
#ax.legend()
ax.set_ylim(-0.9,0.13)
for run in runs_ksnow_idf:
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
for run in runs_ksnow_fbs1:  
    ax.plot(dat[run]['S'][t_90cm[run],1:],
             #dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             dat[run]['depth'][t_90cm[run],1:],#/dat[run]['depth'][t_1m[run],-1], 
             color = colormap(norm(k_values[run])),
             label = 'at ' + str(time[t_90cm[run]])[:-9])
ax.legend()
ax.set_xlim(4.25,10.5)
ax.set_ylim(-0.85,0.15)
for run in runs_ksnow_fbs1:
    ax.vlines(S_bu_mean[run][t_90cm[run]], -1,0.2,color = colormap(norm(k_values[run])), linestyle = 'dashed', alpha = 0.8)
ax.set_xlabel('Salinity [g/kg]')
ax.set_ylabel('depth realtive to maximal ice thickness')
ax.set_title('Salinity profiles at 0.9m ice thickness, with mean bulk Salinity (dashed)')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.savefig('PLOTS/salproflie_fbs1_90cm.png')
plt.show()


#%% salinity profiles with mean bulk salinities at the smae time before warming event
t_plot = 75

fig, ax = plt.subplots(figsize = figsize)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
for run in runs_ksnow_fbs1:    
    ax.plot(dat[run]['S'][t_plot,1:],
             #dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             dat[run]['depth'][t_plot,1:],#/dat[run]['depth'][t_1m[run],-1], 
             color = colormap(norm(k_values[run])))
             #label = 'at ' + str(time[t_plot])[:-9])
#ax.legend()
ax.set_xlim(4.25,13.5)
ax.set_ylim(-1,0.15)
for run in runs_ksnow_fbs1:
    ax.vlines(S_bu_mean[run][t_plot], -1,0.2,color = colormap(norm(k_values[run])), linestyle = 'dashed', alpha = 0.8)
ax.set_xlabel('Salinity [g/kg]')
ax.set_ylabel('depth realtive to maximal ice thickness')
ax.set_title('Salinity profiles at ' + str(time[t_plot])[:-9] + ', with mean bulk Salinity (dashed)')
plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
plt.show()


#%% calculate mean bulk salinities at t_90cm:

#S_mean_90cm = {}
#for run in runs_ksnow_fbs1:
#    S_mean_90cm[run] = np.sum(dat[run]['S'][t_90cm[run],1:] * layer_weight[run][t_90cm[run],:]) / dat[run]['vital_signs'][t_90cm[run],3]

#S_mean_90cm


















