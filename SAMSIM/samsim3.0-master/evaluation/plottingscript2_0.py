#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:21:28 2024

@author: jakobp
"""

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
import functions as fun


#%% constants from model
rho_l               = 920.      # density of liquid [kg/m³]
rho_s               = 1028.     # density of solid ice [kg/m³]
rho_snow            = 330.      # density of (new) snow [kg/m³]
psi_s_min           = 0.05      # minimal solid fraction for a layer to count as sea ice



#%% routine to collect data: 
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

# oceanic heating forcing with massless snow (fbs = 1) and idealized forcing
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

k_values = {'ksnow_0_005_fbs0': 0.005, 'ksnow_0_01_fbs0': 0.01, 'ksnow_0_05_fbs0': 0.05, 'ksnow_0_1_fbs0': 0.1, 'ksnow_0_175_fbs0': 0.175, 
               'ksnow_0_25_fbs0': 0.25, 'ksnow_0_5_fbs0': 0.5, 'ksnow_1_0_fbs0': 1.0, 'ksnow_1_5_fbs0': 1.5, 
               'ksnow_0_01_fbs1': 0.01, 'ksnow_0_1_fbs1': 0.1, 'ksnow_0_15_fbs1': 0.15, 'ksnow_0_2_fbs1': 0.2, 'ksnow_0_3_fbs1': 0.3, 'ksnow_0_4_fbs1': 0.4, 'ksnow_0_5_fbs1': 0.5, 
               'ksnow_1_0_fbs1': 1.0, 'ksnow_1_5_fbs1': 1.5, 'ksnow_2_0_fbs1': 2.0, 
               'ksnow_0_1_idf': 0.1, 'ksnow_0_2_idf': 0.2, 'ksnow_0_5_idf': 0.5, 
               'ksnow_1_0_idf': 1.0, 'ksnow_2_0_idf': 2.0}


dat = {}
for run in runs:
    dat[run] = fun.load_data_grid(paths[run], free_flag = 1)
    if run in k_values.keys():
        dat[run]['k'] = k_values[run]
    else: 
        dat[run]['k'] = np.nan
    dat[run]['label'] = run

#%%calculate mean bulk salinity:
S_bu_mean = fun.weighted_mean_bulk_salinity(dat, runs)

#%%released salt:
m_salt = fun.released_salt_mass(dat, runs)

#%% select specific runs:
# k_snow with normal snow fbs0:
runs_ksnow_fbs0 = (['ksnow_0_01_fbs0', 'ksnow_0_1_fbs0', 'ksnow_0_25_fbs0', 'ksnow_1_5_fbs0'])
runs_ksnow_fbs1 = (['ksnow_0_1_fbs1', 'ksnow_0_2_fbs1', 'ksnow_0_5_fbs1', 'ksnow_2_0_fbs1'])
runs_ksnow_idf = (['ksnow_0_1_idf', 'ksnow_0_2_idf', 'ksnow_0_5_idf', 'ksnow_2_0_idf'])

runs_oh_nosnow = (['oh0_nosnow', 'oh10_nosnow', 'oh50_nosnow', 'oh100_nosnow'])
runs_oh_fbs1 = (['oh1', 'oh10', 'oh20'])
runs_oh_idf = (['oh1_idf', 'oh10_idf', 'oh20_idf'])

runs_const_T = (['const_T_neg5C', 'const_T_neg10C', 'const_T_neg15C', 'const_T_neg20C'])
#%% heat conductivity experiments:
# normal snow:
fun.plot_ice_thickness(dat, runs_ksnow_fbs0, 'ksnow experiments')
#fun.plot_sal_vs_t(dat, S_bu_mean, runs_ksnow_fbs0, 'snow heat conduct. experiments')
fun.plot_sal_vs_thickness(dat, S_bu_mean, runs_ksnow_fbs0, 'ksnow experiments')

# massless snow:
fig1 = fun.plot_ice_thickness(dat, runs_ksnow_fbs1, 'ksnow experiments, massless snow')
#fun.plot_sal_vs_t(dat, S_bu_mean, runs_ksnow_fbs1, 'snow heat conduct. experiments, massless snow')
fig2, ax = fun.plot_sal_vs_thickness(dat, S_bu_mean, runs_ksnow_fbs1, 'ksnow experiments, massless snow')
fun.plot_absolute_mass_salt_vs_thickness(dat, m_salt, runs_ksnow_fbs1, 'ksnow experiments, massless snow')

# idealized atmospaheric forcing:
fun.plot_ice_thickness(dat, runs_ksnow_idf, 'ksnow experiments, idealized forcing, massless snow')
fun.plot_sal_vs_thickness(dat, S_bu_mean, runs_ksnow_idf, 'ksnow experiments, idealized forcing, massless snow')
fun.plot_absolute_mass_salt_vs_t(dat, m_salt, runs_ksnow_idf, 'ksnow experiments, idealized forcing, massless snow')
fun.plot_absolute_mass_salt_vs_thickness(dat, m_salt, runs_ksnow_idf, 'ksnow experiments, idealized forcing, massless snow')



#%% oceanic heating
# with massless snow:
fig2 = fun.plot_ice_thickness(dat, runs_oh_fbs1, 'oceanic heatflux forcing, massless snow')
fig3, ax = fun.plot_sal_vs_thickness(dat, S_bu_mean, runs_oh_fbs1, 'oceanic heatflux forcing, massless snow')
fig4, ax = fun.plot_absolute_mass_salt_vs_thickness(dat, m_salt, runs_oh_fbs1, 'oh forcing, massless snow, ideal atmosph. forcings')

#without snow at all:
fun.plot_ice_thickness(dat, runs_oh_nosnow, 'oceanic heatflux forcing, no snow')
fun.plot_sal_vs_thickness(dat, S_bu_mean, runs_oh_nosnow, 'oceanic heatflux forcing, no snow')
fun.plot_absolute_mass_salt_vs_thickness(dat, m_salt, runs_oh_nosnow, 'oh forcing, massless snow, ideal atmosph. forcings')

# with massless snow and idealized forcing:
fig5 = fun.plot_ice_thickness(dat, runs_oh_idf, 'oh forcing, massless snow, ideal atmosph. forcings')
fig6, ax  = fun.plot_sal_vs_thickness(dat, S_bu_mean, runs_oh_idf, 'oh forcing, massless snow, ideal atmosph. forcings')
fig7, ax = fun.plot_absolute_mass_salt_vs_thickness(dat, m_salt, runs_oh_idf, 'oh forcing, massless snow, ideal atmosph. forcings')

#%% constant temperature forcing:
fig8 = fun.plot_ice_thickness(dat, runs_const_T, 'constant T forcing')
fig9, ax = fun.plot_sal_vs_thickness(dat, S_bu_mean, runs_const_T, 'constant T forcing')
fig10, ax = fun.plot_absolute_mass_salt_vs_t(dat, m_salt, runs_const_T, 'constant T forcing')
fig11, ax = fun.plot_absolute_mass_salt_vs_thickness(dat, m_salt, runs_const_T, 'constant T forcing')

#%% compare idealized with MOSAiC forcing
fun.plot_ice_thickness(dat, (['ksnow_0_2_fbs1', 'ksnow_0_2_idf']), 'comparison between measured and idealized forcing')
fun.plot_sal_vs_thickness(dat, S_bu_mean, (['ksnow_0_2_fbs1', 'ksnow_0_2_idf']), 'comparison between measured and idealized forcing')

#%% ploting T_top to see if const_T experiments make sense
fig, ax = plt.subplots(figsize = (10,6))
ax.plot(dat['ksnow_0_2_fbs1']['T2m_T_top'][:,1])
ax.plot(dat['ksnow_0_2_idf']['T2m_T_top'][:,1])
ax.set_title('Temperature of top ice layer')
ax.set_xlabel('time [days]')
ax.set_ylabel('temperature [°C]')
plt.show()

fig, ax = plt.subplots(figsize = (10,6))
ax.plot(dat['ksnow_0_2_fbs1']['T2m_T_top'][:,0])
ax.plot(dat['ksnow_0_2_idf']['T2m_T_top'][:,0])
ax.set_title('Temperature at 2m above ice')
ax.set_xlabel('time [days]')
ax.set_ylabel('temperature [°C]')
plt.show()

#%%plotting 
fig, ax = plt.subplots(figsize = (10,6))
for run in runs_oh_fbs1:
    ax.plot(dat[run]['T'][:,10], label = dat[run]['label'])
ax.legend()
ax.set_title('Temperature of top ice layer')
ax.set_xlabel('time [days]')
ax.set_ylabel('temperature [°C]')
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

fig, ax = plt.subplots(figsize = (10,6))
for run in runs_ksnow_fbs1:
    ax.plot(dat[run]['vital_signs'][:t_plot,3], 
             S_bu_mean[run][:t_plot,],label = dat[run]['label'])
ax.plot(X, exp(X, *popt),# - popt[2], 
         color = 'red', linestyle = '--', 
         label = 'exponential fit: f(x) = ' + str(round(popt[0],1)) + 
                 '*exp(-' + str(round(popt[1],1)) + '*x) + ' + str(round(popt[2],1)))
ax.legend()
#ax.set_yscale('log')
ax.set_xlabel('ice thickness [m]')
ax.set_ylabel('mean bulk salinity [g/kg]')
ax.set_title('Mean bulk salinites until mid-November')
plt.savefig('PLOTS/MBSoverthick_fbs1_midNov_fit.png')
plt.show()













