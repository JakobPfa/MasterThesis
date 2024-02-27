##### defining some functions for SAMSIM data analysis

import numpy
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable


#%% load data
#function to load data and simulatiously construct Xgrid and depth to plot adaptable SAMSIM grid #####
# requires path of run, with output folder located in said path 
# returns dictionary to put into bigger dictionary of runs
def load_data_grid(run_path, free_flag):
    data_grid = {}
    S          = numpy.loadtxt(run_path + "/output/dat_S_bu.dat")
    T          = numpy.loadtxt(run_path + "/output/dat_T.dat")
    psi_l      = numpy.loadtxt(run_path + "/output/dat_psi_l.dat")  #volume fraction of liquid
    psi_s      = numpy.loadtxt(run_path + "/output/dat_psi_s.dat")  #volume fraction of liquid    
    thick      = numpy.loadtxt(run_path + "/output/dat_thick.dat")
    freeboard  = numpy.loadtxt(run_path + "/output/dat_freeboard.dat")
    snow       = numpy.loadtxt(run_path + "/output/dat_snow.dat")
    vital_signs= numpy.loadtxt(run_path + "/output/dat_vital_signs.dat")
    T2m_T_top  = numpy.loadtxt(run_path + "/output/dat_T2m_T_top.dat")
    
    
    # Load config file
    with open(run_path + '/config.json') as json_file:
        config = json.load(json_file)
    
    # build time axis
    offset = pd.DateOffset(seconds=config['time_out'])
    time = pd.date_range(config['start_time'], freq=offset, periods=config['time_total']/config['time_out'] + 1).to_series()
    dx           = config['time_out']/(60*60*24)  # get dx in days
    timeunit     = '[days]'
    
    #Setting freeboard to zero if free_flag = 0
    if free_flag == 0:
        freeboard[:] = 0.
    
    ylen = len(thick[0,:])
    xlen = len(thick[:,0])
    
    #getting snow data
    T_snow     = snow[:,1]
    T_snow     = T_snow.reshape(xlen,1)
    psi_l_snow = snow[:,2]
    psi_l_snow = psi_l_snow.reshape(xlen,1)
    psi_s_snow = snow[:,3]
    psi_s_snow = psi_l_snow.reshape(xlen,1)
    thick_snow = snow[:,0]
    thick_snow = thick_snow.reshape(xlen,1)
    S_snow     = T_snow*0.0
    
    #adding snow data to ice data
    
    thick = numpy.hstack((thick_snow,thick))
    T     = numpy.hstack((T_snow,T))
    psi_l = numpy.hstack((psi_l_snow,psi_l))
    psi_s = numpy.hstack((psi_s_snow,psi_s))
    S     = numpy.hstack((S_snow,S))
    
    #Restructuring the data so it can be ploted by pcolor
    depth = thick*1.
    depth_contour = thick*1.
    Xgrid = thick*1.
    
    ylen = len(thick[0,:])
    xlen = len(thick[:,0])
    Xaxis = numpy.arange(0,xlen*dx,dx)
    i=0
    j=0
    ireal = 0.
    while (i<xlen):
        while (j<ylen):
            depth[i,j]=-sum(thick[i,0:j])+freeboard[i]+thick_snow[i]
            #Contour depth is slightly different
            depth_contour[i,j]=-sum(thick[i,0:j])-thick[i,j]/2.+freeboard[i]+thick_snow[i]
            Xgrid[i,j]=ireal
            j=j+1
        i=i+1
        j=0
        ireal=ireal+dx
    
    #depth = numpy.column_stack((depth,depth[:,-1]-(depth[:,-1]-depth[:,-2])))
    #Xgrid = numpy.column_stack((Xgrid,Xgrid[:,-1]))
    
    #depth = numpy.vstack((depth, depth[-1,:]))
    #Xgrid = numpy.vstack((Xgrid, Xgrid[-1,:]))
    
    # resolve bottom layer with psi_s_min:
    #psi_s_min = 0.05
    #Nactive = np.sum(thick != 0., axis = 1) - 1
    #for i in np.arange(Nactive.shape[0]):
    #    thick[i,Nactive[i]] = thick[i,Nactive[i]] * psi_s[i,Nactive[i]] /psi_s_min
    #    depth[i,Nactive[i]] = depth[i,Nactive[i]] * psi_s[i,Nactive[i]] /psi_s_min


    data_grid = {'T': T, 'S': S, 'psi_l': psi_l, 'psi_s': psi_s, 'thick': thick, 'freeboard': freeboard, 'snow': snow, 
                 'Xgrid': Xgrid, 'depth': depth, 'vital_signs': vital_signs, 'T2m_T_top': T2m_T_top}
    
    return data_grid


#%% Mean bulk Salinity:
# calculates the mean bulk salinity with a weighted boundary layer (psi_s_bound/psi_s_min)
# requires data dictionary with run names as keys
def weighted_mean_bulk_salinity(dat, runs):
    psi_s_min = 0.05
    runs = dat.keys()
    Nactive, layer_weight, S_bu_mean = {},{},{} 
    for run in runs:
        layer_weight[run] = dat[run]['thick'][:,1:].copy()
        Nactive[run] = np.sum(dat[run]['thick'][:,1:] != 0., axis = 1) - 1 # -1 to account for python indexing
        for i in np.arange(Nactive[run].shape[0]):
            layer_weight[run][i,Nactive[run][i]] = (dat[run]['psi_s'][i,Nactive[run][i]+1] / psi_s_min) * dat[run]['thick'][i,Nactive[run][i]+1] #+1 to account for first layer being snow
        S_bu_mean[run] = np.sum(dat[run]['S'][:,1:] * layer_weight[run], axis = 1) / dat[run]['vital_signs'][:,3]#np.sum(layer_weight[run][i,1:])#dat[run]['vital_signs'][:,3]
    return S_bu_mean


#%% shed salt:
# calculate the mass of shed salt in the freezing process
#
def released_salt_mass(dat, runs):
    rho_s = 920. 
    rho_l = 1028.
    runs = dat.keys()
    m, S_bu_diff, m_salt_timestep, m_salt = {},{},{},{}              # mass of layer per area [kg/m²]
    for run in runs:
        m[run]              = ((rho_s * dat[run]['psi_s'] + rho_l * dat[run]['psi_l']) * dat[run]['thick'])#[:t_thick_max[run],:]
        S_bu_diff[run]      = np.full_like(dat[run]['S'], 34.) - dat[run]['S']
        m_salt_timestep[run]         = np.sum(m[run] * S_bu_diff[run], axis = 1)
        m_salt[run]         = np.diff(m_salt_timestep[run])
        m_salt[run]         = np.cumsum(m_salt[run])/1000
    return m_salt


#%% plotting functions:
#ice thickness plots with and without colorbar for k

def plot_ice_thickness_kcolorbar(dat, runs, description):
    colormap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin = 0.1, vmax = 2.)
    sm = matplotlib.cm.ScalarMappable(norm = norm, cmap = colormap)
    fig, ax = plt.subplots(figsize = (10,6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    for run in runs: ax.plot(dat[run]['vital_signs'][:,3],# label = plot_labels[run],
                                            color = colormap(norm(dat[run]['k'])))
    ax.invert_yaxis()
    ax.set_xlabel('time [day]')
    ax.set_ylabel('total ice thickness [m]')
    ax.set_title('Total Ice Thickness for ' + description)
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
    return fig
    
    
# mean bulk salinity plots:
def plot_sal_vs_t_kcolorbar(dat, mean_salinity, runs, description):
    colormap = plt.cm.viridis
    norm = matplotlib.colors.Normalize(vmin = 0.1, vmax = 2.)
    sm = matplotlib.cm.ScalarMappable(norm = norm, cmap = colormap)
    t_thick_max = {}
    for run in runs:
        t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])
        
    fig, ax = plt.subplots(figsize = (10,6))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    for run in runs: ax.plot(mean_salinity[run][:t_thick_max[run],],
                                  color = colormap(norm(dat[run]['k'])))
    ax.set_xlabel('time [days]')
    ax.set_ylabel('mean salinity [psu]')
    ax.set_title('Mean bulk salinities for ' + description)
    plt.colorbar(mappable = sm, cax = cax, label = 'snow heat conductivity [Wm⁻¹K⁻¹]')
    return fig
    
    
def plot_sal_vs_t(dat, mean_salinity, runs, description):
    t_thick_max = {}
    for run in runs:
        t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])
        
    fig, ax = plt.subplots(figsize = (10,6))
    for run in runs: ax.plot(mean_salinity[run][:t_thick_max[run],], label = dat[run]['label'])
    ax.legend()
    ax.set_xlabel('time [days]')
    ax.set_ylabel('mean salinity [psu]')
    ax.set_title('Mean bulk salinities for ' + description)


#mean bulk salinities against ice thickness:
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

def plot_sal_vs_thickness(dat, mean_salinity, runs, description, t_plot = 'max'):
    if t_plot == 'max':
        T_plot = {}
        for run in runs:
            T_plot[run] = np.argmax(dat[run]['vital_signs'][:,3])
    else:
        T_plot = {}
        for run in runs:
            T_plot[run] = t_plot
    
    fig, ax = plt.subplots(figsize = (10,6))                                          
    for run in runs: 
        ax.plot(dat[run]['vital_signs'][:T_plot[run],3],
                 mean_salinity[run][:T_plot[run],], label = dat[run]['label'])
    ax.legend()
    ax.set_xlabel('ice thickness [m]')
    ax.set_ylabel('mean bulk salinity [g/kg]')
    ax.set_title('Mean bulk salinites for ' + description)
    return fig, ax


def plot_absolute_mass_salt_vs_t(dat, salt, runs, description, t_plot = 'max'):
    if t_plot == 'max':
        T_plot = {}
        for run in runs:
            T_plot[run] = np.argmax(dat[run]['vital_signs'][:,3])
    else:
        T_plot = {}
        for run in runs:
            T_plot[run] = t_plot
            
    fig, ax = plt.subplots(figsize = (10,6))
    for run in runs:
        ax.plot(salt[run][:T_plot[run],], label = dat[run]['label'])
    ax.legend()
    ax.set_xlabel('time [days]')
    ax.set_ylabel('total expelled mass of salt [kg]')
    ax.set_title('expelled mass of salt for ' + description)
    return fig, ax

def plot_absolute_mass_salt_vs_thickness(dat, salt, runs, description, t_plot = 'max'):
    if t_plot == 'max':
        T_plot = {}
        for run in runs:
            T_plot[run] = np.argmax(dat[run]['vital_signs'][:,3])
    else:
        T_plot = {}
        for run in runs:
            T_plot[run] = t_plot
            
    fig, ax = plt.subplots(figsize = (10,6))
    for run in runs:
        ax.plot(dat[run]['vital_signs'][:T_plot[run],3],
                salt[run][:T_plot[run],], label = dat[run]['label'])
    ax.legend()
    ax.set_xlabel('ice thickness [m]')
    ax.set_ylabel('total expelled mass of salt [kg]')
    ax.set_title('expelled mass of salt for ' + description)
    return fig, ax





