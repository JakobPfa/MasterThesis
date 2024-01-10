##### defining some functions for SAMSIM data analysis

import numpy
import numpy as np
import os
import json
import pandas as pd


##### function to load data and simulatiously construct Xgrid and depth to plot adaptable SAMSIM grid #####
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