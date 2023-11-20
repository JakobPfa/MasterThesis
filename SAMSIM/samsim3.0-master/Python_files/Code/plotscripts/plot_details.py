#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:20:13 2022

@author: jakobp
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib import rc
import xarray as xr
import pandas as pd
import json
import matplotlib
import os
import scipy.interpolate as sci
from termcolor import colored
import math

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier

def round_down(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


# Warnings:
# Contours are interpolated from the middle of each layer. This is most visible in thick layers, and in the snow layer where the contour lines only extend to the the middle.

# Settings
outputfile   = 'pic_TPhiS'
outputformat = 'png' #e.g. png, jpg, pdf
free_flag    = 1     #1: freeboard is included, 0:freeboard is not included

# set wd to the directory of the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)






outputpath = 'output_10to5'
plotpath = '../../../../../PLOTS/10to5/'

S          = np.loadtxt("../../../" + outputpath + "/dat_S_bu.dat")
T          = np.loadtxt("../../../" + outputpath + "/dat_T.dat")
psi_l      = np.loadtxt("../../../" + outputpath + "/dat_psi_l.dat")  #volume fraction of liquid
thick      = np.loadtxt("../../../" + outputpath + "/dat_thick.dat")
freeboard  = np.loadtxt("../../../" + outputpath + "/dat_freeboard.dat")
snow       = np.loadtxt("../../../" + outputpath + "/dat_snow.dat")

# Load config file
with open('../../../Run_specifics/config.json') as json_file:
    config = json.load(json_file)

# build time axis
offset = pd.DateOffset(seconds=config['time_out'])
time = pd.date_range(config['start_time'], freq=offset, periods=config['time_total']/config['time_out'] + 1).to_series()
dx           = config['time_out']/(60*60*24)  # get dx in days
timeunit     = '[days]'

#Contour levels 
#levelsT      = ([-10,-5,-3,-1])
#levelspsi    = ([0.1, 0.2])
#levelsS      = ([3., 8.])




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
thick_snow = snow[:,0]
thick_snow = thick_snow.reshape(xlen,1)
S_snow     = T_snow*0.0

#adding snow data to ice data

#thick = np.hstack((thick_snow,thick))
#T     = np.hstack((T_snow,T))
#psi_l = np.hstack((psi_l_snow,psi_l))
#S     = np.hstack((S_snow,S))


#Restructuring the data so it can be ploted by pcolor
depth = thick*1.
depth_contour = thick*1.
Xgrid = thick*1.

ylen = len(thick[0,:])
xlen = len(thick[:,0])
Xaxis = np.arange(0,xlen*dx,dx)
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
    
''' # stacking at boundaries & subsetting
depth = np.column_stack((depth,depth[:,-1]-(depth[:,-1]-depth[:,-2])))
Xgrid = np.column_stack((Xgrid,Xgrid[:,-1]))

depth = np.vstack((depth, depth[-1,:]))
Xgrid = np.vstack((Xgrid, Xgrid[-1,:]))


##subset:
x1 = 0
x2 = S.shape[0]
y1 = 0
y2 = S.shape[1]

S_sub = np.transpose(S[x1:x2,y1:y2])
T_sub = np.transpose(T[x1:x2,y1:y2])
psi_l_sub = np.transpose(psi_l[x1:x2,y1:y2])
'''

whatisup=freeboard+snow[:,0]
ymin=(depth.min()+freeboard.min())*1.08
ymax=whatisup.max()+depth.min()*-0.03

#aspect = 0.5
figsize = (5.5*2,2.5*2)
matplotlib.rcParams.update({'font.size': 16})

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, T)
#plt.axis([Xgrid.min(), Xgrid.max(), ymin, ymax])
plt.colorbar()
plt.contour(Xgrid, depth, T, [-10,-8,-6,-4,-2, 0], colors = 'black')
plt.xlabel('time [days]')
plt.ylabel('depth [m]')
plt.title('temperature  T [°C]')
plt.savefig(plotpath + "temp.png")
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, S)
plt.colorbar() 
plt.contour(Xgrid, depth, S, [1,5,10,20,30], colors = 'black')
plt.title('salinity S [ppt]')
plt.xlabel('time [days]')
plt.ylabel('depth [m]')
plt.savefig(plotpath + 'salinity.png')
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, psi_l)#, vmin = 0, vmax = 0.1)
#plt.contour(Xgrid, depth, psi_l, [0,0.1,0.2], colors = 'black')
plt.title('liquid volume fraction')
plt.xlabel('time [days]')
plt.ylabel('depth [m]')
#plt.xlim(1,3)
#plt.ylim(-0.2,0.05)
plt.colorbar()
plt.show()



##### interpolate onto regular grid with scipy.griddata #####

x = Xgrid[:,0]
ytop = round_up(np.max(depth),2)
ybot = round_down(np.min(depth),2)
ylen = int(np.ceil((ytop - ybot) / 0.005))+1
y = np.linspace(ybot, ytop, ylen)

grid_x, grid_y = np.meshgrid(x, y)

points = np.stack((Xgrid.flatten(), depth.flatten()), axis = 1)

T_int = np.transpose(sci.griddata(points, T.flatten(), xi = (grid_x, grid_y), method = 'cubic'))
thick_int = np.transpose(sci.griddata(points, thick.flatten(), xi = (grid_x, grid_y), method = 'nearest'))
psi_l_int = np.transpose(sci.griddata(points, psi_l.flatten(), xi = (grid_x, grid_y), method = 'cubic'))

grid_x = grid_x.transpose() #transpose for plotting and calculating gradients 
grid_y = grid_y.transpose()


plt.figure(figsize = figsize)
plt.pcolor(grid_x, grid_y, T_int, vmin = -10, vmax = 0)
plt.title('interpolated T')
plt.colorbar()
plt.show()

'''
thick_int = sci.griddata(points, T.flatten(), xi = (grid_x, grid_y), method = 'cubic')

plt.imshow(T_int)
plt.colorbar()  
plt.show()

plt.imshow(np.transpose(T))
plt.colorbar()
plt.show()
'''



################## first draft on salinity calculations ###################

##### manual gradient calculation by taking differences and dividing through distances #####
def gradman(T, X, Y, thick, axis):
    # fuction to manually calculate gradients in the adaptive SAMSIM grid ()
    if T.shape != X.shape or X.shape != Y.shape or Y.shape != thick.shape:
        print(colored('Warning: Shapes do not match!', 'red'))
        
    gradman = np.empty_like(T)
    xdim = T.shape[0]
    ydim = T.shape[1]
    
    if axis == 'time':
        for i in range(0,xdim):
            for j in range(0,ydim):
                # calculate differentials (mean of two differences for most points, at the boundaries distance from boundary to second (to last) point)
                if i == 0:
                    dT = T[i+1,j] - T[i,j]
                    dt = (X[i+1,j] - X[i,j]) * 24*60*60
                if i == xdim-1:
                    dT = T[i,j] - T[i-1,j]
                    dt = (X[i,j] - X[i-1,j]) * 24*60*60
                else:
                    dT = (T[i+1,j] - T[i-1,j]) 
                    dt = (X[i+1,j] - X[i-1,j]) * 24*60*60
                
                gradman[i,j] = dT/dt
            
    if axis == 'depth':
        for i in range(0,xdim):
            for j in range(0,ydim):
                # calculate differentials (mean of two differences for most points, at the boundaries distance from boundary to second (to last) point)
                if j == 0:
                    dT = T[i,j+1] - T[i,j]
                    #dz = Y[i,j+1] - Y[i,j]
                    dz = (thick[i,j+1] + thick[i,j])/2
                    #dz = thick[i,j]
                if j == ydim-1:
                    dT = T[i,j] - T[i,j-1]
                    #dz = Y[i,j] - Y[i,j-1]
                    dz = (thick[i,j] + thick[i,j-1])/2
                    #dz = thick[i,j]
                else:
                    dT = (T[i,j+1] - T[i,j-1])
                    #dz = (Y[i,j+1] - Y[i,j-1])
                    dz = (thick[i,j+1] + thick[i,j] + thick[i,j-1]) / 3
                    #dz = thick[i,j]
                
                gradman[i,j] = dT/dz
    return gradman
                    
# manually calculated gradient using function gradman, using the mean of two adjacent differentials                 
dTdt = gradman(T, Xgrid, depth, thick, axis = 'time')
dTdz = gradman(T, Xgrid, depth, thick, axis = 'depth')
d2Tdz2 = gradman(dTdz, Xgrid, depth, thick, axis = 'depth')

# manually calculated gradients from interpolated T field
dTdt_int = gradman(T_int, grid_x, grid_y, thick_int, axis = 'time')
dTdz_int = gradman(T_int, grid_x, grid_y, thick_int, axis = 'depth')
d2Tdz2_int = gradman(dTdz_int, grid_x, grid_y, thick_int, axis = 'depth')


'''
plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, d2Tdz2_int, vmin = -5e2, vmax = 5e2)
plt.title('manually calculated d²T/dz² on interpolated T')
plt.ylim(-0.02, 0.02)
plt.colorbar()
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, T_int, vmin = -10, vmax = 0)
plt.title('T_int')
plt.ylim(-0.02,0.02)
plt.colorbar()
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, thick_int)
#plt.pcolormesh(Xgrid, depth, thick)
plt.title('thick_int')
plt.ylim(-0.02, 0.02)
plt.colorbar()
plt.show()
''' 



# using np.gradient to calculate gradients, devide through time and depth steps respectively
grad_T_t = np.gradient(T, axis = 1)/config['time_out'] # /output zeitschritt
grad_T_z = np.gradient(T, axis = 0)/(thick)
grad2_T_z = np.gradient(grad_T_z, axis = 0)/(thick)       

# gradients from interpolated T-field
grad_T_t_int = np.gradient(T_int, axis = 1)/config['time_out']
grad_T_z_int = np.gradient(T_int, axis = 0)/thick_int
grad2_T_z_int = np.gradient(grad_T_z_int, axis = 0)/thick_int


'''##### manual calculations of gradients in between of gridpoints #####                       
#manually calculate gradients between gridpoints
for i in range(0,xdim):
    for j in range(0,ydim):
        dT = (T[i+1,j] - T[i,j])
        dt = (Xgrid[i+1,j] - Xgrid[i,j]) * 24*60*60 # time between outputs in seconds
        dTdt[i,j] = dT/dt
        
for i in range(0,xdim):
    for j in range(0,ydim):
        dT = (T[i,j+1] - T[i,j])
        dz = thick[i,j] #(depth[i,j-1] - depth[i,j+1])/2 # other way round would be negative
        dTdz[i,j] = dT/dz
        
        if i<xdim-2 and j<ydim-2:
            d2T = dTdz[i,j] - dTdz[i,j+1]
            d2Tdz2[i,j] = d2T/dz
'''            

##### plotting different gradient calculations #####
plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, dTdt, vmin = -1e-4, vmax = 1e-4)
plt.title('manual dT/dt')
plt.colorbar()
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, grad_T_t, vmin = -1e-4, vmax = 1e-4)
plt.colorbar()
plt.title('numpy gradient dT/dt')
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, dTdt_int, vmin = -5e-5, vmax = 5e-5)
plt.xlabel('time t [days]')
plt.ylabel('depth [m]')
plt.title('Temporal Gradient in Temperature dT/dt [°C/s]')
plt.colorbar()
plt.savefig(plotpath + 'dTdt.png')
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, grad_T_t_int, vmin = -1e-4, vmax = 1e-4)
plt.title('interpolated, numpy gradient dT/dt')
plt.colorbar()
plt.show()


diff_intdt = grad_T_t_int - dTdt_int

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, diff_intdt)#, vmin = -5e-5, vmax = 5e-5)
plt.title('difference btw interpolated, numpy & manual gradient dT/dt')
plt.colorbar()
plt.show()



plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, d2Tdz2, vmin = -1e2, vmax = 1e2)
plt.title('manually calculated d²T/dz²')
plt.colorbar()
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, grad2_T_z, vmin = -1e2, vmax = 1e2)
plt.title('numpy gradient d²T/dz²')
plt.colorbar()
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, d2Tdz2_int, vmin = -2e2, vmax = 2e2)
plt.xlabel('time t [days]')
plt.ylabel('depth [m]')
plt.title('Second Vertical Temperature Gradient d²T/dz² [°C²/m²]')
plt.colorbar()
plt.savefig(plotpath +  'd2Tdz2.png')
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, dTdz_int, vmin = -40, vmax = 40)#, cmap = 'seismic')
plt.xlabel('time t [days]')
plt.ylabel('depth [m]')
plt.title('Vertical Temperature Gradient dT/dz [°C/m]')
plt.colorbar()
plt.savefig(plotpath + 'dTdz.png')
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, grad2_T_z_int, vmin = -1e2, vmax = 1e2)
plt.title('numpy gradient d²T/dz² on interpolated T')
plt.colorbar()
plt.show()


'''#unnecessary plotting #
plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, d2Tdz2)#, vmin = -100, vmax = 100)
plt.title('manually calculated d²T/dz²')
plt.colorbar()
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, grad2_T_z_int)
plt.title('numpy gradient d²T/dz² on interpolated T')
plt.colorbar()
plt.show()


# plot numpy gradients
plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, grad_T_t)#, vmin = -2e-4, vmax = 2e-4)
plt.colorbar()
plt.title('dT/dt using np.gradient')
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, grad_T_z)#, vmin = -15, vmax = 15)
plt.title('dT/dz using np.gradient')
plt.colorbar()
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, grad2_T_z)#, vmin = -1e2, vmax = 1e2)
plt.title('d²T/dz² using np.gradient')
plt.colorbar()
plt.show()
'''

'''#plot differences #


plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, diff_dTdt)
plt.colorbar()
plt.title('difference between non- and interpolated np.gradient dT/dt')
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, diff_dTdz)#, vmin = -30, vmax = 30)
plt.colorbar()
plt.title('difference between manually calculated dT/dz and np.gradient')
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, diff_d2Tdz2)
plt.colorbar()
plt.title('difference between manually calculated d2T/dz2 and np.gradient')
plt.show() 
'''
cmap = matplotlib.cm.get_cmap()
cmap.set_bad(color = 'white')

#kappa and salinity calculations
### calculating heat diffusivity kappa and salinity S from temperature fields

def kappa(T, grad_t, grad2_z):
    kappa = np.empty_like(grad_t)
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if grad2_z[i,j] == 0:
                kappa[i,j] = float('nan')
            if grad2_z[i,j] != 0:
                kappa[i,j] = grad_t[i,j] / grad2_z[i,j]
    return kappa

kappa_grad = kappa(T, grad_T_t, grad2_T_z)

kappa_grad_int = kappa(T_int, grad_T_t_int, grad2_T_z_int)

kappa_man_int = kappa(T_int, dTdt_int, d2Tdz2_int)

'''
plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, kappa_grad_int, vmin = 0, vmax = 1e-3, cmap = cmap)
plt.title('kappa from numpy gradients on interpolated T')
plt.colorbar()
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, kappa_grad, vmin = 0, vmax = 1e-3, cmap = cmap)
plt.title('kappa from numpy gradients on original T')
plt.colorbar()
plt.show()
''' 

plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, kappa_man_int, vmin = -5e-7, vmax = 5e-7, cmap = cmap)
plt.title('thermal diffusivity \u03BA [m²/s]')
plt.xlabel('time t [days]')
plt.ylabel('depth d [m]')
plt.colorbar()
plt.savefig(plotpath + 'thermaldiff.png')
plt.show()


'''# manually calculated kappa  
kappa_man = np.empty_like(dTdt)

for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        if d2Tdz2[i,j] == 0:
            kappa_man[i,j] = float('nan')
        else:
            kappa_man[i,j] = dTdt[i,j] / d2Tdz2[i,j]
    
plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, kappa_man, vmin = 0, vmax = 1e-5)
plt.title('kappa from manual gradients')
plt.colorbar()
plt.show()
'''


##### kappa -> S #####

def sal(T, psi, kappa):
    Sbr =  -1.2 - 21.8*T - 0.919*T**2 - 0.0178*T**3
    ks = 2.21 - 1e-2*T + 3.44*1e-5*T**2
    kl = 0.52325*(1-Sbr/(1e3)) + 0.01256*T + 5.8604*1e-5*T**2
    
    cs = 2020.0 + 7.6973*T
    L0 = 333.5
    alpha = 0.05411
    rhos = 920.0
    rhol = 1028.0
    rhobar = rhol*psi + rhos*(1-psi)
    
    S = (Sbr*(ks - kappa * rhobar *  cs)) / (kappa * rhobar * L0 * Sbr* alpha / (T)**2 + ks + kl)
    return S

Sbu = sal(T, psi_l, kappa_grad)

Sbu_int = sal(T_int, psi_l_int, kappa_grad_int)

Sbu_man_int = sal(T_int, psi_l_int, kappa_man_int)


'''
plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, Sbu_int, vmin = 0, vmax = 100)
plt.title('bulk salinity from interpolated T with numpy gradients')
plt.colorbar()
plt.show()

plt.figure(figsize = figsize)
plt.pcolormesh(Xgrid, depth, Sbu, vmin = 0, vmax = 100)
plt.title('bulk salinity from original T with numpy gradients')
plt.colorbar()
plt.show()
''' 
                                                                              
plt.figure(figsize = figsize)
plt.pcolormesh(grid_x, grid_y, Sbu_man_int, vmin = 0, vmax = 1e2)
plt.title('bulk salinity S [ppt]')
plt.xlabel('time t [days]')
plt.ylabel('depth [m]')
plt.colorbar()
plt.savefig(plotpath + 'bulksalinity.png')
plt.show()
