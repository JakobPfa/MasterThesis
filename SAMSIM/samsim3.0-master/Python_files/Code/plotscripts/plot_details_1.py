#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:20:13 2022

@author: jakobp
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


outputpath = 'output_10to5'

S          = np.loadtxt("../../../" + outputpath + "/dat_S_bu.dat")
T          = np.loadtxt("../../../" + outputpath + "/dat_T.dat")
psi_l      = np.loadtxt("../../../" + outputpath + "/dat_psi_l.dat")
thick      = np.loadtxt("../../../" + outputpath + "/dat_thick.dat")
freeboard  = np.loadtxt("../../../" + outputpath + "/dat_freeboard.dat")
snow       = np.loadtxt("../../../" + outputpath + "/dat_snow.dat")

##subset:
x1 = 0#45
x2 = 90#85
y1 = 0
y2 = S.shape[1]#int(S.shape[1]-S.shape[1]/5)

S_sub = np.transpose(S[x1:x2,y1:y2])
T_sub = np.transpose(T[x1:x2,y1:y2])
psi_l_sub = np.transpose(psi_l[x1:x2,y1:y2])

plt.imshow(S_sub, aspect = .25)
plt.colorbar() 
plt.contour(S_sub, [10,20,30], colors = 'black')
plt.title('salinity S')
plt.show()


plt.imshow(T_sub, aspect = .25)
plt.colorbar()
plt.contour(T_sub, [-10,-8, -6, -4, -2, 0], colors = 'black')
plt.title('temperature T')
plt.show()

plt.imshow(psi_l_sub, aspect = 0.25)
plt.colorbar()
plt.contour(psi_l_sub, [0,0.1,0.2], colors = 'black')
plt.title('liquid volume fraction')
plt.show()


################## first draft on salinity calculations ###################
grad_T_t = np.gradient(T, axis = 0)
grad_T_z = np.gradient(T, axis = 1)
gradgrad_T_z = np.gradient(grad_T_z, axis = 1)


########################################
#Tx = xr.DataArray(np.transpose(T))
#xr.plot.imshow(Tx)
#plt.show()
#dTdt = xr.DataArray(grad_T_t)
########################################

grad_T_t_norm = (grad_T_t - np.mean(grad_T_t)) / np.std(grad_T_t)


plt.imshow(np.transpose(grad_T_t_norm), aspect = 0.25, vmin = -0.5, vmax = 1)
plt.colorbar()
plt.title('dT/dt')
plt.show()

plt.imshow(np.transpose(grad_T_z), aspect = 0.25, vmin = -0.3, vmax = 0.15)
plt.title('dT/dz')
plt.colorbar()
plt.show()

plt.imshow(np.transpose(gradgrad_T_z), aspect = 0.25, vmin = -1e-3, vmax = 1e-3)
plt.title('d²T/dz²')
plt.colorbar()
plt.show()

kappa = np.empty_like(grad_T_t)

for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        if gradgrad_T_z[i,j] == 0:
            kappa[i,j] = 1e-10
        if gradgrad_T_z[i,j] != 0:
            kappa[i,j] = grad_T_t[i,j] / gradgrad_T_z[i,j]
        
kappam = grad_T_t / gradgrad_T_z

plt.imshow(np.transpose(kappa), aspect = 0.25, vmin = -80, vmax = 80)
plt.title('kappa')
plt.colorbar()
plt.show()


'''
kappax = xr.DataArray(kappa)
xr.plot.imshow(kappax, robust = True)
plt.show()


plt.plot(grad_T_t, np.ones(T.shape), linestyle = 'None', marker = '.', color = 'black')
plt.title('Zahlenstrahl dT/dt')
plt.show()


plt.imshow(np.transpose(T),aspect = 0.25, vmin = -10.5, vmax = -5)
plt.colorbar()
plt.show()


plt.hist(kappa, bins = np.linspace(-4e2,4e2,41))#, linestyle = 'None', marker = '.', color = 'black')
'''


