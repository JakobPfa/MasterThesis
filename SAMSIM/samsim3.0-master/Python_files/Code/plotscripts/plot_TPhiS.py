#Standard plot script for Temperature, liquid fraction, and bulk salinity
#Steps:
#1. Set output name with format to save (jpg, png, pdf, or eps). Pdf and eps look better but can be very large and slow to load.
#2. Do you want the freeboard to by incorporated into your y-axis? set free_flag
#3. Run the script! (python plot_TPhiS.py)
#4. Open the outputfile in the imageviewer of your choice.
#5. Optional: Adjust contour levels and repeat.

#Loading modules and setting fonts
import numpy
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib import rc
import matplotlib
import os
import json
import pandas as pd

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


#loading data
outputpath = 'output'

S          = numpy.loadtxt("../../../" + outputpath + "/dat_S_bu.dat")
T          = numpy.loadtxt("../../../" + outputpath + "/dat_T.dat")
psi_l      = numpy.loadtxt("../../../" + outputpath + "/dat_psi_l.dat")  #volume fraction of liquid
thick      = numpy.loadtxt("../../../" + outputpath + "/dat_thick.dat")
freeboard  = numpy.loadtxt("../../../" + outputpath + "/dat_freeboard.dat")
snow       = numpy.loadtxt("../../../" + outputpath + "/dat_snow.dat")

# Load config file
with open('../../../Run_specifics/config.json') as json_file:
    config = json.load(json_file)

# build time axis
offset = pd.DateOffset(seconds=config['time_out'])
time = pd.date_range(config['start_time'], freq=offset, periods=config['time_total']/config['time_out'] + 1).to_series()
dx           = config['time_out']/(60*60*24)  # get dx in days
timeunit     = '[days]'

#Contour levels 
levelsT      = ([-10,-5,-3,-1])
levelspsi    = ([0.1, 0.2])
levelsS      = ([3., 8.])



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

thick = numpy.hstack((thick_snow,thick))
T     = numpy.hstack((T_snow,T))
psi_l = numpy.hstack((psi_l_snow,psi_l))
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

depth = numpy.column_stack((depth,depth[:,-1]-(depth[:,-1]-depth[:,-2])))
Xgrid = numpy.column_stack((Xgrid,Xgrid[:,-1]))

depth = numpy.vstack((depth, depth[-1,:]))
Xgrid = numpy.vstack((Xgrid, Xgrid[-1,:]))


#Custom colormaps
#Liquid fraction
cdict = {'red':   ((0., 1., 1.),(0.1, 0.95 , 0.95 ),(0.3, 0.55 , 0.55 ),(1.0, 0.0, 0.0)),
         'green': ((0., 1., 1.),(0.2, 0.55, 0.55),(1.0, 0.0, 0.0)),
         'blue':  ((0., 1., 1.),(0.2, 0.55, 0.55),(1.0, 0.0, 0.0))}

psi_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

#Temperature
cdict = {'blue': [(0.0,   0.41960784792900085, 0.41960784792900085),
                  (0.25, 0.61176472902297974, 0.61176472902297974), 
                  (0.5,  0.70980393886566162, 0.70980393886566162),
                  (0.6,  0.7764706015586853,0.7764706015586853), 
                  (0.7,   0.83921569585800171, 0.83921569585800171), 
                  (0.8,    0.88235294818878174, 0.88235294818878174), 
                  (0.9,    0.93725490570068359, 0.93725490570068359),
                  (0.925,   0.9686274528503418,0.9686274528503418), 
                  (0.95,     1.0, 1.0),  
                  (1.0,     0.0, 0.0),  ],

        'green': [(0.0,   0.18823529779911041, 0.18823529779911041),
                  (0.25,    0.31764706969261169, 0.31764706969261169), 
                  (0.5,    0.44313725829124451, 0.44313725829124451), 
                  (0.6,     0.57254904508590698, 0.57254904508590698), 
                  (0.7,   0.68235296010971069, 0.68235296010971069), 
                  (0.8,     0.7921568751335144, 0.7921568751335144), 
                  (0.9,     0.85882353782653809, 0.85882353782653809), 
                  (0.925,     0.92156863212585449, 0.92156863212585449), 
                  (0.95,      0.9843137264251709, 0.9843137264251709),  
                  (1.0,      0.3                , 0.3               ),  ],

          'red': [(0.0,     0.031372550874948502, 0.031372550874948502),
                  (0.25,     0.031372550874948502, 0.031372550874948502), 
                  (0.5,     0.12941177189350128, 0.12941177189350128), 
                  (0.6,     0.25882354378700256, 0.25882354378700256), 
                  (0.7,     0.41960784792900085, 0.41960784792900085), 
                  (0.8,     0.61960786581039429, 0.61960786581039429), 
                  (0.9,     0.7764706015586853, 0.7764706015586853), 
                  (0.925,     0.87058824300765991, 0.87058824300765991),
                  (0.95,   0.9686274528503418, 0.9686274528503418), 
                  (1.0,   1.00000000000000000, 1.0000000000000000), ]}
T_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

#Salinity
cdict = {'blue': [(0.0,    1.0, 1.0), 
                  (0.1,    0.87843137979507446,0.87843137979507446), 
                  (0.15,   0.75294119119644165,0.75294119119644165), 
                  (0.2,    0.60784316062927246,0.60784316062927246), 
                  (0.25,   0.46274510025978088, 0.46274510025978088),
                  (0.3,    0.364705890417099, 0.364705890417099), 
                  (0.5,    0.27058824896812439, 0.27058824896812439), 
                  (0.8,    0.17254902422428131, 0.17254902422428131), 
                  (1.0,    0.0588235408067703,0.0588235408067703)],

        'green': [(0.0,      1.0, 1.0), 
                  (0.1,    0.96078431606292725, 0.96078431606292725), 
                  (0.15,     0.91372549533843994, 0.91372549533843994), 
                  (0.2,     0.85098040103912354, 0.85098040103912354), 
                  (0.25,      0.76862746477127075, 0.76862746477127075), 
                  (0.3,     0.67058825492858887, 0.67058825492858887), 
                  (0.5,    0.42745098471641541, 0.42745098471641541), 
                  (0.8,      0.26666668057441711, 0.26666668057441711),
                  (1.0,      0.026666668057441711, 0.026666668057441711)],

        'red':   [(0.0,    1.0, 1.0), 
                  (0.1,    0.89803922176361084, 0.89803922176361084), 
                  (0.15,    0.78039216995239258, 0.78039216995239258), 
                  (0.2,    0.63137257099151611, 0.63137257099151611), 
                  (0.25,    0.45490196347236633, 0.45490196347236633), 
                  (0.3,    0.25490197539329529, 0.25490197539329529), 
                  (0.5,    0.13725490868091583, 0.13725490868091583), 
                  (0.8,    0.0, 0.0),
                  (1.0,    0.0, 0.0)]}

S_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

#######################################################
#Plotting
#######################################################

##################################################
plt.pcolormesh(Xgrid,depth,T)
#plt.axis([Xgrid.min(), Xgrid.max(), ymin, ymax])
plt.colorbar()
plt.show()
##################################################


fig1=plt.figure(figsize=(1.25*9.,1.25*5.))
fsize=10.
whatisup=freeboard+snow[:,0]
ymin=(depth.min()+freeboard.min())*1.03
ymax=whatisup.max()+depth.min()*-0.03
plt.rcParams['contour.negative_linestyle'] = 'solid'

#Set distances between subplots and margins
fig1.subplots_adjust(hspace=0.05)
fig1.subplots_adjust(left=0.07)
fig1.subplots_adjust(right=1.08)
fig1.subplots_adjust(top=0.95)
fig1.subplots_adjust(bottom=0.08)

#######################################################
#Plotting temperature
#######################################################
ax1 = fig1.add_subplot(311)
zmin=T.min()
zmax=0
#plt.pcolor(Xgrid,depth,T,cmap='Blues_r',vmin=zmin,vmax=zmax)
plt.pcolor(Xgrid,depth,T,cmap=T_cmap,vmin=zmin,vmax=zmax)
c1 = plt.colorbar(pad=0.01)
c1.set_label(r'T')
plt.axis([Xgrid.min(), Xgrid.max(), ymin, ymax])
#ax1.fill_between(Xaxis[:],freeboard[:], snow[:,0]+freeboard[:,],facecolor='white',edgecolor='white')
CS1 = plt.contour(Xgrid[:-1,:-1], depth_contour, T, levelsT,colors='k')
plt.clabel(CS1, fontsize=9, inline=1,fmt='%1.0f')
plt.plot(Xaxis[:],freeboard[:],'k--')
plt.ylabel(r'depth [m]')
ax1.set_title('Temperature, liquid volume fraction, and bulk salinity')
ax1.set_facecolor((0.5,0.5,0.5))
ax1.xaxis.set_ticklabels([])

#######################################################
#Plotting liquid fraction
#######################################################
ax2 = fig1.add_subplot(312)
fsize=10.
zmin=0.
zmax=1.

#plt.pcolor(Xgrid,depth,psi_l,cmap='bone_r',vmin=zmin,vmax=zmax)
plt.pcolor(Xgrid,depth,psi_l,cmap=psi_cmap,vmin=zmin,vmax=zmax)
plt.axis([Xgrid.min(), Xgrid.max(), ymin, ymax])
c2 = plt.colorbar(pad=0.01)
c2.set_label(r'$\phi_l$')
CS2 = plt.contour(Xgrid[:-1,:-1], depth_contour, psi_l, levelspsi, colors='k')
plt.clabel(CS2, fontsize=9, inline=1,fmt='%1.1f')
#ax2.fill_between(Xaxis[:],freeboard[:], snow[:,0]+freeboard[:,],facecolor='white',edgecolor='white')
plt.plot(Xaxis[:],freeboard[:],'k--')
plt.ylabel(r'depth [m]')
ax2.xaxis.set_ticklabels([])
ax2.set_facecolor((0.5,0.5,0.5))


#######################################################
#Plotting salinity
#######################################################
ax3 = fig1.add_subplot(313)
fsize=10.
zmin=0.
zmax=S.max()
#plt.pcolor(Xgrid,depth,S,cmap='Greens',vmin=zmin,vmax=zmax)
plt.pcolor(Xgrid,depth,S,cmap=S_cmap,vmin=zmin,vmax=zmax)
plt.axis([Xgrid.min(), Xgrid.max(), ymin, ymax])
c3 = plt.colorbar(pad=0.01)
c3.set_label(r'$S_{bu}$')
CS3 = plt.contour(Xgrid[:-1,:-1], depth_contour, S, levelsS,colors='k')
plt.clabel(CS3, fontsize=9, inline=1,fmt='%2.0f')
#ax3.fill_between(Xaxis[:],freeboard[:], snow[:,0]+freeboard[:,],facecolor='white',edgecolor='white')
plt.plot(Xaxis[:],freeboard[:],'k--')
plt.ylabel(r'depth [m]')
ax3.set_facecolor((0.5,0.5,0.5))

# Set x ticks with times
ticks = ax3.get_xticks() * (60*60*24)
index = (ticks / config['time_out']).astype(int)
times_plot = [str(time[idx].strftime(format='%Y-%m-%d %Hh')) for idx in index[:-1]]
#ax3.set_xticklabels(times_plot)
plt.xticks(rotation=20)

plt.tight_layout()

#Saving and exporting
plt.savefig('../../Plots/'+outputfile+'.'+outputformat, dpi=1000)
plt.show()
plt.close()

