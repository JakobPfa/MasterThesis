#Standard plot script for Temperature, liquid fraction, and bulk salinity
#Steps:
#1. Set output name with format to save (jpg, png, pdf, or eps). Pdf and eps look better but can be very large and slow to load.
#2. Do you want the freeboard to by incorporated into your y-axis? set free_flag
#3. Run the script! (python plot_TPhiS.py)
#4. Open the outputfile in the imageviewer of your choice.
#5. Optional: Adjust contour levels and repeat.

#Loading modules and setting fonts
import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib import rc
import matplotlib
import os
import json
import pandas as pd
import datetime

# Warnings:
# Contours are interpolated from the middle of each layer. This is most visible in thick layers, and in the snow layer where the contour lines only extend to the the middle.

#%% Settings:
outputfile   = 'pic_TPhiS'
outputformat = 'png' #e.g. png, jpg, pdf
free_flag    = 1     #1: freeboard is included, 0:freeboard is not included

# set wd to the directory of the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# importing data from runs
path_snow = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/precip/run_MOSAiC_FYI_realsnow/output'
path_double_snow = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/precip/run_MOSAiC_FYI_double_realsnow/output'
path_half_snow = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/precip/run_MOSAiC_FYI_half_realsnow/output'
path_no_snow = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/precip/run_MOSAiC_FYI_no_realsnow/output'

paths = [path_snow, path_double_snow, path_half_snow, path_no_snow]
runs = ['real','double','half', 'no']

#dat = {}
#for run, path in zip(runs,paths):
#    dat[run] = {var: np.loadtxt(path + '/dat_'+var+'.dat') for var in ['T', 'S_bu' ,'psi_l', 'psi_s', 'thick', 'freeboard', 'snow', 'vital_signs']}


#%% define function to load data and build Xgrid 
def load_data_grid(run_path): # returns dictionary to put into bigger dictionary of runs
    data_grid = {}
    S          = numpy.loadtxt(run_path + "/output/dat_S_bu.dat")
    T          = numpy.loadtxt(run_path + "/output/dat_T.dat")
    psi_l      = numpy.loadtxt(run_path + "/output/dat_psi_l.dat")  #volume fraction of liquid
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

    data_grid = {'T': T, 'S': S, 'psi_l': psi_l, 'thick': thick, 'freeboard': freeboard, 'snow': snow, 
                 'Xgrid': Xgrid, 'depth': depth, 'vital_signs': vital_signs, 'T2m_T_top': T2m_T_top}
    
    return data_grid
    
# collect data: 
path_runs = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/runs/'

path_k_0_005 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_005'
path_k_0_01 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_01'
path_k_0_05 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_05'
path_k_0_1 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_1'
path_k_0_175 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_175'
path_k_0_25 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_25'
path_k_0_5 = path_runs + 'k/run_MOSAiC_FYI_ksnow_0_5'
path_k_1_0 = path_runs + 'k/run_MOSAiC_FYI_ksnow_1_0'
path_k_1_5 = path_runs + 'k/run_MOSAiC_FYI_ksnow_1_5'

paths = {'k_0_005': path_k_0_005, 'k_0_01': path_k_0_01, 'k_0_05': path_k_0_05, 'k_0_1': path_k_0_1, 'k_0_175': path_k_0_175, 
         'k_0_25': path_k_0_25, 'k_0_5': path_k_0_5, 'k_1_0': path_k_1_0, 'k_1_5': path_k_1_5}
runs = ['k_0_005', 'k_0_01','k_0_05','k_0_1','k_0_175','k_0_25','k_0_5', 'k_1_0', 'k_1_5']
plot_labels = {'k_0_005':'k_snow = 0.005', 'k_0_01':'k_snow = 0.01', 'k_0_05':'k_snow = 0.05', 'k_0_1': 'k_snow = 0.1', 'k_0_175': 'k_snow = 0.175', 
               'k_0_25': 'k_snow = 0.25', 'k_0_5': 'k_snow = 0.5', 'k_1_0': 'k_snow = 1.0', 'k_1_5': 'k_snow = 1.5'}

dat = {}
for run in runs:
    dat[run] = load_data_grid(paths[run])

#%% loading data
outputpath = 'output'

S          = numpy.loadtxt(path_snow + "/dat_S_bu.dat")
T          = numpy.loadtxt(path_snow + "/dat_T.dat")
psi_l      = numpy.loadtxt(path_snow + "/dat_psi_l.dat")  #volume fraction of liquid
psi_s      = numpy.loadtxt(path_snow + "/dat_psi_s.dat")
thick      = numpy.loadtxt(path_snow + "/dat_thick.dat")
freeboard  = numpy.loadtxt(path_snow + "/dat_freeboard.dat")
snow       = numpy.loadtxt(path_snow + "/dat_snow.dat")

S_double          = numpy.loadtxt(path_double_snow + "/dat_S_bu.dat")
T_double          = numpy.loadtxt(path_double_snow + "/dat_T.dat")
psi_l_double      = numpy.loadtxt(path_double_snow + "/dat_psi_l.dat")  #volume fraction of liquid
thick_double      = numpy.loadtxt(path_double_snow + "/dat_thick.dat")
freeboard_double  = numpy.loadtxt(path_double_snow + "/dat_freeboard.dat")
snow_double       = numpy.loadtxt(path_double_snow + "/dat_snow.dat")

S_half          = numpy.loadtxt(path_half_snow + "/dat_S_bu.dat")
T_half          = numpy.loadtxt(path_half_snow + "/dat_T.dat")
psi_l_half      = numpy.loadtxt(path_half_snow + "/dat_psi_l.dat")  #volume fraction of liquid
thick_half      = numpy.loadtxt(path_half_snow + "/dat_thick.dat")
freeboard_half  = numpy.loadtxt(path_half_snow + "/dat_freeboard.dat")
snow_half       = numpy.loadtxt(path_half_snow + "/dat_snow.dat")

S_no          = numpy.loadtxt(path_no_snow + "/dat_T.dat")
psi_l_no      = numpy.loadtxt(path_no_snow + "/dat_psi_l.dat")  #volume fraction of liquid
thick_no      = numpy.loadtxt(path_no_snow + "/dat_thick.dat")
freeboard_no  = numpy.loadtxt(path_no_snow + "/dat_freeboard.dat")
snow_no       = numpy.loadtxt(path_no_snow + "/dat_snow.dat")
  

# Load config file
with open('../Run_specifics/config.json') as json_file:
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
psi_s_snow = snow[:,3]
psi_s_snow = psi_s_snow.reshape(xlen,1)
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

depth = numpy.column_stack((depth,depth[:,-1]-(depth[:,-1]-depth[:,-2])))
Xgrid = numpy.column_stack((Xgrid,Xgrid[:,-1]))

depth = numpy.vstack((depth, depth[-1,:]))
Xgrid = numpy.vstack((Xgrid, Xgrid[-1,:]))


# resolve bottom layer with psi_s_min:
#psi_s_min = 0.05
#Nactive = np.sum(thick != 0., axis = 1) - 1
#for i in np.arange(Nactive.shape[0]):
#    thick[i,Nactive[i]] = thick[i,Nactive[i]] * psi_s[i,Nactive[i]] /psi_s_min
#    depth[i,Nactive[i]] = depth[i,Nactive[i]] * psi_s[i,Nactive[i]] /psi_s_min

#%% Custom colormaps
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
#%%Plotting
#######################################################

##################################################
#plt.pcolormesh(Xgrid,depth,T)
#plt.axis([Xgrid.min(), Xgrid.max(), ymin, ymax])
#plt.colorbar()
#plt.show()
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
fsize=10.#[:399,:80]
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
#plt.savefig('../../Plots/'+outputfile+'.'+outputformat, dpi=1000)
plt.show()
#plt.close()




#%% comparing salinity profiles

# take time of maximal thickness:
t_thick_max = {}    
date_max_thick = {}
for run in runs:
    t_thick_max[run] = np.argmax(dat[run]['vital_signs'][:,3])
    print('time of maximal thickness for ', plot_labels[run], ': ', 
          datetime.date(year = 2020, month = 9, day = 1) + datetime.timedelta(days=int(t_thick_max[run])))
    date_max_thick[run] = datetime.date(year = 2020, month = 9, day = 1) + datetime.timedelta(days=int(t_thick_max[run]))

    
for run in runs:    
    plt.plot(dat[run]['S'][t_thick_max[run],:],
             dat[run]['depth'][t_thick_max[run],1:]/dat[run]['vital_signs'][t_thick_max[run],3], 
             label = plot_labels[run] + ' at ' + str(date_max_thick[run]))
plt.legend()
plt.xlabel('Salinity [g/kg]')
plt.ylabel('depth realtive to maximal ice thickness')
plt.title('Salinity profiles at point of maximal ice thickness')
plt.show()

for run in runs:    
    plt.plot(dat[run]['T'][t_thick_max[run],:],
             dat[run]['depth'][t_thick_max[run],1:]/dat[run]['vital_signs'][t_thick_max[run],3], 
             label = plot_labels[run])
plt.legend()
plt.xlabel('temperature [°C]')
plt.ylabel('depth realtive to maximal ice thickness')
plt.title('Temperature profiles at point of maximal ice thickness')
plt.show()


for run in runs:    
    plt.plot(dat[run]['S'][228,:],
             dat[run]['depth'][228,1:]/dat[run]['vital_signs'][228,3], 
             label = plot_labels[run])
plt.legend()
plt.xlabel('Salinity [g/kg]')
plt.ylabel('depth realtive to maximal ice thickness')
plt.title('Salinity profiles in mid-april')
plt.show()

#%% plot 

for run in runs:
    plt.pcolor(dat[run]['Xgrid'], dat[run]['depth'], dat[run]['S'], cmap = S_cmap)
    plt.plot(dat[run]['depth'][:,0], color = 'darkgreen')
    plt.plot(dat[run]['depth'][:,1], color = 'darkgreen')
    c1 = plt.colorbar(pad=0.01)
    c1.set_label(r'S [g/kg]')
    plt.axis([dat[run]['Xgrid'].min(), dat[run]['Xgrid'].max(), 
              dat[run]['depth'].min()*1.03, dat[run]['depth'].max()*1.05])
    plt.xlabel('time [days]')
    plt.ylabel('depth [m]')
    plt.title('Salinity profile for ' + plot_labels[run])
    plt.show()




