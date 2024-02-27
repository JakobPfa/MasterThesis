import json
import sys
import os
sys.path.insert(0, '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/Python_files/Code/build_config_files/')
from func_bound_values import *
import os
import numpy as np



# set wd to build_config_files folder
wd = os.getcwd()
# check if already in correct wd
if wd[-18:] == 'build_config_files':
    pass
# if not, change to build_config_files
else:
    path = '/home/jakobp/MasterThesis/SAMSIM/samsim3.0-master/Python_files/Code/build_config_files'
    os.chdir(path)

# set constants
rho_l = 1028
c_l = 3400

# **********************************************************************************************************************
# Initialize array containig all parameters and set path to its directory
# **********************************************************************************************************************
config = {}
path_config = r'../../../Run_specifics/'
path_input = r'../../../input/'

# **********************************************************************************************************************
# Set description of run
# **********************************************************************************************************************
description = "Testcase_MOSAiC"
file = open(path_config + r"description.txt", "w")
file.write(description)
file.close()

# **********************************************************************************************************************
# Initial layer thickness, timestep, output time and simulation time
# **********************************************************************************************************************
# time to shorten input (starting at 01/01/2019) by
t_del = (31*5 + 30*2 + 28) * 24*60. # time from 01/01/2019 - 31/08/2019 in minutes

config['start_time'] = '2019-09-01 00:00:00'  # start time in YYYY-mm-dd HH:MM:SS
config['dt'] = 20.  # time increment [s]]
config['time'] = 0.0  # initial value of time [s]
config['time_out'] = 86400.  # time between outputs [s]
config['time_total'] = 55468800. - t_del*60 # total length of simulation [s]



# **********************************************************************************************************************
# Time settings needed when input is given
# **********************************************************************************************************************
config['timestep_data'] = 60  # timestep of input data [s]
config['length_input'] = config['time_total'] / 60. + 1  # Length of your input files. Must match with timestep_data

# **********************************************************************************************************************
# Layer settings and allocation
# **********************************************************************************************************************
config['thick_0'] = 0.02
config['Nlayer'] = 80
config['N_active'] = 1
config['N_top'] = 20
config['N_bottom'] = 20
config['N_middle'] = config['Nlayer'] - config['N_top'] - config['N_bottom']

# **********************************************************************************************************************
# Flags
# **********************************************************************************************************************
# ________________________top heat flux____________
config['boundflux_flag'] = 1
config['albedo_flag'] = 2
# ________________________brine_dynamics____________
config['grav_heat_flag'] = 2
config['flush_heat_flag'] = 2
config['flood_flag'] = 2
config['flush_flag'] = 5
config['grav_flag'] = 2
config['harmonic_flag'] = 2
# ________________________Salinity____________
config['prescribe_flag'] = 1
config['salt_flag'] = 1
# ________________________bottom setting______________________
config['turb_flag'] = 2  # was on two in 203
config['bottom_flag'] = 1
config['tank_flag'] = 1
# ________________________snow______________________
config['precip_flag'] = 1
config['freeboard_snow_flag'] = 1  # < Niels, 2017
config['snow_flush_flag'] = 1  # < Niels, 2017
config['styropor_flag'] = 0
config['lab_snow_flag'] = 0
# ________________________debugging_____________________
config['debug_flag'] = 1  # set to 2 for output of all ice layers each timestep
# ________________________bgc_______________________
config['bgc_flag'] = 1
# ________________________initial state_______________________
config['initial_state_flag'] = 1  # 2 if initial state is given

# **********************************************************************************************************************
# Tank and turbulent fluxes settings
# **********************************************************************************************************************
config['tank_depth'] = 0
config['alpha_flux_stable'] = 0
config['alpha_flux_instable'] = 0

# **********************************************************************************************************************
# BGC Settings
# **********************************************************************************************************************
config['N_bgc'] = 2
config['bgc_bottom_1'] = 400
config['bgc_bottom_2'] = 500

# **********************************************************************************************************************
# Construct Input files
# **********************************************************************************************************************

# Set constant inputs
const_inputs = {'T_bottom': -1.8, 'S_bu_bottom': 34,
                'fl_q_bottom': 1., 'precip_s':0., 'precip_l':0., 
                'fl_lw': 0, 'fl_sw': 0, 'fl_sen': 0, 'fl_lat':0,
                'T_top': -5}
for input in list(const_inputs.keys()):
    data = np.ones(int(config['length_input'])) * const_inputs[input]
    np.savetxt(path_input + input + '.txt', data) 
    
# shorten input files by Jan-Aug2019 & put them into main input folder, that feeds SAMSIM
#input_data = {'fl_lw', 'fl_sw', 'fl_sen', 'fl_lat', 'T2m'}#, 'precip_l'}
#for input in list(input_data):
#    data = np.loadtxt(wd + '/input/' + input + '.txt')
#    np.savetxt(path_input + input + '.txt', data[int(t_del-1):])
    


# **********************************************************************************************************************
# setting the initial values of the top and only layer - not needed if initial ice properties are given - set to zero
# **********************************************************************************************************************
config['thick_1'] = config['thick_0']
config['m_1'] = config['thick_0'] * rho_l
config['S_abs_1'] = config['m_1'] * const_inputs['S_bu_bottom']
config['H_abs_1'] = config['m_1'] * const_inputs['T_bottom'] * c_l

# **********************************************************************************************************************
# Write init to .json file
# **********************************************************************************************************************
json_object = json.dumps(config, indent=4)

with open(path_config + "config.json", "w") as outfile:
    outfile.write(json_object)
