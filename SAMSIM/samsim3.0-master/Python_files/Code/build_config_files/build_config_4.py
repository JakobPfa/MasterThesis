import json
from func_bound_values import *
import os

# set wd to the directory of the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


# set constants
rho_l = 1028
c_l = 3400

# **********************************************************************************************************************
# Initialize array containig all parameters and set path to its directory
# **********************************************************************************************************************
config = {}
path_config = r'../../../Run_specifics/'
path_input = '../../../input/'

# **********************************************************************************************************************
# Set description of run
# **********************************************************************************************************************
description = "Testcase_4"
file = open(path_config + r"description.txt", "w")
file.write(description)
file.close()

# **********************************************************************************************************************
# Initial layer thickness, timestep, output time and simulation time
# **********************************************************************************************************************
config['start_time'] = '2022-01-01 00:00:00'  # start time in YYYY-mm-dd HH:MM:SS
config['dt'] = 10  # time increment [s]]
config['time'] = 0.0  # initial value of time [s]
config['time_out'] = 86400  # time between outputs [s]
config['time_total'] = config['time_out'] * 365 * 4.5  # total length of simulation [s]

# **********************************************************************************************************************
# Time settings needed when input is given
# **********************************************************************************************************************
config['timestep_data'] = 3600*3  # timestep of input data [s]
config['length_input'] = 13169  # Length of your input files. Must match with timestep_data

# **********************************************************************************************************************
# Layer settings and allocation
# **********************************************************************************************************************
config['thick_0'] = 0.01
config['Nlayer'] = 100
config['N_active'] = 1
config['N_top'] = 20
config['N_bottom'] = 20
config['N_middle'] = config['Nlayer'] - config['N_top'] - config['N_bottom']

# **********************************************************************************************************************
# Flags
# **********************************************************************************************************************
# ________________________top heat flux____________
config['boundflux_flag'] = 2
config['albedo_flag'] = 2
# ________________________brine_dynamics____________
config['grav_heat_flag'] = 1
config['flush_heat_flag'] = 2
config['flood_flag'] = 2
config['flush_flag'] = 5
config['grav_flag'] = 2
config['harmonic_flag'] = 2
# ________________________Salinity____________
config['prescribe_flag'] = 1
config['salt_flag'] = 1
# ________________________bottom setting______________________
config['turb_flag'] = 2
config['bottom_flag'] = 1
config['tank_flag'] = 1
# ________________________snow______________________
config['precip_flag'] = 1
config['freeboard_snow_flag'] = 0  # < Niels, 2017
config['snow_flush_flag'] = 1  # < Niels, 2017
config['styropor_flag'] = 0
config['lab_snow_flag'] = 0
# ________________________debugging_____________________
config['debug_flag'] = 1  # set to 2 for output of all ice layers each timestep
# ________________________bgc_______________________
config['bgc_flag'] = 1
# ________________________initial state_______________________
config['initial_state_flag'] = 1

# **********************************************************************************************************************
# Tank and turbulent fluxes settings
# **********************************************************************************************************************
config['tank_depth'] = 1
config['alpha_flux_stable'] = 15
config['alpha_flux_instable'] = 22

# **********************************************************************************************************************
# BGC Settings, only relevant if bcg_flag = 2
# **********************************************************************************************************************
config['N_bgc'] = 2
config['bgc_bottom_1'] = 385
config['bgc_bottom_2'] = 385

# **********************************************************************************************************************
# Construct Input files
# **********************************************************************************************************************
# Set constant inputs
const_inputs = {'T_bottom': -1, 'S_bu_bottom': 34,
                'fl_q_bottom': 8, 'precip_s': 0, 'fl_sen': 0, 'fl_lat': 0}
for input in list(const_inputs.keys()):
    data = np.ones(config['length_input']) * const_inputs[input]
    np.savetxt(path_input + input + '.txt', data)


# Set variable inputs
fl_q_bottom = np.ones(config['length_input'])
for i in range(config['length_input']):
    fl_q_bottom[i] = -7 * np.sin(i * config['timestep_data']*(2*np.pi)/(86400*365))+7

np.savetxt(path_input + 'fl_q_bottom.txt', fl_q_bottom)

T2m = np.loadtxt('../../Data/T2m.txt.input')
precip_l = np.loadtxt('../../Data/precip.txt.input')
fl_sw = np.loadtxt('../../Data/flux_sw.txt.input')
fl_lw = np.loadtxt('../../Data/flux_lw.txt.input')
np.savetxt(path_input + '/T2m.txt', T2m)
np.savetxt(path_input + '/precip_l.txt', precip_l)
np.savetxt(path_input + '/fl_sw.txt', fl_sw)
np.savetxt(path_input + '/fl_lw.txt', fl_lw)

# **********************************************************************************************************************
# setting the initial values of the top and only layer
# **********************************************************************************************************************
config['thick_1'] = config['thick_0']
config['m_1'] = config['thick_0'] * rho_l
config['S_abs_1'] = config['m_1'] * const_inputs['S_bu_bottom']
config['H_abs_1'] = 0

# **********************************************************************************************************************
# Write init to .json file
# **********************************************************************************************************************
json_object = json.dumps(config, indent=4)

with open(path_config + "config.json", "w") as outfile:
    outfile.write(json_object)

