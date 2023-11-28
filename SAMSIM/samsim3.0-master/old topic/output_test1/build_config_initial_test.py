import json
from func_bound_values import *
import os
import matplotlib.pyplot as plt

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
description = "Testcase_1"
file = open(path_config + r"description.txt", "w")
file.write(description)
file.close()

# **********************************************************************************************************************
# Initial layer thickness, timestep, output time and simulation time
# **********************************************************************************************************************
# start time in YYYY-mm-dd HH:MM:SS
config['start_time'] = '2022-01-01 00:00:00'
config['dt'] = 1  # time increment [s]]
config['time'] = 0.0  # initial value of time [s]
config['time_out'] = 3600  # time between outputs [s]
config['time_total'] = config['time_out'] * 24 * 4  # total length of simulation [s]

# **********************************************************************************************************************
# Time settings needed when input is given
# **********************************************************************************************************************
# Length of your input files. Must match with timestep_data
config['length_input'] = config['time_total'] / config['dt']
config['timestep_data'] = 1  # timestep of input data [s]

# **********************************************************************************************************************
# Layer settings and allocation
# **********************************************************************************************************************
config['thick_0'] = 0.005
config['Nlayer'] = 90
# N_active gets calculated in build_initial_stae() and depends on prescribes ice thickness, thick_0 and the choosen
# numbers for N_top and N_bottom
config['N_top'] = 10
config['N_bottom'] = 10
config['N_middle'] = config['Nlayer'] - config['N_top'] - config['N_bottom']

# **********************************************************************************************************************
# Flags
# **********************************************************************************************************************
# ________________________top heat flux____________
config['boundflux_flag'] = 1
config['albedo_flag'] = 2
# ________________________brine_dynamics____________
config['grav_heat_flag'] = 1
config['flush_heat_flag'] = 1
config['flood_flag'] = 2
config['flush_flag'] = 1
config['grav_flag'] = 2
config['harmonic_flag'] = 2
# ________________________Salinity____________
config['prescribe_flag'] = 1
config['salt_flag'] = 2
# ________________________bottom setting______________________
config['turb_flag'] = 1
config['bottom_flag'] = 1
config['tank_flag'] = 1
# ________________________snow______________________
config['precip_flag'] = 0
config['freeboard_snow_flag'] = 0
config['snow_flush_flag'] = 1
config['styropor_flag'] = 0
config['lab_snow_flag'] = 0
# ________________________debugging_____________________
config['debug_flag'] = 1  # set to 2 for output of all ice layers each timestep
# ________________________bgc_______________________
config['bgc_flag'] = 2
# ________________________initial state_______________________
config['initial_state_flag'] = 2  # 2 if initial state is given

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
const_inputs = {'T2m': 0, 'T_bottom': -1.72, 'S_bu_bottom': 34,
                'fl_q_bottom': 0, 'precip_l': 0, 'precip_s': 0}
for input in list(const_inputs.keys()):
    data = np.ones(
        int(config['time_total']/config['timestep_data'])) * const_inputs[input]
    np.savetxt(path_input + input + '.txt', data)

# Set variable inputs
T_top = Tstep(-10, -5, config['time_total'], config['timestep_data'])
np.savetxt(path_input + 'T_top.txt', T_top)

# **********************************************************************************************************************
# setting the initial values of the top and only layer - not needed if initial ice properties are given - set to zero
# **********************************************************************************************************************
config['thick_1'] = 0
config['m_1'] = 0
config['S_abs_1'] = 0
config['H_abs_1'] = 0

# **********************************************************************************************************************
# Construct initial state fields H_abs, S_abs, m, thick
# **********************************************************************************************************************

# individual configuration
Ice_thickness = 0.5
S_bu_top = 8.
T_top = -10

H_abs, S_abs, m, thick, config['N_active'] = build_inital_state(
    Ice_thickness, S_bu_top, T_top, config, const_inputs)

for (fname, data) in [('H_abs.txt', H_abs), ('S_abs.txt', S_abs), ('thick.txt', thick), ('m.txt', m)]:
    np.savetxt(path_input + fname, data)

# **********************************************************************************************************************
# Write init to .json file
# **********************************************************************************************************************
json_object = json.dumps(config, indent=4)

with open(path_config + "config.json", "w") as outfile:
    outfile.write(json_object)
