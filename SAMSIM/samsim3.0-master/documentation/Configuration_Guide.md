# Configuration Guide 
How to configure SAMSIM3.0 with config.json files and the respective inputs

## Brief Description
This Guide will show you an example run of SAMSIM3.0 with testcase 1. We go step-by-step through the python file build_config_1.py which is used to build the config.json file and
the input files needed to run SAMSIM3.0 for testcase 1. For a detailed documentation of the different variables, please look at
reference manual and SAMISM3.md. 

## Build the configuration file 

We use the build_config_1.py file, which is used to create the config.json and the respective input needed for 
testcase 1 as an example. 

The first section imports the json lbrary and the functions used to build the input files for the respective testcases.
It also defines two constants used below and sets the working directory to the directory of the build_config file. 

```python
import json
from func_bound_values import *
import os

# set wd to the directory of the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)

# set constants
rho_l = 1028
c_l = 3400
os.chdir(dname)
```
The second section initializes the config array which is converted into the config.json file in the end and sets the
paths for the storage spaces of the input files and the config file.
```python
# **********************************************************************************************************************
# Initialize array containig all parameters and set path to its directory
# **********************************************************************************************************************
config = {}
path_config = r'../../../Run_specifics/'
path_input = '../../../input/'
```
The tird section sets the description of the run which is then saved to the file description.txt in the Run_specifics folder
```python
# **********************************************************************************************************************
# Set description of run
# **********************************************************************************************************************
description = "Testcase_1"
file = open(path_config + r"description.txt", "w")
file.write(description)
file.close()
``` 
In the fourth section, the time settings of the simulation are defined. Note that 'time_total' 
should be a multiple of 'time_out'. 'start_time' sets the beginning of your run, which is important if you 
want to have a realistic timeframe. For testcase 1-4 this is arbitrary. 
```python
# **********************************************************************************************************************
# Initial layer thickness, timestep, output time and simulation time
# **********************************************************************************************************************
config['start_time'] = '2022-01-01 00:00:00'  # start time in YYYY-mm-dd HH:MM:SS
config['dt'] = 1  # time increment [s]]
config['time'] = 0.0  # initial value of time [s]
config['time_out'] = 3600  # time between outputs [s]
config['time_total'] = config['time_out'] * 72  # total length of simulation [s]
```
As we are always working with input data in SAMSIM3.0 the length of the input, which is just the total number of
datapoints in time of your input. For the testcases 1-3 we construct the input below with the same timestep 'dt' 
as the simulation, which is why 'length_input = 'time_total' / 'dt'. 
```python
# **********************************************************************************************************************
# Time settings needed when input is given
# **********************************************************************************************************************
config['length_input'] = config['time_total'] / config['dt']  # Length of your input files. Must match with timestep_data
config['timestep_data'] = 1  # timestep of input data [s]
```
The next section just sets the nuber and distribution of layers.
```python
# **********************************************************************************************************************
# Layer settings and allocation
# **********************************************************************************************************************
config['thick_0'] = 0.002
config['Nlayer'] = 90
config['N_active'] = 1
config['N_top'] = 5
config['N_bottom'] = 5
config['N_middle'] = config['Nlayer'] - config['N_top'] - config['N_bottom']
```
The next section sets all the required flags. The only flag wich is new introduced in SAMSIM3.0 is the 'initial_state_flag'
which is 2 for a given initial ice state and 1 for no initial state. 
```python
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
config['debug_flag'] = 1  
# ________________________bgc_______________________
config['bgc_flag'] = 2
# ________________________initial state_______________________
config['initial_state_flag'] = 1
```
This section defines the settings for tank experiments (testcase 2). If no tank experiment is carried out ('tank_flag' = 1) 
```python
# **********************************************************************************************************************
# Tank and turbulent fluxes settings
# **********************************************************************************************************************
config['tank_depth'] = 0
config['alpha_flux_stable'] = 0
config['alpha_flux_instable'] = 0
```
This section defines the nuber of chemicals and their initial concentration
```python
# **********************************************************************************************************************
# BGC Settings
# **********************************************************************************************************************
config['N_bgc'] = 2
config['bgc_bottom_1'] = 400
config['bgc_bottom_2'] = 500
```
### Set Input 
This section constructs the input files and is very individual for each testcase. The inputs which should be constant 
are defined in const_inputs. They are then expanded to a timeseries of length_input and saved as .txt files in the 
input folder. Below, the variable inputs are calculated and saved, which is T_top for testcase 1. 
Note that input files can also be given directly (see testcase 4). However, all inputs must be provided to run the model.
So if you don't one, just define it as constantly 0. 
```python
# **********************************************************************************************************************
# Construct Input files
# **********************************************************************************************************************

# Set constant inputs
const_inputs = {'T2m': 0, 'T_bottom': -1, 'S_bu_bottom': 34,
                'fl_q_bottom': 0, 'precip_l': 0, 'precip_s': 0}
for input in list(const_inputs.keys()):
    data = np.ones(int(config['time_total']/config['timestep_data'])) * const_inputs[input]
    np.savetxt(path_input + input + '.txt', data)

# Set variable inputs
T_top = t_top_testcase_1(config['time_total'], config['timestep_data'])
np.savetxt(path_input + 'T_top.txt', T_top)
```
Here, the initial states of the top and only layer are set.
```python
# **********************************************************************************************************************
# setting the initial values of the top and only layer
# **********************************************************************************************************************
config['thick_1'] = config['thick_0']
config['m_1'] = config['thick_0'] * rho_l
config['S_abs_1'] = config['m_1'] * const_inputs['S_bu_bottom']
config['H_abs_1'] = config['m_1'] * const_inputs['T_bottom'] * c_l
```
### Give Initial State
If an initial state should be provided, this section from build_config_initial_state can be used, which uses the function 
build_initial_state() to construcht the necessary arrays and calculate the number of active layers fo a given ice thickness#
Note that this is just one possible way of constructing the initial state. The temperature and salinity profile 
given in build_initial_state() can be cahngeged arbitrarily. 
````python
# **********************************************************************************************************************
# Construct initial state fields H_abs, S_abs, m, thick
# **********************************************************************************************************************

# individual configuration
Ice_thickness = 0.1
S_bu_top = 3.
T_top = -5

H_abs, S_abs, m, thick, config['N_active'] = build_inital_state(Ice_thickness, S_bu_top, T_top, config, const_inputs)

for (fname, data) in [('H_abs.txt', H_abs), ('S_abs.txt', S_abs), ('thick.txt', thick), ('m.txt', m)]:
    np.savetxt(path_input + fname, data)
````

Write the config dictionary to a .json file and save it in the respective folder. 
```python
# **********************************************************************************************************************
# Write config to .json file
# **********************************************************************************************************************
json_object = json.dumps(config, indent=4)

with open(path_config + "config.json", "w") as outfile:
    outfile.write(json_object)
```
### Run config script
You can run the build_config_1 file to prepare testcase 1 by simpely typing 
````shell
python3 Python_files/Code/build_config_files/build_config_1.py
````
into your console. Note that the path can be different, depending on what folder you are in. 
Running the script will put the needed input in the input folder and the config.json and the description.txt file
in the Run_specifics folder. 

### Run SAMSIM
Now, you can run the model by going to the directory with the makefile in it (home directory of the repository) 
and typing 
````shell
make
````
into the console. This will compile the fortran code and produce a .mod and a .o file for every module and an executable file 
called samsim.x. Alternatively, if you do not want to compile the model, you can use the executable file provided in the 
repository, which is also called samsim.x

Having done that, you can now run SAMSIM by typing
````shell
./samsim.x
````
into the console, which will run the executable file. One good thing about SAMSIM3.0 is that you 
don't have to recompile it after you changed the inputs or the config file. You can simpely type the above command again
to rerun it for a different setting. 

Runnig the SAMSIM will produce some output in your console, showing the progress and some other model variables.

After SAMSIM finishes, you can find the model output in the output foler. 

### Plotting the output
The model output can simpely be plotted by running one of the plotscrips, depending on what you want to investigate. 
For a good first overview I suggest using the plot_TPhiS.py script, which plots temperature, liquid fraction and salinity of the 
sea ice profile over time. To do so, type
````shell
python3 Python_files/Code/plotscripts/plot_TPhiS.py
````
The path again depends on which folder you are currently in. This produces a nice plot which you can find in the folder 
Python_files/Plots and should look like this for testcase 1:

![Testcase1](Plots/pic_TPhiS.png?raw=true)