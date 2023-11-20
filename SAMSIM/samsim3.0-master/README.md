SAMSIM3.0 
============
A more user-friendy version of SAMSIM2.0

## Overview
This Readme provides you with information on how to compile and run SAMSIM3.0 and how to plot the output.

Further documentation on the differences of SAMSIM3.0 compared to its predecessor SAMSIM2.0 can be found in SAMSIM3. 
New vaiables introduced in SAMSIM3.0 are also listed in this file. 

For information on all other variables, please have a look at 
the reference manual. 

A detailed description on how SAMSIM works can be found in Griewank and Notz (2013, 2015).
A detailed guide on how to use the python configuration files to configure SAMSIM runs is given in the Configuration_Guide. 


## Brief Description
This is an improved version of SAMSIM2.0 to make a more clear distingtion between inputs of SAMSIM and 
internal model processes. Several coding errors producing warnings from the compiler have been removed.
Different to SAMSIM2.0, this version does not need to be recompiled for changing model inputs/testcases.

Additionally, the option of initializing the model with prescribed sea-ice conditions 
has been added. 

You should be able to either use the executable file to run the model or compile it yourself. Both workflows are described below. 

## Running the model with the pre-compiled executalbe 
If you don't want to complie the model on your own computer, you should be able to use the provided executable (samsim.x). 
This way, you don't need to have a fortran compiler on your local maschine. You only need to have python installed.


## Compiling the Model

You need a Fortran compiler, preferably gfortran, which can be installed as follows:
### Windows: 

````bash 
# open Linux sub-system: 
bash
# Install gfortran
sudo apt-get install gfortran
````

### Linux
````bash
# Install gfortran
sudo apt-get install gfortran
````

### Mac 
````bash
# Install gfortran with Homebrew
brew install gcc
````

Further information can be found [here](https://fortran-lang.org/learn/os_setup/install_gfortran).

Once you have a fortran compiler installed you need to compile the json-fortran-master library which we need to read the config.json file. 
There are various ways of doing this, which are described in ./json-fortran-master/README.md. We use the build.sh script which should work with all operating systems. 
Tu run the build.sh script, you will need to install FoBis.py and ford by typing: 
`````bash
pip install FoBis.py
pip install ford
`````

Now you can run the build.sh script in Linux or Windowns in the linux sub system (see above) by typing: 
````bash
# Move to right directory
cd json-fortran-master
# run build.sh
./build.sh
````
The build.sh script might throw a few errors but should work. If this doesn't work for you, check the json README for alternative ways of building the library. 

Now you should be able to compile SAMSIM by moving back to the home directory (where the make file is) and running the make file. 
For Windows (in Linux sub-system) and Linux just type the following:

````bash
make
````
This produces the executable file samsim.x which can be run in Windows (Linux sub-system) and Linux by running 
````bash
./samsim.x
````
from the home directory. 


## Workflow

After compiling the model (see above) you can produce different model runs by changing the model configuration in the 
config.json file and the respective input files which can be found in the input folder.
Four testcases are available for you to start with and explore the behaviour of samsim. To run those testcases, you need 
to run the respective build_config.py script. Running one of the build_config.py scripts produces a config.json and a 
description.txt file which are stored in the folder Run_specifics. 
Furthermore, it creates the input files for SW, LW, sensitive and latent heatflux, for solid and liquid
precipitation, 2m temperature, surface temperature, bottom temperature, bottom salinity and bottom heatflux. 
If one of those input variables is not used, it is set to zero.

The config.json file for testcase 1 can be produced with the makefile. 
Windows user should activate their Linux sub-system to be 
able to use the makefile by typing:
````bash
bash
````
into the console. Now you can use all Linux commands. The config.json file for testcase 1 can be created with the makefile by typing 

````bash
make config.json
````

If you want to run one of the testcases 2-4, you need to run the respective build_config.py file with python
which can simpely be done by typing 

````bash
python3 Python_files/Code/build_config_files/build_config_x.py
````
and replace x with the number of the testcase you want to run.

To run the model with initial sea-ice profiles, run the build_config_initial_state.py script. 

If you want to set up your own model run, you need to change the config.json file and the input files accordingly. 
For further information on how to use the build_config files, please have a look at the Configuration Guide in the documentation folder.

After producing the config.json file and putting the right inputs in the input folder, SAMSIM just needs to be run by typing: 

````bash
./samsim.x
````
This can either be the pre-compiled samsim.x file from the repository or the one you compiled yourself. 

### Files you need to run SAMSIM

- **Config.json** with all parameters (look at parameters given in the build_config.py files or in mo.init). 
Config.json should be stored in the Run_specifics folder.
- **Input files** Which input files you need depends on your boundflux_flag.
- Boundflux flag = 1:
    - T_top
    - T_bottom
    - S_bu_bottom
    - fl_q_bottom
    - precip_l
    - precip_s
- Boundflux_flag = 2:
  - fl_lw
  - fl_sw
  - fl_sens
  - fl_lat
  - T2m
  - T_bottom
  - S_bu_bottom
  - fl_q_bottom
  - precip_l
  - precip_s
- Boundflux_flag = 3:
    - T2m
    - T_top (just for inital state of the ice)
    - T_bottom
    - S_bu_bottom
    - fl_q_bottom
    - precip_l
    - precip_s

    
  

## Plotting the Model Output
Creating first plots of the output can be done by running the plotscripts in Python_files/Code/plotscripts. Please follow 
the advice given at the top of each plot script. 


## Documentation 
Further documentation can be found in the documentation folder. 
A detailed description of the differences compared to SAMSIM2.0 is given in SAMSIM3.md and a detailed guide on how to
configure SAMSIM together with an example run is given in the Configuration_Guide.md 

## Contact
In case you have questions which can not be solved by scanning the documentation, feel free to contact
jakob.deutloff@gmail.com or niels.fuchs@uni-hamburg.de. 

