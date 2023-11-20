# SAMSIM3.0 
Differences and improvements compared to SAMSIM2.0

## Warnings removed 
Athe warnings which have been thrown in SAMSIM2.0 have been removed. They were mostly caused by dummy variables or the
comparisions of floating point numbers. The latter has been removed by introducing a tolerance in mo.parameters: 
````fortran
  REAL(wp)  ::  tolerance = 1.0D-50
````
With this, folating point number comparisions have been rewritten like this: 

````fortran
IF (abs(time - time_input(time_counter)) < tolerance) THEN
````

Another cause for a lot of warnings was the use of the same variable fot T and T_in in mo.snow in the snowcoupling subroutine. 
This warning have been removed by introducing an additional variable T_in: 

````fortran
    T_in = T_snow
    CALL getT(H_abs_snow/m_snow, S_abs_snow/m_snow,  T_in, T_snow, phi_s, 5701)
````
This improvement of the code caused minor changes in the output of the testcases which have a snow cover (3&4). For testcase 3, the changes look like this: 

![testcase3_changes](Plots\testcase3_v1_v2.png)

There have been a lot of other warnings removed but these were the major ones. 

## Config File and mo.init 
The file mo.init has been changed completely. Instead of taking a testcase, it always takes the parameters from the config.json file now. 
This maked the need of recompiling SAMSIM for every testcase redundant. Instead, SAMSIM only needs to be compiled once and all 
changes of the model run can be implemented by changing the config.json file or the input files (see Readme - Workflow). 
To make this work, the json-fortran library was downloaded from [here](https://github.com/jacobwilliams/json-fortran). 
In the mo.init file, every parameter from the config.json file is read in like this: 

```fortran
CALL json%get('dt', dt, is_found)
IF (.not. is_found) THEN; PRINT*, 'dt not found'; STOP; END IF
```
Every parameter needed in the input file needs to be included in the config.json file, otherwise SAMSIM stopps and prints to the console which parameter is missing.


## Input Files  
Different to SAMSIM2.0 all inputs needed are defined outside SAMSIM and given as .txt files which are then read in. 
This way, it should become more transperent which inputs are used. In SAMSIM2.0, boundary values like T_top have been
calculated in the model which made it very untransparent. Which inputs are needed depends on the boundaryflux_flag (see Readme - workflow).
The input files are read in by sub_input() in mo_init. At the beginning of every timestep, the input is read in and if necessary linearly interpolated
between the closest data point in mo_grotz: 
````fortran
  IF (abs(time - time_input(time_counter)) < tolerance) THEN
                solid_precip = precip_s_input(time_counter)
                liquid_precip = precip_l_input(time_counter)
                T2m = T2m_input(time_counter)
                T_bottom = T_bottom_input(time_counter)
                fl_q_bottom = fl_q_bottom_input(time_counter)
                IF (boundflux_flag == 1) THEN
                    T_top = T_top_input(time_counter)
                END IF
                ! in tank experiments, bulk salinity at bottom varies since total amount of salt must be conserved
                IF (tank_flag .NE. 2) THEN
                    S_bu_bottom = S_bu_bottom_input(time_counter)
                END IF

            ! If timestep is different, interpolate linearly between boundary values
            ELSE
                temp = (time - time_input(time_counter - 1)) / (time_input(time_counter) - time_input(time_counter - 1))
                solid_precip = (1._wp - temp) * precip_s_input(time_counter - 1) + temp * precip_s_input(time_counter)
                liquid_precip = (1._wp - temp) * precip_l_input(time_counter - 1) + temp * precip_l_input(time_counter)
                T2m = (1._wp - temp) * T2m_input(time_counter - 1) + temp * T2m_input(time_counter)
                T_bottom = (1._wp - temp) * T_bottom_input(time_counter - 1) + temp * T_bottom_input(time_counter)
                fl_q_bottom = (1._wp - temp) * fl_q_bottom_input(time_counter - 1) + temp * fl_q_bottom_input(time_counter)
                IF (boundflux_flag == 1) THEN
                    T_top = (1._wp - temp) * T_top_input(time_counter - 1) + temp * T_top_input(time_counter)
                END IF
                IF (tank_flag .NE. 2) THEN
                    S_bu_bottom = (1._wp - temp) * S_bu_bottom_input(time_counter - 1) + temp * S_bu_bottom_input(time_counter)
                END IF
            END IF
````

## New implementation of precipitation
To make the use of precipitation more transparent, we added the option to give solid (precip_s) and liquid (precip_l) precipitation to the model. 
If liquid precipitation is given, precip_flag decides if the liquid precipitation stays liquid (precip_flag = 0) 
or if its phase is determined by T2m (precip_flag = 1).

## Boundflux Flag and Removal of Atmoflux Flag
The boundflux flag is arguably the most important flag, as it decides over the heat fluxes at the upper boundary (see Reference manual).
For boundflux_flag = 2, atmospheric heatfluxes are used to determine the surface temperature. In SAMSIM2.0, 
the atmoflux_flag then decided which atmospheric fluxes were used within the model. This is now changed in a way that the atmoflux_flag was removed and 
different atmospheric fluxes must be given in the input files. 

## Start from Inital Conditions
To enable model runs starting from an initial sea-ice profile, we added the flag initial_state_flag. If it is set to 2, initial 
values for S_abs, H_abs, thick and m are read in by sub_initial_input for the active layers. 
The total nuber and distribution of layers is defined in the config.json file (see build_config_initial_state).

## Diffenrences in the Model Output compared to SAMSIM2.0 
Compared to SAMSIM2.0, SAMSIM3.0 produces slightly different output, as seen in the plots below for each testcase. 
Some of those were introduced by the changes in the snow routine to remove the warnings (see above) and the rest is probably due to indexing errors, 
while reading the input data. The exact cause could not be identified but the differences are regarded small enough to  be negligible. 
If you can find the cause for the errors, contact me (jakob.deutloff@gmail.com) and I invite you for a drink ;). 

Differences testcase 1 (none):
![testcase1_diff](Plots\differeces_sam_final1.png)
Differences testcase 2:
![testcase2_diff](Plots\differeces_sam_final2.png)
Differences Testcase 3:
![testcase3_diff](Plots\differeces_sam_final3.png)
Differences Testcase 4:
![testcase4_diff](Plots\differeces_sam_final4.png)
## List of New Variables
- T2m_input - Input file for T2m
- T_top_input - Input file for T_top
- T_bottom_input - Input file for T_bottom
- fl_q_bottom_input - Input file for fl_q_bottom
- S_bu_bottom_input - Input file for S_bu_bottom
- precip_l_input - Input file for liquid_precip
- precip_s_input - Input file for solid_precip
- Tolerance - Tolerance for float comparisions
- initial_state_flag - Flag for initial state - 1: no initial state, 2: initial state 
- timestep_data - timestep of input data 
- T_in used to make copys of T and T_snow in mo_snow to avoid warnings 