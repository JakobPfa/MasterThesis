############################################################################################################
#Three compiler options
#-Nagfor is usefull for finding bugs, but can't compile minpack which is necesarry for optimization runs
#-Ifort is fast
#-Gfortran is free!
############################################################################################################
#Choose your compiler by commenting it out with #
############################################################################################################

#objects = mo_kind.o mo_parameters.o mo_data.o mo_functions.o mo_init.o mo_thermo_functions.o mo_mass.o mo_grav_drain.o mo_output.o mo_layer_dynamics.o mo_flush.o mo_snow.o mo_flood.o mo_heat_fluxes.o mo_testcase_specifics.o mo_grotz.o
objects = mo_parameters.o mo_data.o mo_functions.o mo_init.o mo_thermo_functions.o mo_mass.o mo_grav_drain.o mo_output.o mo_layer_dynamics.o mo_flush.o mo_snow.o mo_flood.o mo_heat_fluxes.o mo_grotz.o

#Nagfor
#Comp		  = nagfor
#FLAGS	  = -C=all -gline -nan -g


#ifort
#Comp		  = ifort
#FLAGS		 = -fast 

#gfortran
Comp		  = gfortran
FLAGS		 = -I./json-fortran-master/lib -Wall -Wextra -fbounds-check



## Compile the Fortran code and create a samsim.x executable
samsim.x : SAMSIM.f90 $(objects)
	$(Comp) ${FLAGS} -o $@   $(objects) SAMSIM.f90 ./json-fortran-master/lib/libjsonfortran.a


%.o : %.f90
	$(Comp) ${FLAGS} -c  $<


## Remove temporary files
clean:
	rm  *o *mod 


## Create a first config.json file
config.json:
	python3 Python_files/Code/build_config_files/build_config_1.py

## Plot T, Phi, S timeseries
plot_TPhiS:
	python3 Python_files/Code/plotscripts/plot_TPhiS.py


# ==================== Don't put anything below this line ====================
.DEFAULT_GOAL := show-help
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)";echo;sed -ne"/^## /{h;s/.*//;:d" -e"H;n;s/^## //;td" -e"s/:.*//;G;s/\\n## /---/;s/\\n/ /g;p;}" ${MAKEFILE_LIST}|LC_ALL='C' sort -f|awk -F --- -v n=$$(tput cols) -v i=19 -v a="$$(tput setaf 6)" -v z="$$(tput sgr0)" '{printf"%s%*s%s ",a,-i,$$1,z;m=split($$2,w," ");l=n-i;for(j=1;j<=m;j++){l-=length(w[j])+1;if(l<= 0){l=n-i-length(w[j])-1;printf"\n%*s ",-i," ";}printf"%s ",w[j];}printf"\n";}'|more
