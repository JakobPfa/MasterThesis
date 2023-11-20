!>
!! Allocates Arrays and sets initial data for SAMSIM
!! All informations are read in from the config.json file which is generated in python and the input files.
!!
!!
!! @author Philipp Griewank
!!
!!  COPYRIGHT
!!
!! This file is part of SAMSIM.
!!
!!  SAMSIM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
!!
!!  SAMSIM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
!!
!!  You should have received a copy of the GNU General Public License along with SAMSIM. If not, see <http://www.gnu.org/licenses/>.
!
!!
!!
!! @par Revision History
!! first version created to deal with first multi-layer tests. by Philipp Griewank, IMPRS (2010-07-22)
!! Add Testcases: 101-105 are simulations of master theses experiments 1-5, 111 can be used to compare SAMSIM with salinity harps field data by Niels Fuchs, MPIMET (2017-03-01)
!!
MODULE mo_init

    USE mo_parameters
    USE mo_data
    USE json_module
    USE mo_functions

    IMPLICIT NONE


CONTAINS
    !>
    !! Sets initial conditions according to what is defined in the config.json file.
    !!
    !! For different initial conditions the Arrays are allocated and the initial values are set.
    !! Following must always be:
    !! 1. Nlayer = N_top+N_middle+N_bottom
    !! 2. N_active is set correctly, N_active <= Nlayer
    !! 3. fl_q_bottom >= 0
    !! 4. T_bottom > freezing point of for S_bu_bottom
    !! 5. A too high dt for a too small thick_0 leads to numerical thermodynamic instability. For a conservative guess dt [s] should be smaller than 250000 * (dz [m])**2
    !!

    !!
    !!
    !! @par Revision History
    !! First set up by Philipp Griewank, IMPRS (2010-07-22>)
    SUBROUTINE init ()

        TYPE(json_file) :: json
        LOGICAL :: is_found

        ! Initialise the json_file object.
        CALL json%initialize()

        ! Load the file.
        CALL json%load_file('./Run_specifics/config.json'); IF (json%failed()) STOP "config file could not be loaded"


        !##########################################################################################
        !Here I just set a lot of things to zero, arrays are set to zero after they are allocated
        !##########################################################################################
        thick_snow = 0.0_wp
        m_snow = 0.0_wp
        psi_g_snow = 0.0_wp
        psi_l_snow = 0.0_wp
        psi_s_snow = 0.0_wp
        H_abs_snow = 0.0_wp
        S_abs_snow = 0.0_wp
        phi_s = 0.0_wp
        T_snow = 0.0_wp
        liquid_precip = 0.0_wp
        solid_precip = 0.0_wp
        fl_sw = 0.0_wp
        fl_rest = 0.0_wp
        albedo = 0.0_wp
        T_top = 0.0_wp
        T2m = 0.0_wp
        fl_q_bottom = 0.0_wp


        !*************************************************************************************************************************
        !Initial layer thickness, timestep, output time and simulation time
        !*************************************************************************************************************************
        CALL json%get('dt', dt, is_found)
        IF (.not. is_found) THEN; PRINT*, 'dt not found'; STOP; END IF
        CALL json%get('time', time, is_found)
        IF (.not. is_found) THEN; PRINT*, 'time not found'; STOP; END IF
        CALL json%get('time_out', time_out, is_found)
        IF (.not. is_found) THEN; PRINT*, 'time_out not found'; STOP; END IF
        CALL json%get('time_total', time_total, is_found)
        IF (.not. is_found) THEN; PRINT*, 'time_total not found'; STOP; END IF

        !*************************************************************************************************************************
        !Time settings needed when input is given
        !*************************************************************************************************************************
        CALL json%get('length_input', length_input, is_found)
        IF (.not. is_found) THEN; PRINT*, 'lenght_input not found'; STOP; END IF
        CALL json%get('timestep_data', timestep_data, is_found)
        IF (.not. is_found) THEN; PRINT*, 'timestep_data not found'; STOP; END IF

        !*************************************************************************************************************************
        !Layer settings and allocation
        !*************************************************************************************************************************
        CALL json%get('thick_0', thick_0, is_found)
        IF (.not. is_found) THEN; PRINT*, 'thick_0 not found'; STOP; END IF
        CALL json%get('Nlayer', Nlayer, is_found)
        IF (.not. is_found) THEN; PRINT*, 'Nlayer not found'; STOP; END IF
        CALL json%get('N_active', N_active, is_found)
        IF (.not. is_found) THEN; PRINT*, 'N_active not found'; STOP; END IF
        CALL json%get('N_top', N_top, is_found)
        IF (.not. is_found) THEN; PRINT*, 'N_top not found'; STOP; END IF
        CALL json%get('N_bottom', N_bottom, is_found)
        IF (.not. is_found) THEN; PRINT*, 'N_bottom not found'; STOP; END IF
        N_middle = Nlayer - N_top - N_bottom
        CALL sub_allocate(Nlayer)

        !*************************************************************************************************************************
        !Flags
        !*************************************************************************************************************************
        !________________________top heat flux____________
        CALL json%get('boundflux_flag', boundflux_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'boundflux_flag not found'; STOP; END IF
        CALL json%get('albedo_flag', albedo_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'albedo_flag not found'; STOP; END IF
        !________________________brine_dynamics____________
        CALL json%get('grav_heat_flag', grav_heat_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'grav_heat_flag not found'; STOP; END IF
        CALL json%get('flush_heat_flag', flush_heat_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'flush_heat_flag not found'; STOP; END IF
        CALL json%get('flood_flag', flood_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'flood_flag not found'; STOP; END IF
        CALL json%get('flush_flag', flush_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'flush_flag not found'; STOP; END IF
        CALL json%get('grav_flag', grav_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'grav_flag not found'; STOP; END IF
        CALL json%get('harmonic_flag', harmonic_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'harmonic_flag not found'; STOP; END IF
        !________________________Salinity____________
        CALL json%get('prescribe_flag', prescribe_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'prescribe_flag not found'; STOP; END IF
        CALL json%get('salt_flag', salt_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'salt_flag not found'; STOP; END IF
        !________________________bottom setting______________________
        CALL json%get('turb_flag', turb_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'turb_flag not found'; STOP; END IF
        CALL json%get('bottom_flag', bottom_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'bottom_flag not found'; STOP; END IF
        CALL json%get('tank_flag', tank_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'tank_flag not found'; STOP; END IF
        !________________________snow______________________
        CALL json%get('precip_flag', precip_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'precip_flag not found'; STOP; END IF
        CALL json%get('freeboard_snow_flag', freeboard_snow_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'freeboard_snow_flag not found'; STOP; END IF
        CALL json%get('snow_flush_flag', snow_flush_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'snow_flush_flag not found'; STOP; END IF
        CALL json%get('styropor_flag', styropor_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'styropor_flag not found'; STOP; END IF
        CALL json%get('lab_snow_flag', lab_snow_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'lab_snow_flag not found'; STOP; END IF
        !________________________debugging_____________________
        CALL json%get('debug_flag', debug_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'debug_flag not found'; STOP; END IF
        !________________________bgc_______________________
        CALL json%get('bgc_flag', bgc_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'bcg_flag not found'; STOP; END IF
        CALL json%get('N_bgc', N_bgc, is_found)
        IF (.not. is_found) THEN; PRINT*, 'N_bcg not found'; STOP; END IF
        !________________________initial state_______________________
        CALL json%get('initial_state_flag', initial_state_flag, is_found)
        IF (.not. is_found) THEN; PRINT*, 'initial_state_flag not found'; STOP; END IF

        !##########################################################################################
        !Input/Forcing are read in
        !##########################################################################################
        time_counter = 1
        CALL sub_input(length_input, timestep_data, T_top_input, T2m_input, T_bottom_input, S_bu_bottom_input, &
                & fl_q_bottom_input, time_input,fl_sw_input, fl_lw_input, fl_lat_input, fl_sen_input, precip_l_input, &
                & precip_s_input, boundflux_flag, precip_flag)

        ! Unpack first values of inputs used for each boundflux_flag. The initial values are used to initialize the sea ice
        ! which is done before they are read in in the first tie step in mo_grotz, which is why we unpack them here.
        T_bottom = T_bottom_input(1)
        S_bu_bottom = S_bu_bottom_input(1)
        fl_q_bottom = fl_q_bottom_input(1)

        IF (boundflux_flag ==1) THEN
            T_top = T_top_input(1)
        ELSE IF (boundflux_flag == 2 ) THEN
            T2m = T2m_input(1)
        ELSE IF (boundflux_flag == 3) THEN
            T_top = T_top_input(1)
            T2m = T2m_input(1)
        END IF

        !*************************************************************************************************************************
        !Tank and turbulent fluxes settings
        !*************************************************************************************************************************
        CALL json%get('alpha_flux_instable', alpha_flux_instable, is_found)
        IF (.not. is_found) THEN; PRINT*, 'alpha_flux_instable'; STOP; END IF
        CALL json%get('alpha_flux_stable', alpha_flux_stable, is_found)
        IF (.not. is_found) THEN; PRINT*, 'alpha_flux_stable not found'; STOP; END IF
        CALL json%get('tank_depth', tank_depth, is_found)
        IF (.not. is_found) THEN; PRINT*, 'tank_depth not found'; STOP; END IF
        IF (tank_flag == 2) THEN
            m_total = rho_l * tank_depth             !Multiply with tank depth in meters
            S_total = rho_l * S_bu_bottom_input(1) * tank_depth
        END IF

        !*************************************************************************************************************************
        !setting the initial values of the top and only layer in case no initial state is given
        !*************************************************************************************************************************
        IF (initial_state_flag .NE. 2) THEN
            CALL json%get('thick_1', thick(1), is_found)
            IF (.not. is_found) THEN; PRINT*, 'thick_1 not found'; STOP; END IF
            CALL json%get('m_1', m(1), is_found)
            IF (.not. is_found) THEN; PRINT*, 'm_1 not found'; STOP; END IF
            CALL json%get('S_abs_1', S_abs(1), is_found)
            IF (.not. is_found) THEN; PRINT*, 'S_abs not found'; STOP; END IF
            CALL json%get('H_abs_1', H_abs(1), is_found)
            IF (.not. is_found) THEN; PRINT*, 'H_abs not found'; STOP; END IF
        END IF

        !*************************************************************************************************************************
        ! Read initial state in case it is given
        !**************************************************************************************************************************
        IF (initial_state_flag == 2) THEN
            CALL sub_initial_input(H_abs, S_abs, thick, m, N_active, Nlayer)
        END IF

        !*************************************************************************************************************************
        !BGC settings, only relevant if bgc_flag is set to 2
        !*************************************************************************************************************************
        IF (bgc_flag==2) THEN
            !*************************************************************************************************************************
            !Setting number of tracers
            !*************************************************************************************************************************
            CALL json%get('N_bgc', N_bgc, is_found)
            IF (.not. is_found) THEN; PRINT*, 'N_bcg not found'; STOP; END IF
            CALL sub_allocate_bgc(Nlayer, N_bgc)

            !*************************************************************************************************************************
            !Setting bottom concentrations
            !*************************************************************************************************************************
            CALL json%get('bgc_bottom_1', bgc_bottom(1), is_found)
            IF (.not. is_found) THEN; PRINT*, 'bgc_bottom_1 not found'; STOP; END IF
            CALL json%get('bgc_bottom_2', bgc_bottom(2), is_found)
            IF (.not. is_found) THEN;PRINT*, 'bgc_bottom_2 not found'; STOP; END IF

            !*************************************************************************************************************************
            !setting the initial values of the top and only layer
            !*************************************************************************************************************************
            bgc_abs(1, :) = bgc_bottom(:) * m(1)
            bgc_bu (1, :) = bgc_bottom(:)
            bgc_br (1, :) = bgc_bottom(:)
        end if



        !Varius default settings
        IF (initial_state_flag .NE. 2) THEN
            T = T_bottom
            thickness = 0._wp
            S_bu = S_bu_bottom
        END IF

        psi_s = 0._wp
        ray = 0.0_wp
        phi = 0.0_wp
        psi_l = 1.0_wp
        fl_rad = 0.0_wp
        bulk_salin = 0._wp
        grav_salt = 0._wp
        grav_drain = 0._wp

        thick_min = thick_0 / 2._wp
        melt_thick_output(:) = 0._wp !< Niels, 2017

        !Calculates the number of timesteps for given time_total and dt as well as
        !timesteps between output
        i_time = INT(time_total / dt)
        i_time_out = INT(time_out / dt)
        n_time_out = 0

        grav_drain = 0._wp
        grav_salt = 0._wp
        grav_temp = 0._wp
        melt_thick = 0
        bulk_salin = SUM(S_abs(1:N_active)) / SUM(m(1:N_active))



        !Small sanity checks
        IF(N_top<3) THEN
            PRINT*, 'Problem occurs when N_top smaller then 3, so just change it to 3 or more'
            STOP 666
        END IF

        IF(bgc_flag==2 .AND. (grav_flag + flush_flag + flood_flag).NE.9) THEN
            PRINT*, 'WARNING: Biogeochemistry is on, but none or not all complex brine parametrizations &
                    & are  activated. Make sure this is your intent'
        END IF

        IF (tank_flag==2 .and. abs(tank_depth) < tolerance) THEN
            PRINT*, 'Tank_flag 2 used but tank depth not defined or set to zero'
            STOP 667
        END IF


    END SUBROUTINE init

    !>
    !! Allocates Arrays.
    !!
    !! For a given number of layers Nlayers all arrays are allocated
    SUBROUTINE sub_allocate (Nlayer, length_input_lab)

        INTEGER, INTENT(in) :: Nlayer  !<  number of layers
        INTEGER, INTENT(in), OPTIONAL :: length_input_lab  !< Niels, 2017 add:  dimension of input arrays

        !allocated Nlayer

        ALLOCATE(H(Nlayer), H_abs(Nlayer))
        ALLOCATE(Q(Nlayer))
        ALLOCATE(T(Nlayer))
        ALLOCATE(S_abs(Nlayer), S_bu(Nlayer), S_br(Nlayer))
        ALLOCATE(thick(Nlayer))
        ALLOCATE(m(Nlayer))
        ALLOCATE(V_s(Nlayer), V_l(Nlayer), V_g(Nlayer))
        ALLOCATE(V_ex(Nlayer))
        ALLOCATE(phi(Nlayer))
        ALLOCATE(perm(Nlayer))
        ALLOCATE(flush_v(Nlayer))   !< Niels, 2017
        ALLOCATE(flush_h(Nlayer))   !< Niels, 2017
        ALLOCATE(flush_v_old(Nlayer))   !< Niels, 2017
        ALLOCATE(flush_h_old(Nlayer))   !< Niels, 2017
        ALLOCATE(psi_s(Nlayer), psi_l(Nlayer), psi_g(Nlayer))
        ALLOCATE(fl_rad(Nlayer))
        IF (present(length_input_lab)) THEN
            ALLOCATE(Tinput(length_input_lab))   !< Niels, 2017
            !ALLOCATE(precipinput(length_input_lab))  !< Niels, 2017
            ALLOCATE(ocean_T_input(length_input_lab))    !< Niels, 2017
            ALLOCATE(ocean_flux_input(length_input_lab)) !< Niels, 2017
            ALLOCATE(styropor_input(length_input_lab))   !< Niels, 2017
            ALLOCATE(Ttop_input(length_input_lab))   !< Niels, 2017
        END IF

        !allocated Nlayer+1

        ALLOCATE(fl_Q(Nlayer + 1))
        ALLOCATE(fl_m(Nlayer + 1))
        ALLOCATE(fl_s(Nlayer + 1))


        !Allocate Nlayer-1
        ALLOCATE(ray(Nlayer - 1))

        m = 0._wp
        S_abs = 0._wp
        H_abs = 0._wp
        thick = 0._wp

        flush_v(:) = 0._wp   !< Niels, 2017
        flush_h(:) = 0._wp   !< Niels, 2017

    END SUBROUTINE sub_allocate

    !>
    !! Allocates BGC Arrays.
    !!
    SUBROUTINE sub_allocate_bgc (Nlayer, N_bgc)

        INTEGER, INTENT(in) :: Nlayer, N_bgc

        !Brine flux matrix
        ALLOCATE(fl_brine_bgc(Nlayer + 1, Nlayer + 1))
        !Chemical matrices
        ALLOCATE(bgc_abs(Nlayer, N_bgc), bgc_bu(Nlayer, N_bgc), bgc_br(Nlayer, N_bgc))
        !Bottom values
        ALLOCATE(bgc_bottom(N_bgc), bgc_total(N_bgc))

        bgc_abs = 0.0_wp
        fl_brine_bgc = 0.0_wp
        bgc_bu = 0.0_wp
        bgc_br = 0.0_wp
        bgc_bottom = 0.0_wp
        bgc_total = 0.0_wp

    END SUBROUTINE sub_allocate_bgc


    !>
    !! Deallocates Arrays.
    !!
    SUBROUTINE sub_deallocate
        DEALLOCATE(H, H_abs)
        DEALLOCATE(Q)
        DEALLOCATE(T)
        DEALLOCATE(S_abs, S_bu, S_br)
        DEALLOCATE(thick)
        DEALLOCATE(m)
        DEALLOCATE(V_s, V_l, V_g)
        DEALLOCATE(V_ex)
        DEALLOCATE(phi)
        DEALLOCATE(perm)
        DEALLOCATE(psi_s, psi_l, psi_g)
        DEALLOCATE(fl_rad)

        !allocated Nlayer+1
        DEALLOCATE(fl_Q)
        DEALLOCATE(fl_m)
        DEALLOCATE(fl_s)

        !Allocate Nlayer-1
        DEALLOCATE(ray)

    END SUBROUTINE sub_deallocate

END MODULE mo_init

