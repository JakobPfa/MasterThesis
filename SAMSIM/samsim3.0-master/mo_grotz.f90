!>
!! The most important module of SAMSIM
!! 
!! The module mo_grotz contains the most important subroutine grotz (Named after GRiewank nOTZ).
!! Mo_grotz is called by SAMSIM.f90.
!! Subroutine grotz contains the time loop, as well as  the initialization, and calls all other branches of the model.
!! This model was developed from scratch by Philipp Griewank during and after his PhD at  Max Planck Institute of Meteorology from 2010-2014.
!! The code is intended to be understandable and most subroutines, modules, functions, parameters, and global variables have doxygen compatible descriptions. 
!! In addition to the doxygen generated description, some python plotscripts are available to plot model output.
!!
!! 
!!
!!
!! @author Philipp Griewank
!!
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
!!
!! @par Revision History
!! Started by Philipp Griewank 2012-08-28
!!

MODULE mo_grotz

CONTAINS

    !>
    !!
    !! Main subroutine of SAMSIM, a 1D thermodynamic seaice model.
    !! A semi-adaptive grid is used which is managed by mo_layer_dynamics.
    !!
    !! The basic rundown of the time loop is:
    !! 1. Calculate the current ice/snow state and forcing, as well as gravity drainage and flooding
    !! 2. Apply all the fluxes, recalculate ice state
    !! 3. Flushing and layer dynamics
    !!
    !! Here is the full rundown of what happens in mo_grotz:
    !!
    !! - Initialization: all fields are initialized for the given config.json file, and the output is formatted
    !! - Input and Forcing read in.
    !! TIME LOOP BEGINS:
    !!    - Calculate the total ice properties, total freshwater, thermal resistivity, energy, bulk salinity
    !!    - Determine snow and rain rates
    !!    - Calculate snow thermodynamics
    !!    - Calculate inner ice thermodynamic fluxes
    !!    - Calculate brine flux from expulsion
    !!    - Raw output written out if debug_flag is set to 2
    !!    - Standard output written
    !!    - Flooding parametrized
    !!    - Lowest layer mixing with underlying water
    !!    - Gravity drainage parametrized
    !!    - Calcuating and applying the heat fluxes
    !!    - After heatfluxes are applied new liquidus thermal equilibrium is calculated
    !!    - Flushing is parametrized
    !!    - Chemistry advection calculated
    !!    - Layer Dynamics
    !! TIME LOOP ENDS
    !! -Final output, files closed, and fields deallocated
    !!
    !!
    !! IMPORTANT:
    !! To get the correct freshwater amount make sure the freshwater is calculated using a salinity value to compare against.
    !!
    !!
    !! Common errors leading to termination are: too small timestep, bad programming
    !!
    !! @par Revision History
    !! Basic thermodynamics and layer_dynamics for fixed boundaries seem stable, backup made. by griewank (2010-08-10) \n
    !! Add some more outputs, changed routine names and arguments with respect to newly introduces flags by Niels Fuchs, MPIMET (2017-03-01) \n
    !! Added a bit of description with the run down of what happends by Philipp Griewank, Uni K (2018-08-08)
    SUBROUTINE grotz ()

        USE mo_parameters
        USE mo_thermo_functions
        USE mo_data
        USE mo_init
        USE mo_layer_dynamics
        USE mo_mass
        USE mo_grav_drain
        USE mo_output
        USE mo_flush
        USE mo_flood
        USE mo_snow
        USE mo_functions
        USE mo_heat_fluxes

        IMPLICIT NONE

        CHARACTER*12000 :: description   !< String to describes simulation which is output into dat_settings

        !Bastard variables, are used for various dirty deeds
        INTEGER :: jj
        REAL(wp) :: temp, temp2, temp4, temp5, temp_2017_H, temp_2017_m    !when a real is needed !< Niels, 2017 add: temp_2017_m
        temp5 = 0.




        !##########################################################################################
        !Initialization
        !##########################################################################################
        CALL init()

        ! Get description
        OPEN(unit = 99, file = 'Run_specifics/description.txt')
        READ(99, *) description
        CLOSE(99)

        CALL output_begin(Nlayer, debug_flag, format_T, format_psi, format_thick, format_snow, format_T2m_top, &
                & format_perm, format_melt)
        IF (bgc_flag==2) THEN
            CALL output_begin_bgc(Nlayer, N_bgc, format_bgc)
        END IF
        CALL output_settings(description, N_top, N_bottom, Nlayer, fl_q_bottom, T_bottom, S_bu_bottom, thick_0, time_out, &
                & time_total, dt, boundflux_flag, albedo_flag, grav_flag, flush_flag, flood_flag, &
                & grav_heat_flag, flush_heat_flag, &
                harmonic_flag, prescribe_flag, salt_flag, turb_flag, bottom_flag, tank_flag, precip_flag, bgc_flag, &
                & N_bgc, k_snow_flush)


        !##########################################################################################
        !Time Loop
        !##########################################################################################
        DO i = 1, i_time

            !        IF (n_time_out == i_time_out .OR. i==1) THEN
            !            WRITE(*,*), '0: Tsnow:', T_snow, ' so_prec:', solid_precip, ' T_Top:', T_top
            !        END IF

            !##########################################################################################
            !Vital signs. Calculates stored energy, freshwater column, thermal resistivity, ice thickness, and bulk salinity
            !##########################################################################################
            !stored energy
            energy_stored = H_abs_snow + SUM(H_abs(1:N_active)) - T_bottom * SUM(m(1:N_active)) * c_l

            !freshwater
            freshwater = SUM(m(1:N_active)) / rho_l
            freshwater = freshwater * (1._wp - SUM(S_abs(1:N_active)) / SUM(m(1:N_active)) / ref_salinity)
            freshwater = freshwater + m_snow / rho_l

            !Total resistivity only includes a fraction of the lowest layer.
            total_resist = 0._wp
            DO jj = 1, N_active - 1
                total_resist = total_resist + thick(jj) / (psi_l(jj) * k_l + psi_s(jj) * k_s)
            END DO
            total_resist = total_resist + thick(N_active) * psi_s(N_active) / psi_s_min * &
                    & (psi_s_min * k_s + 1._wp - psi_s_min * k_l)
            IF (thick_snow>thick_min / 110._wp) THEN
                total_resist = total_resist + thick_snow / func_k_snow(m_snow, thick_snow)
            END IF

            !Thickness is the thickness of all layers plus a fraction of the lowest layer
            IF (N_active>1) THEN
                thickness = SUM(thick(1:N_active - 1))
            ELSE
                thickness = 0.0
            END IF
            thickness = thickness + thick(N_active) * psi_s(N_active) / psi_s_min

            !Bulk salinity of the ice
            IF (N_active>1) THEN
                bulk_salin = SUM(S_abs(1:N_active - 1)) + S_abs(N_active) * psi_s(N_active) / psi_s_min
                bulk_salin = bulk_salin / (SUM(m(1:N_active - 1)) + m(N_active) * psi_s(N_active) / psi_s_min)
            ELSE
                bulk_salin = S_abs(1) / m(1)
            END IF



            !##########################################################################################
            !Selection and linear interpolation of boudary values
            !##########################################################################################

            IF (time>time_input(time_counter)) THEN   !
                time_counter = time_counter + 1
            END IF

            ! check if timestep for boundary values equals timestep of simulation
            ! If so, use boundary value at same timestep
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


            !##########################################################################################
            !Snow fall
            !##########################################################################################
            IF (precip_flag==1) THEN
                IF (MAX(liquid_precip, solid_precip)>0.0_wp .AND. N_Active>1) THEN      !Precip on/in snow layer
                    CALL snow_precip (m_snow, H_abs_snow, thick_snow, dt, liquid_precip, T2m)
                ELSE IF(MAX(liquid_precip, solid_precip)>0.0_wp .AND. N_Active==1) THEN !Precip in water layer
                    CALL snow_precip_0 (H_abs(1), S_abs(1), m(1), T(1), dt, liquid_precip, T2m)
                ELSE IF(MIN(liquid_precip, solid_precip)<0.0_wp .AND. thick_snow>0._wp .AND. N_Active>1) THEN      !Erosion of snow layer
                    CALL snow_erosion (m_snow, H_abs_snow, thick_snow, dt, liquid_precip, T2m)
                END IF

            ELSE IF (precip_flag==0) THEN
                IF (MAX(liquid_precip, solid_precip)>0.0_wp .AND. N_Active>1) THEN !Precip on/in snowlayer
                    CALL snow_precip (m_snow, H_abs_snow, thick_snow, dt, liquid_precip, T2m, solid_precip)
                    test = thick_snow
                ELSE IF (MAX(liquid_precip, solid_precip)>0.0_wp .AND. N_Active==1) THEN !Precip in water layer
                    CALL snow_precip_0 (H_abs(1), S_abs(1), m(1), T(1), dt, liquid_precip, T2m, solid_precip)
                ELSE IF (MAX(liquid_precip, solid_precip)>0.0_wp .AND. N_Active>1) THEN !Erosion of snowlayer
                    CALL snow_erosion (m_snow, H_abs_snow, thick_snow, dt, liquid_precip, T2m, solid_precip)
                END IF
            END IF


            !##########################################################################################
            !Snow thermodynamics and volume adjustment
            !##########################################################################################

            !< Niels, 2017 add: adjusted to snow_flush_flag==1
            IF (thick_snow>0.0_wp) THEN
                IF (snow_flush_flag == 0) THEN
                    CALL snow_thermo (psi_l_snow, psi_s_snow, psi_g_snow, thick_snow, S_abs_snow, &
                            & H_abs_snow, m_snow, T_snow, m(1), thick(1), H_abs(1))
                    melt_thick_snow = 0._wp
                ELSE IF (snow_flush_flag == 1) THEN
                    melt_thick_snow = 0._wp
                    CALL snow_thermo_meltwater (psi_l_snow, psi_s_snow, psi_g_snow, thick_snow, S_abs_snow, &
                            & H_abs_snow, m_snow, T_snow, m(1), thick(1), H_abs(1), melt_thick_snow)
                END IF
            ELSE
                thick_snow = 0.0_wp
                m_snow = 0.0_wp
                psi_s_snow = 0.0_wp
                psi_l_snow = 0.0_wp
                psi_g_snow = 0.0_wp
                H_abs_snow = 0.0_wp
                S_abs_snow = 0.0_wp
                melt_thick_snow = 0.0_wp  !< Niels, 2017 add: melt_thick_snow
            END IF

            exp_heat=exp_heat+SUM(H_abs(:))
            !##########################################################################################
            !Inner layer thermodynamics and Expulsion
            !##########################################################################################
            T_test = T_bottom
            DO k = N_active, 1, -1
                S_bu(k) = S_abs(k) / m(k)
                H(k) = H_abs(k) / m(k)
                CALL getT(H(k), S_bu(k), T_test, T(k), phi(k), k)
                T_test = T(k)
                S_br(k) = func_S_br(T(k), S_bu(k))
                !Expulsion
                CALL Expulsion(phi(k), thick(k), m(k), psi_s(k), psi_l(k), psi_g(k), V_ex(k))
            END DO

            !##########################################################################################
            !Brine flux due to Expulsion
            !##########################################################################################
            CALL expulsion_flux (thick, V_ex, Nlayer, N_active, psi_g, fl_m, m)
            IF (i .NE. 1) THEN !is disabled on first timestep to avoid expulsion due to unbalanced initial conditions
                !This ensures that the salinity is not effected if the initial mass of the layers is to high
                CALL mass_transfer  (Nlayer, N_active, T, H_abs, S_abs, S_bu, T_bottom, S_bu_bottom, fl_m)
                IF (bgc_flag ==2) THEN
                    DO k = 1, N_active
                        fl_brine_bgc(k, k + 1) = -fl_m(k + 1)
                    END DO
                END IF
            END IF
            exp_heat=exp_heat-SUM(H_abs(:))

            !##########################################################################################
            !Raw Output, if debug flag is set to 2 then all layer data is output
            !each timestep
            !##########################################################################################
            IF (debug_flag==2) THEN
                CALL output_raw(Nlayer, N_active, time, T, thick, S_bu, psi_s, psi_l, psi_g)
                CALL output_raw_snow(time, T_snow, thick_snow, S_abs_snow, m_snow, psi_s_snow, psi_l_snow, psi_g_snow)
            END IF

            DO k = N_active, 1, -1
                S_bu(k) = S_abs(k) / m(k)
            END DO

            !##########################################################################################
            !Standard output at each time_out
            !##########################################################################################
            IF (n_time_out == i_time_out .OR. i==1) THEN

                !Various things are calculated or recalculated before they are output
                IF (N_active>1) THEN
                    freeboard = func_freeboard(N_active, Nlayer, psi_s, psi_g, m, thick, m_snow, freeboard_snow_flag)
                ELSE
                    freeboard = 0._wp
                END IF

                IF (grav_flag == 2) then
                    IF (abs(grav_drain) < tolerance) THEN
                        grav_temp = 0._wp
                    ELSE
                        grav_temp = grav_temp / grav_drain
                    END IF
                    grav_salt = grav_salt / time_out
                    grav_drain = grav_drain / time_out
                end if

                grav_heat_flux_down = grav_heat_flux_down / time_out
                grav_heat_flux_up = grav_heat_flux_up / time_out
                exp_heat = exp_heat / time

                !Calling standard output
                CALL output(Nlayer, T, psi_s, psi_l, thick, S_bu, ray, format_T, format_psi, &
                        format_thick, format_snow, freeboard, thick_snow, T_snow, psi_l_snow, psi_s_snow, &
                        energy_stored, freshwater, total_resist, thickness, bulk_salin, &
                        grav_drain, grav_salt, grav_temp, T2m, T_top, perm, format_perm, flush_v, flush_h, psi_g, &
                        & melt_thick_output, grav_heat_flux_down, grav_heat_flux_up, exp_heat, format_melt)

                IF (bgc_flag==2) THEN
                    CALL output_bgc(Nlayer, N_active, bgc_bottom, N_bgc, bgc_abs, psi_l, thick, m, format_bgc)
                END IF


                !Printed to console
                WRITE(*, '(A10,I2,A15,F6.3,A12,F5.3,A14,F7.3,A30,F3.1,A14,F7.4,A10,F7.3,A7,F8.3)')&
                        'progress: ', INT(100._wp * (time + dt) / time_total), &
                        '%,  thickness: ', thickness, &
                        'm,  albedo: ', func_albedo(thick_snow, T_snow, psi_l(1), thick_min, albedo_flag), &
                        ',  surface T: ', T_top, &
                        'C,  thermal stability (<0.5): ', k_s * dt / rho_s / c_s / MIN(thick(1), thick_0)**2._wp, &
                        ',  snow_thick:', thick_snow, &
                        !',  snow gt zero: ', (thick_snow>0._wp),&   !< uncomment if you wish to know if there is a very thin snow layer on the ice
                        !',  Flush?:', flush_question,& !< tells you if flushing occurs in this iteration
                        ',  T_snow:', T_snow, &
                        ',  T2m:', T2m


                ! if you uncomment snow gt zero and flush, you need the following write statement:
                ! WRITE(*,'(A10,I2,A15,F4.3,A12,F5.3,A14,F7.3,A30,F3.1,A14,F5.4,A16,L1,A10,F7.3,A7,F7.3,A10,A3)')&
                flush_question = 'no!'

                ! Write other boundary data to console
                WRITE(*,*)&
                        'fl_q_bottom: ', fl_q_bottom, &
                        'T_bottom: ', T_bottom, &
                        'S_bu_bottom: ', s_bu_bottom, &
                        'T_top: ', T_top, &
                        'fl_sw: ', fl_sw, &
                        'fl_rest; ', fl_rest, &
                        ', Habs_top:', H_abs(2), &
                        ', Habs_bot:', H_abs(N_active)




                !Resetting gravity drainage things and time_out counter
                grav_drain = 0.0_wp
                grav_salt = 0.0_wp
                grav_temp = 0.0_wp
                exp_heat =0.0_wp
                grav_heat_flux_down = 0._wp
                grav_heat_flux_up = 0._wp
                melt_thick_output(:) = 0._wp

                n_time_out = 0
            ELSE
                n_time_out = n_time_out + 1
            END IF



            !##########################################################################################
            !When the lowest layer contains gas, it is replaced with ocean water
            !##########################################################################################
            IF (psi_g(N_active)>0.0_wp) THEN
                temp2 = psi_g(N_active) * thick(N_active) * rho_l
                m(N_active) = m(N_active) + temp2
                S_abs(N_active) = S_abs(N_active) + temp2 * S_bu_bottom
                H_abs(N_active) = H_abs(N_active) + temp2 * c_l * T_bottom
            END IF


            !##########################################################################################
            !Snow top layer coupling for thin snow
            !When snow is thinner than thick_min it is considered to be in thermal
            !equilibrium with the top ice layer
            !##########################################################################################
            IF (m_snow>0.0_wp .AND. thick_snow<thick_min)  THEN
                CALL snow_coupling (H_abs_snow, phi_s, T_snow, H_abs(1), H(1), phi(1), T(1), m_snow, S_abs_snow, m(1), &
                        & S_bu(1))
            END IF




            !##########################################################################################
            !Flooding
            !##########################################################################################
            IF (N_active>1 .AND. flood_flag>1) THEN
                freeboard = func_freeboard(N_active, Nlayer, psi_s, psi_g, m, thick, m_snow, freeboard_snow_flag)
                IF (freeboard<0.0_wp) THEN
                    IF (flood_flag==2) THEN
                        IF (bgc_flag == 2) THEN
                            CALL flood (freeboard, psi_s, psi_l, S_abs, H_abs, m, T, thick, dt, Nlayer, N_active, &
                                    &T_bottom, S_bu_bottom, H_abs_snow, m_snow, thick_snow, psi_g_snow, debug_flag,&
                                    & fl_brine_bgc)
                        ELSE
                            CALL flood (freeboard, psi_s, psi_l, S_abs, H_abs, m, T, thick, dt, Nlayer, N_active, &
                                    &T_bottom, S_bu_bottom, H_abs_snow, m_snow, thick_snow, psi_g_snow, debug_flag)

                        END IF
                    ELSE IF (flood_flag==3 .AND. freeboard<neg_free) THEN
                        CALL flood_simple (freeboard, S_abs, H_abs, m, thick, T_bottom, S_bu_bottom, H_abs_snow, &
                                &m_snow, thick_snow, psi_g_snow, Nlayer, N_active, debug_flag)
                    END IF
                END IF
            END IF

            !##########################################################################################
            !Turbulence, mixes lowest layer with underlying water
            !##########################################################################################
            IF (turb_flag==2) THEN
                IF (bgc_flag == 2) then
                    Call sub_turb_flux(T_bottom, S_bu_bottom, T(N_active), S_abs(N_active), m(N_active), dt, &
                            & N_bgc, bgc_bottom, bgc_abs(N_active, :))
                ELSE
                    Call sub_turb_flux(T_bottom, S_bu_bottom, T(N_active), S_abs(N_active), m(N_active), dt, N_bgc)
                end if
            END IF


            !##########################################################################################
            !Gravity drainage: grav_flag 2 complex, 3 simple
            !##########################################################################################
            IF (grav_flag==2 .AND. N_active>1) THEN
                IF (bgc_flag == 2) THEN
                    CALL fl_grav_drain (S_br, S_bu, psi_l, psi_s, thick, S_abs, H_abs, T, m, dt, Nlayer, N_active, &
                            &ray, T_bottom, S_bu_bottom, grav_drain, grav_temp, grav_salt, &
                            grav_heat_flag, harmonic_flag, grav_heat_flux_down, grav_heat_flux_up, fl_brine_bgc)
                ELSE
                    CALL fl_grav_drain (S_br, S_bu, psi_l, psi_s, thick, S_abs, H_abs, T, m, dt, Nlayer, N_active, &
                            &ray, T_bottom, S_bu_bottom, grav_drain, grav_temp, grav_salt, &
                            grav_heat_flag, harmonic_flag, grav_heat_flux_down, grav_heat_flux_up)
                END IF

            ELSE IF (grav_flag==3 .AND. N_active>1) THEN
                CALL fl_grav_drain_simple (psi_s, psi_l, thick, S_abs, S_br, Nlayer, N_active, ray, &
                        grav_drain, harmonic_flag)
            end if

            !##########################################################################################
            !Prescribes salinity profile for prescribe_flag = =2
            !##########################################################################################
            IF (prescribe_flag==2) THEN

                !Linear advance over the lowest 15 cm from S_bu_bottom to 4 ppt
                k = N_active
                DO WHILE (k>1 .AND. SUM(thick(k:N_active))<0.15_wp)
                    S_bu(k) = S_bu_bottom - SUM(thick(k:N_active)) / 0.15_wp * (S_bu_bottom - 4._wp)
                    k = k - 1
                END DO
                DO WHILE (k>1 .AND. SUM(thick(k:N_active))>=0.15_wp)
                    S_bu(k) = 4._wp - 4._wp * (SUM(thick(k:N_active)) - 0.15_wp) / (SUM(thick(1:N_active)) - 0.15_wp)
                    k = k - 1
                    S_bu(1) = 0._wp
                END DO
                S_bu(N_active) = S_bu_bottom
                S_abs = S_bu * m
            END IF



            !##########################################################################################
            !Calculating water concentrations if tank_flag == 2
            !##########################################################################################
            IF(tank_flag==2) then
                S_bu_bottom = (S_total - SUM(S_abs(:))) / (m_total - SUM(m))
                IF(bgc_flag==2) then
                    bgc_bottom = (bgc_total(1) - SUM(bgc_abs(:, 1))) / (m_total - SUM(m))
                end if
            end if

            !##########################################################################################
            !Calculating and applying heatfluxes
            !##########################################################################################

            call sub_heat_fluxes()



            !##########################################################################################
            !T and phi are recalculated before applying layer dynamics and flushing
            !##########################################################################################

            T_test = T_bottom
            DO k = N_active, 1, -1
                S_bu(k) = S_abs(k) / m(k)
                H(k) = H_abs(k) / m(k)
                CALL getT(H(k), S_bu(k), T_test, T(k), phi(k), k)
                T_test = T(k) !The temperature of the layer i is used to initialize the iteration for layer i-1
            END DO

            temp_2017_H = H_abs(1) + H_abs_snow + melt_thick_snow * rho_l * c_l * T_snow   !< Niels, 2017 add: checking energy and mass conservation later
            temp_2017_m = m(1) + m_snow + melt_thick_snow * rho_l !< Niels, 2017

            melt_thick_snow_old = melt_thick_snow    !< Niels, 2017 add: keep meltwater during recalculation of thermodynamics
            IF (thick_snow>0.0_wp) THEN
                IF (snow_flush_flag == 0) THEN
                    CALL snow_thermo (psi_l_snow, psi_s_snow, psi_g_snow, thick_snow, S_abs_snow, H_abs_snow, m_snow, &
                            & T_snow, m(1), thick(1), H_abs(1))
                    melt_thick_snow = 0._wp
                ELSE IF (snow_flush_flag == 1) THEN
                    melt_thick_snow = 0._wp
                    CALL snow_thermo_meltwater (psi_l_snow, psi_s_snow, psi_g_snow, thick_snow, S_abs_snow, H_abs_snow,&
                            & m_snow, T_snow, m(1), thick(1), H_abs(1), melt_thick_snow)
                END IF

            ELSE
                thick_snow = 0.0_wp
                m_snow = 0.0_wp
                psi_s_snow = 0.0_wp
                psi_l_snow = 0.0_wp
                psi_g_snow = 0.0_wp
                H_abs_snow = 0.0_wp
                S_abs_snow = 0.0_wp
                melt_thick_snow = 0.0_wp  !< Niels, 2017
            END IF
            melt_thick_snow = melt_thick_snow_old + melt_thick_snow !< Niels, 2017 add: keep meltwater during recalculation of thermodynamics


            !##########################################################################################
            !Flushing preparations
            !##########################################################################################
            !Subroutines are called to calculate the melt thickness for flush_flag greater 3,4,5,6 for boundflux_flag 2 and 3
            IF (N_active>1 .AND. flush_flag>2) THEN
                IF (boundflux_flag ==2) THEN
                    T_freeze = func_T_freeze(S_abs(1) / m(1), salt_flag)
                    melt_thick = 0.0_wp
                    IF (func_freeboard(N_active, Nlayer, psi_s, psi_g, m, thick, m_snow, freeboard_snow_flag)> &
                            & 0.0000000000001_wp) THEN
                        IF (psi_s(1)<psi_s_top_min .OR. T_top>=T_freeze) THEN

                            CALL sub_melt_thick(psi_l(1), psi_s(1), psi_g(1), T(1), T_freeze, T_top, fl_Q(1), &
                                    & thick_snow, dt, melt_thick, thick(1), thick_min)
                            IF (thick_snow>=thick_min / 100._wp .AND. melt_thick>0.00000000001_wp .AND. &
                                    & abs(melt_thick_snow) < tolerance) THEN
                                !< Niels, 2017 add: .and.  melt_thick_snow == 0.
                                CALL sub_melt_snow(melt_thick, thick(1), thick_snow, H_abs(1), H_abs_snow, m(1), &
                                        & m_snow, psi_g_snow)
                            END IF
                        END IF
                    END IF
                END IF

                IF (boundflux_flag==3) THEN
                    T_freeze = func_T_freeze(S_abs(1) / m(1), salt_flag)
                    melt_thick = 0.0_wp
                    !< Niels, 2017 add: freeboard_snow_flag, import parameter for lab experiments, see in definition
                    IF (func_freeboard(N_active, Nlayer, psi_s, psi_g, m, thick, m_snow, freeboard_snow_flag)> &
                            & 0.0000000000001_wp) THEN
                        IF (psi_s(1)<psi_s_top_min .OR. T2m>=T_freeze) THEN
                            CALL sub_melt_thick(psi_l(1), psi_s(1), psi_g(1), T(1), T_freeze, T2m, fl_Q(1), thick_snow,&
                                    & dt, melt_thick, thick(1), &
                                    &thick_min)
                            melt_thick = max(melt_thick, 0._wp) !< Niels, 2017 add: just for the case
                            IF (thick_snow>=thick_min / 100._wp .AND. melt_thick>0.00000000001_wp .AND. &
                                    & abs(melt_thick_snow) < tolerance) THEN
                                CALL sub_melt_snow(melt_thick, thick(1), thick_snow, H_abs(1), H_abs_snow, m(1), &
                                        & m_snow, psi_g_snow)
                            END IF
                        END IF
                    END IF
                END IF
            END IF


            !##########################################################################################
            !Flushing
            !##########################################################################################
            freeboard = func_freeboard(N_active, Nlayer, psi_s, psi_g, m, thick, m_snow, freeboard_snow_flag)

            melt_thick_output(1) = melt_thick_output(1) + melt_thick !< Niels, 2017 add: for later evaluation: accumulated melt_thick
            melt_thick_output(2) = melt_thick_output(2) + melt_thick_snow !< Niels, 2017 add: accumulated melt_thick_snow

            melt_thick = melt_thick + melt_thick_snow    !< Niels, 2017 add: Ice melt plus snow melt

            IF (melt_thick_snow > 0._wp) THEN
                !< Niels, 2017 add: Add snow melt water to upper ice layer to keep balances
                H_abs(1) = H_abs(1) + melt_thick_snow * rho_l * c_l * T_snow
                S_abs(1) = S_abs(1) + melt_thick_snow * rho_l * func_S_br(T_snow, S_abs_snow / m_snow)
                thick(1) = thick(1) + melt_thick_snow
                m(1) = m(1) + melt_thick_snow * rho_l
                S_bu(1) = S_abs(1) / m(1)
                H(1) = H_abs(1) / m(1)
            END IF

            !< Niels, 2017 add: energy and mass balance test
            IF (temp_2017_H - (H_abs(1) + H_abs_snow) > 0.00000001_wp)  THEN
                print*, i, ' enthalpy problem during snow melt:', temp_2017_H, H_abs(1) + H_abs_snow
            ELSE IF (temp_2017_m - (m(1) + m_snow) > 0.00000001_wp) THEN
                print*, i, 'mass problem during snow melt:', temp_2017_m, m(1) + m_snow
            END IF

            ! Start flushing routines

            flush_v_old(:) = flush_v(:)  !< Niels, 2017 add: used for accumulated flush output
            flush_h_old(:) = flush_h(:)

            flush_v(:) = 0._wp
            flush_h(:) = 0._wp

            IF (N_active>1 .and. freeboard>0.001_wp) THEN
                IF (flush_flag==4) THEN
                    IF (melt_thick>0.000000000001_wp .AND. N_active>2) THEN
                        H_abs(1) = H_abs(1) - melt_thick * rho_l * c_l * T(1)
                        S_abs(1) = S_abs(1) * (1._wp - (melt_thick * rho_l) / m(1))
                        thick(1) = thick(1) - melt_thick
                        m(1) = m(1) - melt_thick * rho_l
                        IF (S_abs(1)<0.0_wp) THEN
                            PRINT*, 'sorry bro, but you got a problem', melt_thick, psi_l * thick(1)
                        END IF
                    END IF

                ELSE IF (flush_flag==5) THEN
                    IF (melt_thick>0.000000000001_wp .AND. N_active>2 .AND. freeboard>0.0_wp) THEN
                        freeboard = func_freeboard(N_active, Nlayer, psi_s, psi_g, m, thick, m_snow, freeboard_snow_flag)
                        flush_question = 'yes'  !< Niels, 2017 add: used for stdout
                        IF (bgc_flag == 2) THEN
                            CALL flush3(freeboard, psi_l, thick, thick_0, S_abs, H_abs, m, T, dt, Nlayer, N_active, &
                                    &T_bottom, S_bu_bottom, melt_thick, debug_flag, flush_heat_flag, melt_err, perm, &
                                    & flush_v, flush_h, psi_g, rho_l, snow_flush_flag, fl_brine_bgc)   !< Niels, 2017 add: increased number of arguments
                        ELSE
                            CALL flush3(freeboard, psi_l, thick, thick_0, S_abs, H_abs, m, T, dt, Nlayer, N_active, &
                                    &T_bottom, S_bu_bottom, melt_thick, debug_flag, flush_heat_flag, melt_err, perm, &
                                    & flush_v, flush_h, psi_g, rho_l, snow_flush_flag)    !< Niels, 2017 add: increased number of arguments
                        END IF
                    END IF
                ELSE IF (flush_flag==6) THEN
                    IF (melt_thick>0.000000000001_wp .AND. N_active>2 .AND. thick_snow<thick_0) THEN
                        CALL flush4 (psi_l, thick, T, S_abs, H_abs, m, Nlayer, N_active, melt_thick, debug_flag)
                    END IF
                END IF
            END IF

            flush_v(:) = flush_v(:) + flush_v_old(:) !< Niels, 2017 add: for output
            flush_h(:) = flush_h(:) + flush_h_old(:)

            !##########################################################################################
            !BGC advection
            !##########################################################################################
            IF (bgc_flag == 2) THEN

                CALL bgc_advection (Nlayer, N_active, N_bgc, fl_brine_bgc, bgc_abs, psi_l, thick, bgc_bottom)
                fl_brine_bgc = 0._wp

            END IF

            temp4 = sum(H_abs(:))

            !##########################################################################################
            !Layer Dynamics
            !##########################################################################################
            IF (N_active>1) THEN
                !##########################################################################################
                !Bottom & top  growth or melt
                !##########################################################################################
                IF (phi(N_active)> psi_s_min .OR. phi(N_active - 1)<=psi_s_min / 2._wp .OR. thick(1) / thick_0>1.5_wp &
                        &.OR. thick(1) / thick_0<0.5_wp) THEN
                    IF (bgc_flag == 2) THEN
                        CALL layer_dynamics(phi, N_active, Nlayer, N_bottom, N_middle, N_top, m, S_abs, H_abs, thick, &
                                & thick_0, T_bottom, S_bu_bottom, bottom_flag, debug_flag, melt_thick_output(3), &
                                & N_bgc, bgc_abs, bgc_bottom)
                    ELSE
                        CALL layer_dynamics(phi, N_active, Nlayer, N_bottom, N_middle, N_top, m, S_abs, H_abs, thick, &
                                & thick_0, T_bottom, S_bu_bottom, bottom_flag, debug_flag, melt_thick_output(3), N_bgc)
                    END IF
                END IF


                !If the lowest layer was deactivated, its remaining values are scrubbed.
                IF (N_active<Nlayer .AND. abs(thick(MIN(N_active + 1, Nlayer))) < tolerance) THEN
                    T(N_active + 1) = T_bottom
                    S_bu(N_active + 1) = S_bu_bottom
                    H(N_active + 1) = 0.0_wp
                    psi_l(N_active + 1) = 1.0_wp
                    psi_s(N_active + 1) = 0.0_wp
                    IF (bgc_flag==2) THEN
                        bgc_abs(N_active + 1, :) = 0._wp
                    END IF

                END IF

            ELSE  !If Ice appears in the only active surface layer
                IF (phi(1).GT.psi_s_min) THEN
                    IF (bgc_flag == 2) THEN
                        CALL layer_dynamics(phi, N_active, Nlayer, N_bottom, N_middle, N_top, m, S_abs, H_abs, thick, &
                                & thick_0, T_bottom, S_bu_bottom, bottom_flag, debug_flag, melt_thick_output(3),  &
                                & N_bgc, bgc_abs, bgc_bottom)
                    ELSE
                        CALL layer_dynamics(phi, N_active, Nlayer, N_bottom, N_middle, N_top, m, S_abs, H_abs, thick, &
                                & thick_0, T_bottom, S_bu_bottom, bottom_flag, debug_flag, melt_thick_output(3), N_bgc)
                    END IF
                END IF
            END IF

            temp5 = temp5 + (sum(H_abs(:)) - temp4)!/dt-fl_q(N_active+1)+fl_q(1)

            !##########################################################################################
            !timestep
            !##########################################################################################
            time = time + dt


            !##########################################################################################
            !Health check
            !##########################################################################################
            IF (MINVAL(psi_s(1:N_active))<0.0_wp) THEN
                PRINT*, 'negative solid fraction, aborted'
                PRINT*, 'wtfrho', m(1), thick(1)
                STOP 1337
            ELSE IF(MINVAL(S_abs(1:N_active))<0.0_wp) THEN
                PRINT*, 'negative salinity, aborted'
                PRINT*, 'wtfsalt', S_abs
                FORALL (k = 1:N_active)
                    S_abs(k) = MAX(S_abs(k), 0._wp)
                END FORALL
                !STOP 1337
            END IF









            !##########################################################################################
            !END OF Time LOOP
            !##########################################################################################

        END DO

        !##########################################################################################
        !Final Output
        !##########################################################################################
        WRITE(*, *)'Run completed, total ice thickness at end of run:', SUM(thick(1:N_active - 1)), &
                & ' melt_err= ', melt_err



        !Closing output files
        IF (debug_flag==2) THEN
            DO k = 1, Nlayer
                CLOSE(k + 100)
            END DO
        END IF
        IF (bgc_flag==2) THEN
            DO k = 1, N_bgc
                CLOSE(2 * k + 400)
                CLOSE(2 * k + 401)
            END DO
        END IF
        CLOSE(30)
        CLOSE(31)
        CLOSE(32)
        CLOSE(33)
        CLOSE(34)
        CLOSE(35)
        CLOSE(40)
        CLOSE(41)
        CLOSE(42)
        CLOSE(43)
        CLOSE(44)
        CLOSE(45)
        CLOSE(46)   !< Niels, 2017 add:46-50 for some output files
        CLOSE(47)
        CLOSE(48)
        CLOSE(49)
        CLOSE(50)
        CLOSE(51)
        CLOSE(52)
        CLOSE(66)

        !Deallocating arrays
        CALL sub_deallocate
    END SUBROUTINE grotz

END MODULE mo_grotz
