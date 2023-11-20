"""
This file contains the functions used to build the bundary values used in SAMSIM for testcases 1-4.
The boundary values were computed in mo_testcase_specifics in SAMSIM1.0 and SAMSIM2.0

Created by Jakob Deutloff (jakob.deutloff@gmail.com)
"""

import numpy as np


def Tstep(T1, T2, time_total, dt):

    T_top = np.ones(int(time_total / dt))
    for i in range(int(T_top.size/2)):
        T_top[i] = T1
    for i in range(int(T_top.size/2), T_top.size):
        T_top[i] = T2

    return T_top

def t_top_testcase_1(time_total, dt):
    """
    Function to build array of surface temperatures with one value every timestep dt.
    Developed to construct surface temperatures used in testcase 1.
    :param time_total: total time of simulation from config file
    :param dt: timestep dt from config file
    :return: T_top: surface temperature as used in testcase one in SAMSIM1.0/2.0
    """
    T_top = np.ones(int(time_total / dt))
    j = 0
    for i in range(T_top.size):

        if j / 3600 == 24:
            j = 0

        if j / 3600 < 12:
            T_top[i] = -5
            j += 1
        elif j / 3600 < 24:
            T_top[i] = -10
            j += 1

    return T_top

def t_top_testcase_2(time_total, dt):
    """
    Function to build array of surface temperatures with one value every timestep dt.
    Developed to construct surface temperatures used in testcase 1.
    :param time_total: total time of simulation from config file
    :param dt: timestep dt from config file
    :return: T_top: surface temperature as used in testcase one in SAMSIM1.0/2.0
    """
    T_top = np.ones(int(time_total / dt))
    j = 0
    for i in range(T_top.size):

        if j / 3600 == 24:
            j = 0

        if j / 3600 < 12:
            T_top[i] = -10
            j += 1
        elif j / 3600 < 24:
            T_top[i] = -10
            j += 1

    return T_top

def t2m_testcase_2(time_total, dt):

    T2m = np.ones(int(time_total/dt)) * -20
    T2m[int(86400*15/dt):int(86400*25/dt)] = 1
    T2m[int(86400*25/dt):-2] = 15

    return T2m

def notzflux(time_total, dt):

    fl_sw = np.ones(int(time_total / dt))
    fl_rest = np.ones(int(time_total / dt))

    for i in np.arange(1, time_total, dt):
        day = (i+86400*180)/86400  # seltsamer offset
        while day > 360:  # 365 würde mehr Sinn ergeben ?
            day = day - 360

        fl_sw[int(i/dt)] = 314 * np.exp(-0.5 * ((day - 164) / 47.9) ** 2)
        fl_rest[int(i/dt)] = 118 * np.exp(-0.5 * ((day - 206) / 53.1) ** 2) + 179

        if (day < 60 or day > 300.):
            fl_sw[int(i/dt)] = 0.0

    return fl_sw, fl_rest

def build_inital_state(Ice_thickness, S_bu_top, T_top, config, const):

    # parameters
    c_s = 2020.0
    c_s_beta = 7.6973
    latent_heat = 333500.
    rho_s = 920.
    rho_l = 1028.

    # thermo functions
    def func_S_br(T, S_bu):
        c1 = 0.0
        c2 = -18.7
        c3 = -0.519
        c4 = -0.00535
        S_br = c1 + c2 * T + c3 * T ** 2. + c4 * T ** 3
        S_br[np.less(S_br, S_bu)] = S_bu[np.less(S_br, S_bu)]
        return S_br

    def func_H(T, S_bu):
        return -latent_heat * (1. - S_bu / func_S_br(T, S_bu)) + c_s * T + 1. / 2. * c_s_beta * T ** 2.

    # calculate N_active
    z0 = (config['N_top'] + config['N_bottom']) * config['thick_0']
    z1 = Ice_thickness - z0
    if z1 < 0 :
        N_active = int(Ice_thickness / config['thick_0'])
        thick = np.zeros(config['Nlayer'])
        thick[0:N_active] = config['thick_0']
    else:
        dz1 = z1 / config['N_middle']

        if dz1 > config['thick_0']:
            N_active = config['Nlayer']
            thick = np.array(
                [config['thick_0'] for n in range(config['N_top'])] +
                [(Ice_thickness - config['thick_0'] * (config['N_top'] + config['N_bottom'])) / config['N_middle'] for n in
                range(config['N_middle'])] +
                [config['thick_0'] for n in range(config['N_bottom'])])

        else:
            N_active = config['Nlayer'] - config['N_top'] - config['N_bottom']
            N_middle_active = N_active - config['N_top'] - config['N_bottom']
            thick = np.array(
                [config['thick_0'] for n in range(config['N_top'])] +
                [(Ice_thickness - config['thick_0'] * (config['N_top'] + config['N_bottom'])) / N_middle_active for n in
                range(N_middle_active)] +
                [config['thick_0'] for n in range(config['N_bottom'])])


    # create temperature profile
    T = np.linspace(T_top, const['T_bottom'], N_active, endpoint=True)

    # create bulk salinity profile
    S_bu = np.array(
        [S_bu_top for n in range(np.floor(N_active * 0.75).astype('int'))] +
        list(np.linspace(S_bu_top, const['S_bu_bottom'], N_active - np.floor(N_active * 0.75).astype('int'), endpoint=True))
    )

    # retrieve enthalpy, solid fraction profile

    H = func_H(T, S_bu)
    phi = 1. - S_bu / func_S_br(T, S_bu)

    # retrieve mass profile
    m = thick[0:N_active] * (phi * rho_s + (1. - phi) * rho_l)

    S_abs = S_bu * m
    H_abs = H * m

    return H_abs, S_abs, m, thick, N_active











