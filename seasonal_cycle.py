### This script is used to calculate seasonal cycle for SMAP SM and SMOS TB data


import pandas as pd
import numpy as np
import time
from datetime import timedelta
from scipy.optimize import curve_fit
import math
import xarray as xr

# Read files
data_path = '/burg/glab/users/os2328/data/'

file_data_smos = data_path + 'smos_ic_xr.zarr'
one = xr.open_dataset(file_data_smos)
one = one.to_dataframe()
one = one.reset_index()
smos = one.dropna()
smos['doy'] =  smos['time'].dt.dayofyear


# Define functions that will calculate seasonal cycle. The functions are applied with group_by, for that reason they cannot take any other arguments, hence there are separate functions per each column for which it should be calculated.

# minimum number of datapoints to calculate seasonal cycle
n_min =40
def seas_cycle_10H(df):
    global count_c # counts the number of points, at which seasonal cycle is calculated
    global count_m # counts the number of points, at which curve_fit could not be fit
    global count_l # counts the number of points, at which time series are too short (< n_min)
    t = df['t']
    y = df['BT_H_IC_Fitted']
    # simple sine wave
    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    # only apply if there are at least n_min values
    if len(y)<n_min:
    # if less, median of the values is used
        y_s =np.median(y)*y/y
        count_l = count_l+1

    else:
        try:
        # bounds for TB in K, period ~ 1 year - between 365 and 366 days
            popt, pcov = curve_fit(func, t, y, bounds=([1., 365., 0., 120. ], [80., 366., 2.0*np.pi, 370.]))
            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s =np.median(y)*y/y
            count_m = count_m+1
    return y_s

# same for V-polarization TB
def seas_cycle_10V(df):
    global count_c
    global count_m
    global count_l
    t = df['t']
    y = df['BT_H_IC_Fitted']

    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    if len(y)<n_min:
        y_s =np.median(y)*y/y
        count_l = count_l+1
    else:

        try:
            popt, pcov = curve_fit(func, t, y, bounds=([1., 365., 0., 120. ], [80., 366., 2.0*np.pi, 370.]))
            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s = np.median(y)*y/y
            count_m = count_m+1
    return y_s

def seas_cycle_18H(df):
    global count_c # counts the number of points, at which seasonal cycle is calculated
    global count_m # counts the number of points, at which curve_fit could not be fit
    global count_l # counts the number of points, at which time series are too short (< n_min)
    t = df['t']
    y = df['TB187H']
    # simple sine wave
    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    # only apply if there are at least n_min values
    if len(y)<n_min:
    # if less, median of the values is used
        y_s =np.median(y)*y/y
        count_l = count_l+1

    else:
        try:
        # bounds for TB in K, period ~ 1 year - between 365 and 366 days
            popt, pcov = curve_fit(func, t, y, bounds=([1., 365., 0., 120. ], [80., 366., 2.0*np.pi, 370.]))
            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s =np.median(y)*y/y
            count_m = count_m+1
    return y_s
# same for V-polarization TB
def seas_cycle_18V(df):
    global count_c
    global count_m
    global count_l
    t = df['t']
    y = df['TB187V']

    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    if len(y)<n_min:
        y_s =np.median(y)*y/y
        count_l = count_l+1
    else:

        try:
            popt, pcov = curve_fit(func, t, y, bounds=([1., 365., 0., 120. ], [80., 366., 2.0*np.pi, 370.]))
            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s = np.median(y)*y/y
            count_m = count_m+1
    return y_s


def seas_cycle_23V(df):
    global count_c # counts the number of points, at which seasonal cycle is calculated
    global count_m # counts the number of points, at which curve_fit could not be fit
    global count_l # counts the number of points, at which time series are too short (< n_min)
    t = df['t']
    y = df['TBV23r2']
    # simple sine wave
    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    # only apply if there are at least n_min values
    if len(y)<n_min:
    # if less, median of the values is used
        y_s =np.median(y)*y/y
        count_l = count_l+1

    else:
        try:
        # bounds for TB in K, period ~ 1 year - between 365 and 366 days
            popt, pcov = curve_fit(func, t, y, bounds=([1., 365., 0., 120. ], [80., 366., 2.0*np.pi, 370.]))
            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s =np.median(y)*y/y
            count_m = count_m+1
    return y_s

# same for V-polarization TB
def seas_cycle_36V(df):
    global count_c
    global count_m
    global count_l
    t = df['t']
    y = df['TB365V']

    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    if len(y)<n_min:
        y_s =np.median(y)*y/y
        count_l = count_l+1
    else:

        try:
            popt, pcov = curve_fit(func, t, y, bounds=([1., 365., 0., 120. ], [80., 366., 2.0*np.pi, 370.]))
            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s = np.median(y)*y/y
            count_m = count_m+1
    return y_s

# same for SM, but the BOUNDS are different!
def seas_cycle(df):
    global count_c
    global count_m
    global count_l
    global seas_df
    t = df['t']
    #y = df['vod']
    y = df['sm_am']

    def func(x, A, period, phi, b):
        omega = 2.0 *np.pi/period
        return A * np.sin(omega * x + phi)+b
    if len(y)<n_min:
        y_s =np.median(y)*y/y
        count_l = count_l+1
    else:
        try:
            #popt, pcov = curve_fit(func, t, y, bounds=([0., 365., 0., 0. ], [1.0, 366., 2.0*np.pi, 1.2]))
            popt, pcov = curve_fit(func, t, y, bounds=([0., 365., 0., 0. ], [1.0, 366., 2.0*np.pi, 1.0]))

            y_s = func(t, *popt)
            count_c = count_c+1
        except:
            y_s =np.median(y)*y/y
            count_m = count_m+1
    return y_s







smos = smos.sort_values(by = ['time', 'lat', 'lon'])


d0 = smos['time'].iloc[0]
smos['t']  = (smos['time']-d0)
smos['t']  = smos['t'].dt.days.astype('int16')


count_m = 0
count_l = 0
count_c = 0

grouped_smos = smos.groupby(['lat', 'lon'])
tbh_output = grouped_smos.apply(seas_cycle_H).reset_index()
tbv_output = grouped_smos.apply(seas_cycle_V).reset_index()

tbh_output= tbh_output.set_index('level_2')
tbv_output= tbv_output.set_index('level_2')

tbh_output = tbh_output.drop(columns=['lat', 'lon'])
tbv_output = tbv_output.drop(columns=['lat', 'lon'])

tbh_output.columns = ['tbH_seas']
tbv_output.columns = ['tbV_seas']

smos_final = smos.join(tbh_output)
smos_final = smos_final.join(tbv_output)


smos_joint = smos_final.join(smos_final.groupby(['lat', 'lon', 'doy'])[['tbH_seas', 'tbV_seas']].median(), on=['lat', 'lon', 'doy'], rsuffix='_med')
try:
    smos_joint = smos_joint.drop(columns=['tbH_seas', 'tbV_seas', 't'])
except:
    print('no drop')
    print(smos_joint.head())
    print(smos_joint.columns)
# add soil moisture seasonal cycle
smos_joint_sc = smos_joint.merge(ss, on=['lat', 'lon', 'doy'])


smos_joint_sc['dev_H'] = smos_joint_sc['BT_H_IC_Fitted'] - smos_joint_sc['tbH_seas_med']
smos_joint_sc['dev_V'] = smos_joint_sc['BT_V_IC_Fitted'] - smos_joint_sc['tbV_seas_med']
print('SMOS: The number of points at which seasonal cycle was succesfully calculated using curve_fit: %2d' %count_c)
print('SMOS: The number of points at which curve_fit did not work (median used instead): %2d' %count_m)
print('SMOS: The number of points at which time series were too short to use curve_fit (median used instead): %2d' %count_l)

smos_joint_sc.to_pickle(path_to_save + 'smos_3dm_seas_c_no_std.pkl', protocol = 4)


