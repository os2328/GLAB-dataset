#  This script is used for NN forward run with trained NN available

import sklearn as skl
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import model_from_json
from tensorflow.keras.losses import Huber
from tensorflow.keras.losses import MeanAbsoluteError

import time
from datetime import timedelta
import xarray as xr
import os



path_to_file = '/burg/glab/users/os2328/data/VOD_project/'
path_to_save_nn = '/burg/glab/users/os2328/data/VOD_project/'
path_to_save_output = '/burg/glab/users/os2328/data/VOD_project/'
file_data = path_to_file + 'smos_3dm_seas_c_no_std.pkl'
smos_m = pd.read_pickle(file_data)
smos_m['time'] = pd.to_datetime(smos_m['time'])
smos_m = smos_m.sort_values(by='time')
smos_m['doy'] =  smos_m['time'].dt.dayofyear
smos_seas = smos_m[['lat', 'lon', 'doy', 'tbH_seas_med', 'tbV_seas_med']]
smos_seas = smos_seas.drop_duplicates()

file_data = '/burg/glab/users/os2328/data/amsrE_seas_no_std.pkl'
overlap_m = pd.read_pickle(file_data)
overlap_m['time'] = pd.to_datetime(overlap_m['time'])
overlap_m = overlap_m.sort_values(by='time')

overlap_m = overlap_m.dropna()
overlap_m['sin_lon'] = np.sin(2*np.pi*overlap_m['lon']/360)
print(overlap_m.shape)
overlap_m['doy'] =  overlap_m['time'].dt.dayofyear

overlap_m = overlap_m.merge(smos_seas, on =['lat', 'lon', 'doy'], how = 'inner')
overlap_m = overlap_m.dropna()

file_data = '/burg/glab/users/os2328/data/amsr2_seas_no_std.pkl'
amsr2 = pd.read_pickle(file_data)
amsr2['time'] = pd.to_datetime(amsr2['time'])
amsr2 = amsr2.sort_values(by='time')

overlap_2020 = amsr2.merge(smos_m, on =['lat', 'lon', 'time'], how = 'inner')


overlap_2020 = overlap_2020.dropna()
print('after AMSR2 merge with smos')
print(overlap_2020.shape)
print(np.unique(amsr2['time']))
print(np.unique(smos_m['time']))

overlap_2020['sin_lon'] = np.sin(2*np.pi*overlap_2020['lon']/360)

overlap_2010 = overlap_m.merge(smos_m, on =['lat', 'lon', 'time'], how = 'inner')


def perf_m(true_val,pred, print_mes):

    R = np.corrcoef(pred, true_val)

    R2_op = r2_score( true_val, pred)

    rmse = mean_squared_error(true_val, pred, squared=False)
    print('Correlation R between the NN resid SM signal and the target resid SM signal for the ' + print_mes)
    print(R)
    print('R^2 between the NN SM resid and the target resid  SM signal for the ' + print_mes)


    print(R2_op)

    print('RMSE between the NN resid  SM  and the target resid SM signal for the ' + print_mes)
    print(rmse)
    return print('  ')

custom_objects = {'LeakyReLU': LeakyReLU}

model2 = keras.models.load_model(path_to_save_nn + 'target_scaling_robust-H.h5', custom_objects=custom_objects)


y = model2.predict(X_ovelap_full, batch_size=bs, verbose=0)
y = np.asarray(y).reshape(-1)

perf_m(Y_ovelap_full, y, ' H resid')
print('target median')
print(np.median(Y_ovelap_full))
print('pred median')
print(np.median(y))

H_seas = overlap_2020['tbH_seas_med'].values.astype(float)
full = y + H_seas
full_true = Y_ovelap_full + H_seas

perf_m(full_true, full, ' H full')

model3 = keras.models.load_model(path_to_save_nn + 'target_scaling_robust.h5', custom_objects=custom_objects)


y_v = model3.predict(X_ovelap_full, batch_size=bs, verbose=0)
y_v = np.asarray(y_v).reshape(-1)

perf_m(Y_ovelap_full_V, y_v, ' V resid')
print('target median')
print(np.median(Y_ovelap_full_V))
print('pred median')
print(np.median(y_v))

V_seas = overlap_2020['tbV_seas_med'].values.astype(float)
full_v = y_v + V_seas
full_true_v = Y_ovelap_full_V + V_seas

perf_m(full_true_v, full_v, ' V full')
overlap_2020['nn_out_H_res'] = y
overlap_2020['nn_out_V_res'] = y_v
overlap_2020['nn_out_H_full'] = full
overlap_2020['nn_out_V_full'] = full_v
overlap_2020.to_pickle(path_to_save_output + 'nn_out_TB_as_smos.pkl', protocol = 4)

