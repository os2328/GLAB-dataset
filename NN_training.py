# This script is used to train NN and perform transfer learning

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
overlap_m['doy'] =  overlap_m['time'].dt.dayofyear

overlap_m = overlap_m.merge(smos_seas, on =['lat', 'lon', 'doy'], how = 'inner')
overlap_m = overlap_m.dropna()

file_data = '/burg/glab/users/os2328/data/amsr2_seas_no_std.pkl'
amsr2 = pd.read_pickle(file_data)
amsr2['time'] = pd.to_datetime(amsr2['time'])
amsr2 = amsr2.sort_values(by='time')

overlap_2020 = amsr2.merge(smos_m, on =['lat', 'lon', 'time'], how = 'inner')


overlap_2020 = overlap_2020.dropna()

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

train = overlap_2020.sample(frac=0.8)
test = overlap_2020.drop(train.index)

# features are coordinates and TB residuals
train_dataset = train[[ 'lat', 'sin_lon',  'dev_H10', 'dev_V10', 'dev_H18', 'dev_V18', 'dev_V36' ]]
test_dataset = test[[ 'lat', 'sin_lon',   'dev_H10', 'dev_V10', 'dev_H18', 'dev_V18',  'dev_V36']]

X = train_dataset.values.astype(float)
# target is SMOS TB residuals
Y = train['dev_H'].values.astype(float)

Xt = test_dataset.values.astype(float)
Yt = test['dev_H'].values.astype(float)

X_ovelap_full = overlap_2020[[ 'lat', 'sin_lon',  'dev_H10', 'dev_V10', 'dev_H18', 'dev_V18',  'dev_V36']]
X_ovelap_full = X_ovelap_full.values.astype(float)

X_full = overlap_m[[ 'lat', 'sin_lon',  'dev_H10', 'dev_V10', 'dev_H18', 'dev_V18',  'dev_V36']] #
X_full = X_full.values.astype(float)



scale =preprocessing.RobustScaler()
# fit scaler to all possible AMSR data
scalerX = scale.fit(X_full)

X = scalerX.transform(X)
Xt = scalerX.transform(Xt)
X_ovelap_full = scalerX.transform(X_ovelap_full)


#X_full = scalerX.transform(X_full)




# NN parameters
bs = 512
num_of_units = 1050
adm = optimizers.Adam(lr=0.001)

#########

epoch=100

inputs = tf.keras.layers.Input(shape=(X.shape[1],))
x = tf.keras.layers.Dense(units=1512,  activation=LeakyReLU())(inputs)
x = tf.keras.layers.Dense(units=1000,  activation=LeakyReLU())(x)
x = tf.keras.layers.Dense(units=800,  activation='relu')(x)
x = tf.keras.layers.Dense(units=600, activation='relu')(x)
x = tf.keras.layers.Dense(units=400,  activation='relu')(x)
x = tf.keras.layers.Dense(units=200,  activation='relu')(x)
outputs = tf.keras.layers.Dense(1)(x)


model2 = tf.keras.Model(inputs=inputs, outputs=outputs)

model2.compile(loss=Huber(delta=1.0), optimizer=adm, metrics=['mae'])
#custom_objects = {'LeakyReLU': LeakyReLU}

#model2 = keras.models.load_model(path_to_save_nn + 'loss_compare_leaky.h5', custom_objects=custom_objects)

history = model2.fit(X, Y, epochs=epoch, batch_size=bs, validation_split=0.2, verbose=2)

model2.save(path_to_save_nn +'target_scaling_robust-H.h5')

y = model2.predict(X, batch_size=bs, verbose=0)
y = np.asarray(y).reshape(-1)

perf_m(Y, y, ' TRAINING')
print('target median')
print(np.median(Y))
print('pred median')
print(np.median(y))

y_t = model2.predict(Xt, batch_size=bs, verbose=0)
y_t = np.asarray(y_t).reshape(-1)

perf_m(Yt, y_t, 'test')
print('target median')
print(np.median(Yt))
print('pred median')
print(np.median(y_t))

#custom_objects = {'LeakyReLU': LeakyReLU}

#model = keras.models.load_model(path_to_save_nn + 'loss_compare_huber_100.h5', custom_objects=custom_objects)


train = overlap_2010.sample(frac=0.85)
test = overlap_2010.drop(train.index)

# features are coordinates and TB residuals
train_dataset = train[[ 'lat', 'sin_lon',  'dev_H10', 'dev_V10', 'dev_H18', 'dev_V18', 'dev_V36' ]]
test_dataset = test[[ 'lat', 'sin_lon',   'dev_H10', 'dev_V10', 'dev_H18', 'dev_V18',  'dev_V36']]

X = train_dataset.values.astype(float)
# target is SM residuals
Y = train['dev_H'].values.astype(float)

Xt = test_dataset.values.astype(float)
Yt = test['dev_H'].values.astype(float)

X_ovelap_full = overlap_m[[ 'lat', 'sin_lon',  'dev_H10', 'dev_V10', 'dev_H18', 'dev_V18',  'dev_V36']] # 'dev_H', 'dev_V']]
X_ovelap_full = X_ovelap_full.values.astype(float)
#Y_ovelap_full = overlap['dev_H'].values.astype(float)


#scale =preprocessing.RobustScaler()
# fit scaler to all possible smos data
#scalerX = scale.fit(X_ovelap_full)

X = scalerX.transform(X)
Xt = scalerX.transform(Xt)
X_ovelap_full = scalerX.transform(X_ovelap_full)


#model.save(path_to_save_nn +'NN_TB_H.h5')
#model = keras.models.load_model(path_to_save_nn + 'NN_TB_AMSR2_H.h5')
#print(model.summary())

n_no_train = 3
# set the first n_no_train layers non trainble
for layer in model2.layers[:n_no_train]:
    layer.trainable = False

epoch = 70

history = model2.fit(X, Y, epochs=epoch, batch_size=bs, validation_split=0.15, verbose=2)

model2.save(path_to_save_nn +'target_scaling_afterTL-rob_H.h5')


###########

##import sys
##try:
##    arg1 = os.environ['MY_ARG']
##except:
##    import random
##    arg1 = random.randint(1, 100)
##    print('arg didnt work, going with rand number')
#######################

#arg1 = str(sys.argv[0])
print('Training TB to TB')
print('version H after Transfer learning!!')
###print(arg1)

y = model2.predict(X, batch_size=bs, verbose=0)
y = np.asarray(y).reshape(-1)

perf_m(Y, y, ' TRAINING')
print('target median')
print(np.median(Y))
print('pred median')
print(np.median(y))

y_t = model2.predict(Xt, batch_size=bs, verbose=0)
y_t = np.asarray(y_t).reshape(-1)

perf_m(Yt, y_t, 'test')
print('target median')
print(np.median(Yt))
print('pred median')
print(np.median(y_t))


y_f = model2.predict(X_ovelap_full, batch_size=bs, verbose=0)
y_f = np.asarray(y_f).reshape(-1)
