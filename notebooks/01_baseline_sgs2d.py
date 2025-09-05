# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import random

import matplotlib.pyplot as plt
# %%
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor


# %%
def create_model_1(neurons=50):
    model = Sequential()
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=1, use_bias=True))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=adam,
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model


# %%
def create_model_1a(neurons=50):
    model = Sequential()
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=2, use_bias=True))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=adam,
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model


# %%
def create_model_2lay_sig(neurons=64):
    model = Sequential()
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=2, use_bias=True))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=adam,
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model


# %%
def create_model_4lay_sig(neurons=64):
    model = Sequential()
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=2, use_bias=True))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=adam,
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model


# %%
def create_model_2lay_source(neurons=64):
    model = Sequential()
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=1, use_bias=True))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=adam,
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model


# %%
def create_model_4lay_source(neurons=64):
    model = Sequential()
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=neurons, activation="relu", use_bias=True))
    model.add(Dense(units=1, use_bias=True))
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(
        optimizer=adam,
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model


# %%
train_n4096l_fw16_sig1 = pd.read_csv(
    "upd2_n4096l_fw16_PartialTrainData_sig1.dat", sep="\s+", header=None
)
train_n4096l_fw16_sig1 = pd.DataFrame(train_n4096l_fw16_sig1).to_numpy()
train_n4096l_fw16_sig2 = pd.read_csv(
    "upd2_n4096l_fw16_PartialTrainData_sig2.dat", sep="\s+", header=None
)
train_n4096l_fw16_sig2 = pd.DataFrame(train_n4096l_fw16_sig2).to_numpy()

train_n4096l_fw16_sig = np.concatenate(
    (train_n4096l_fw16_sig1, train_n4096l_fw16_sig2), axis=0
)

train_n4096l_fw32_sig1 = pd.read_csv(
    "upd2_n4096l_fw32_PartialTrainData_sig1.dat", sep="\s+", header=None
)
train_n4096l_fw32_sig1 = pd.DataFrame(train_n4096l_fw32_sig1).to_numpy()
train_n4096l_fw32_sig2 = pd.read_csv(
    "upd2_n4096l_fw32_PartialTrainData_sig2.dat", sep="\s+", header=None
)
train_n4096l_fw32_sig2 = pd.DataFrame(train_n4096l_fw32_sig2).to_numpy()

train_n4096l_fw32_sig = np.concatenate(
    (train_n4096l_fw32_sig1, train_n4096l_fw32_sig2), axis=0
)


train_n4096l_fw16_source = pd.read_csv(
    "upd_n4096l_fw16_PartialTrainData_sourceterm.dat", sep="\s+", header=None
)
train_n4096l_fw16_source = pd.DataFrame(train_n4096l_fw16_source).to_numpy()

train_n4096l_fw32_source = pd.read_csv(
    "upd_n4096l_fw32_PartialTrainData_sourceterm.dat", sep="\s+", header=None
)
train_n4096l_fw32_source = pd.DataFrame(train_n4096l_fw32_source).to_numpy()

# %%
train_n4096l_fw16_sig1.shape

# %%
smag_col_fw16 = np.sqrt(
    2.0 * (train_n4096l_fw16_sig[:, 18]) ** 2
    + train_n4096l_fw16_sig[:, 19] ** 2
    + train_n4096l_fw16_sig[:, 20] ** 2
)
leith_col_fw16 = np.sqrt(
    train_n4096l_fw16_sig[:, 3] ** 2 + train_n4096l_fw16_sig[:, 4] ** 2
)

smag_col_fw32 = np.sqrt(
    2.0 * (train_n4096l_fw32_sig[:, 19]) ** 2
    + train_n4096l_fw32_sig[:, 20] ** 2
    + train_n4096l_fw32_sig[:, 21] ** 2
)
leith_col_fw32 = np.sqrt(
    train_n4096l_fw32_sig[:, 4] ** 2 + train_n4096l_fw32_sig[:, 5] ** 2
)

# %%
train_n4096l_fw16_sig_model2A_y = train_n4096l_fw16_sig[:, np.r_[0, 1]]
train_n4096l_fw16_sig_model2A_X = train_n4096l_fw16_sig[:, np.r_[3, 4, 8, 9, 10, 11]]

train_n4096l_fw16_sig_model2B_y = train_n4096l_fw16_sig[:, np.r_[0, 1]]
# train_n4096l_fw16_sig_model2B_X = np.hstack((train_n4096l_fw16_sig[:,np.r_[3,4,8,9,10,11]], smag_col_fw16.reshape(-1,1), leith_col_fw16.reshape(-1,1)))
train_n4096l_fw16_sig_model2B_X = train_n4096l_fw16_sig[:, 3:18]

train_n4096l_fw32_sig_model2A_y = train_n4096l_fw32_sig[:, np.r_[0, 1]]
train_n4096l_fw32_sig_model2A_X = train_n4096l_fw32_sig[:, np.r_[4, 5, 9, 10, 11, 12]]

train_n4096l_fw32_sig_model2B_y = train_n4096l_fw32_sig[:, np.r_[0, 1]]
# train_n4096l_fw32_sig_model2B_X = np.hstack((train_n4096l_fw32_sig[:,np.r_[4,5,9,10,11,12]], smag_col_fw32.reshape(-1,1), leith_col_fw32.reshape(-1,1)))
train_n4096l_fw32_sig_model2B_X = train_n4096l_fw32_sig[:, 4:19]

train_n4096l_fw16_source_model2B_y = -train_n4096l_fw16_source[:, 0]
train_n4096l_fw16_source_model2B_X = train_n4096l_fw16_source[:, 1:17]

train_n4096l_fw32_source_model2B_y = -train_n4096l_fw32_source[:, 0]
train_n4096l_fw32_source_model2B_X = train_n4096l_fw32_source[:, 1:17]

# %%
train_n4096l_fw16_sig_model2B_X.shape

# %%
# Setting up the neural network models for different filter widths

NN_n4096l_fw16_sig_model2A = create_model_2lay_sig(50)
NN_n4096l_fw32_sig_model2A = create_model_2lay_sig(50)

NN_n4096l_fw16_sig_model2B = create_model_2lay_sig(50)
NN_n4096l_fw32_sig_model2B = create_model_2lay_sig(50)

NN_n4096l_fw16_source_model2B = create_model_2lay_source(50)
NN_n4096l_fw32_source_model2B = create_model_2lay_source(50)

# %%
# Training the models

# callback

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=100, restore_best_weights=True
)

hist_NN_n4096l_fw16_sig_model2A = NN_n4096l_fw16_sig_model2A.fit(
    train_n4096l_fw16_sig_model2A_X,
    train_n4096l_fw16_sig_model2A_y,
    validation_split=0.2,
    batch_size=256,
    epochs=100,
)
NN_n4096l_fw16_sig_model2A.save(
    "Enstrophyform_savedmodels_n4096l/2NN_n4096l_fw16_sig_model2A"
)

hist_NN_n4096l_fw16_sig_model2B = NN_n4096l_fw16_sig_model2B.fit(
    train_n4096l_fw16_sig_model2B_X,
    train_n4096l_fw16_sig_model2B_y,
    validation_split=0.2,
    batch_size=256,
    epochs=100,
)
NN_n4096l_fw16_sig_model2B.save(
    "Enstrophyform_savedmodels_n4096l/2NN_n4096l_fw16_sig_model2B"
)

hist_NN_n4096l_fw32_sig_model2A = NN_n4096l_fw32_sig_model2A.fit(
    train_n4096l_fw32_sig_model2A_X,
    train_n4096l_fw32_sig_model2A_y,
    validation_split=0.2,
    batch_size=256,
    epochs=100,
)
NN_n4096l_fw32_sig_model2A.save(
    "Enstrophyform_savedmodels_n4096l/2NN_n4096l_fw32_sig_model2A"
)

hist_NN_n4096l_fw32_sig_model2B = NN_n4096l_fw32_sig_model2B.fit(
    train_n4096l_fw32_sig_model2B_X,
    train_n4096l_fw32_sig_model2B_y,
    validation_split=0.2,
    batch_size=256,
    epochs=100,
)
NN_n4096l_fw32_sig_model2B.save(
    "Enstrophyform_savedmodels_n4096l/2NN_n4096l_fw32_sig_model2B"
)

hist_NN_n4096l_fw16_source_model2B = NN_n4096l_fw16_source_model2B.fit(
    train_n4096l_fw16_source_model2B_X,
    train_n4096l_fw16_source_model2B_y,
    validation_split=0.2,
    batch_size=256,
    epochs=100,
)
NN_n4096l_fw16_source_model2B.save(
    "Enstrophyform_savedmodels_n4096l/2NN_n4096l_fw16_source_model2B"
)

hist_NN_n4096l_fw32_source_model2B = NN_n4096l_fw32_source_model2B.fit(
    train_n4096l_fw32_source_model2B_X,
    train_n4096l_fw32_source_model2B_y,
    validation_split=0.2,
    batch_size=256,
    epochs=100,
)
NN_n4096l_fw32_source_model2B.save(
    "Enstrophyform_savedmodels_n4096l/2NN_n4096l_fw32_source_model2B"
)

# %%
NN_n4096l_fw16_sig_model2A.save(
    "Enstrophyform_savedmodels_n4096l/NN_n4096l_fw16_sig_model2A"
)
NN_n4096l_fw32_sig_model2A.save(
    "Enstrophyform_savedmodels_n4096l/NN_n4096l_fw32_sig_model2A"
)
NN_n4096l_fw16_sig_model2B.save(
    "Enstrophyform_savedmodels_n4096l/NN_n4096l_fw16_sig_model2B"
)
NN_n4096l_fw32_sig_model2B.save(
    "Enstrophyform_savedmodels_n4096l/NN_n4096l_fw32_sig_model2B"
)
NN_n4096l_fw16_source_model2B.save(
    "Enstrophyform_savedmodels_n4096l/NN_n4096l_fw16_source_model2B"
)
NN_n4096l_fw32_source_model2B.save(
    "Enstrophyform_savedmodels_n4096l/NN_n4096l_fw32_source_model2B"
)

# %%
# %%
import pickle

# #fw16

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw16_sig_model2A", "wb"
) as file_pi:
    pickle.dump(hist_NN_n4096l_fw16_sig_model2A.history, file_pi)

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw16_sig_model2B", "wb"
) as file_pi:
    pickle.dump(hist_NN_n4096l_fw16_sig_model2B.history, file_pi)

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw16_source_model2B", "wb"
) as file_pi:
    pickle.dump(hist_NN_n4096l_fw16_source_model2B.history, file_pi)


# fw32

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw32_sig_model2A", "wb"
) as file_pi:
    pickle.dump(hist_NN_n4096l_fw32_sig_model2A.history, file_pi)

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw32_sig_model2B", "wb"
) as file_pi:
    pickle.dump(hist_NN_n4096l_fw32_sig_model2B.history, file_pi)

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw32_source_model2B", "wb"
) as file_pi:
    pickle.dump(hist_NN_n4096l_fw32_source_model2B.history, file_pi)


# %%
# fw16

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw16_sig_model2A", "rb"
) as file_pi:
    hist_NN_n4096l_fw16_sig_model2A = pickle.load(file_pi)

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw16_sig_model2B", "rb"
) as file_pi:
    hist_NN_n4096l_fw16_sig_model2B = pickle.load(file_pi)

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw16_source_model2B", "rb"
) as file_pi:
    hist_NN_n4096l_fw16_source_model2B = pickle.load(file_pi)


# fw32

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw32_sig_model2A", "rb"
) as file_pi:
    hist_NN_n4096l_fw32_sig_model2A = pickle.load(file_pi)

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw32_sig_model2B", "rb"
) as file_pi:
    hist_NN_n4096l_fw32_sig_model2B = pickle.load(file_pi)

with open(
    "Enstrophyform_history_n4096l/hist_2NN_n4096l_fw32_source_model2B", "rb"
) as file_pi:
    hist_NN_n4096l_fw32_source_model2B = pickle.load(file_pi)

# %%
hist_NN_n4096l_fw32_sig_model2A

# %%
# summarize history for loss

arr_fw16_sig2A_loss = np.array(hist_NN_n4096l_fw16_sig_model2A.history["loss"])
arr_fw16_sig2A_valloss = np.array(hist_NN_n4096l_fw16_sig_model2A.history["val_loss"])

arr_fw32_sig2A_loss = np.array(hist_NN_n4096l_fw32_sig_model2A.history["loss"])
arr_fw32_sig2A_valloss = np.array(hist_NN_n4096l_fw32_sig_model2A.history["val_loss"])

arr_fw16_sig2B_loss = np.array(hist_NN_n4096l_fw16_sig_model2B.history["loss"])
arr_fw16_sig2B_valloss = np.array(hist_NN_n4096l_fw16_sig_model2B.history["val_loss"])

arr_fw32_sig2B_loss = np.array(hist_NN_n4096l_fw32_sig_model2B.history["loss"])
arr_fw32_sig2B_valloss = np.array(hist_NN_n4096l_fw32_sig_model2B.history["val_loss"])

arr_fw16_source_loss = np.array(hist_NN_n4096l_fw16_source_model2B.history["loss"])
arr_fw16_source_valloss = np.array(
    hist_NN_n4096l_fw16_source_model2B.history["val_loss"]
)

arr_fw32_source_loss = np.array(hist_NN_n4096l_fw32_source_model2B.history["loss"])
arr_fw32_source_valloss = np.array(
    hist_NN_n4096l_fw32_source_model2B.history["val_loss"]
)

import scipy.optimize as opt


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


# #fw16

plt.plot(arr_fw16_sig2A_loss[10 : arr_fw16_sig2A_loss.size], color="orange")
plt.plot(arr_fw16_sig2A_valloss[10 : arr_fw16_sig2A_valloss.size], color="blue")
plt.title("hist_NN_n4096l_fw16_sig_model2A loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train"], loc="upper left")
plt.show()

plt.plot(arr_fw16_sig2B_loss[10 : arr_fw16_sig2B_loss.size], color="orange")
plt.plot(arr_fw16_sig2B_valloss[10 : arr_fw16_sig2B_valloss.size], color="blue")
plt.title("hist_NN_n4096l_fw16_sig_model2B loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train"], loc="upper left")
plt.show()

plt.plot(arr_fw16_source_loss[10 : arr_fw16_source_loss.size], color="orange")
plt.plot(arr_fw16_source_valloss[10 : arr_fw16_source_valloss.size], color="blue")
plt.title("hist_NN_n4096l_fw16_source_model2B loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train"], loc="upper left")
plt.show()


# fw32

plt.plot(arr_fw32_sig2A_loss[10 : arr_fw32_sig2A_loss.size], color="orange")
plt.plot(arr_fw32_sig2A_valloss[10 : arr_fw32_sig2A_valloss.size], color="blue")
plt.title("hist_NN_n4096l_fw32_sig_model2A loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train"], loc="upper left")
plt.show()

plt.plot(arr_fw32_sig2B_loss[10 : arr_fw32_sig2B_loss.size], color="orange")
plt.plot(arr_fw32_sig2B_valloss[10 : arr_fw32_sig2B_valloss.size], color="blue")
plt.title("hist_NN_n4096l_fw32_sig_model2B loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train"], loc="upper left")
plt.show()

plt.plot(arr_fw32_source_loss[10 : arr_fw32_source_loss.size], color="orange")
plt.plot(arr_fw32_source_valloss[10 : arr_fw32_source_valloss.size], color="blue")
plt.title("hist_NN_n4096l_fw32_source_model2B loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train"], loc="upper left")
plt.show()

# %%
arr_fw32_source_loss
