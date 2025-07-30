# tf_models/data_loader.py

import numpy as np
import pandas as pd
from .config import (
    TRAIN16_SIG1, TRAIN16_SIG2,
    TRAIN32_SIG1, TRAIN32_SIG2,
    TRAIN16_SOURCE, TRAIN32_SOURCE
)

def load_sig_data(path1, path2):
    """Load & stack signature data for a given filter width."""
    a = pd.read_csv(path1, sep=r"\s+", header=None).values
    b = pd.read_csv(path2, sep=r"\s+", header=None).values
    return np.vstack([a, b])

def load_source_data(path):
    """Load source-term data for a given filter width."""
    return pd.read_csv(path, sep=r"\s+", header=None).values

# --- Signature models, filter width 16 ---
def get_train_fw16_sig_model2A():
    data = load_sig_data(TRAIN16_SIG1, TRAIN16_SIG2)
    y = data[:, [0, 1]]                              # subgrid stress components
    X = data[:, [3, 4, 8, 9, 10, 11]]                # features for model2A
    return X, y

def get_train_fw16_sig_model2B():
    data = load_sig_data(TRAIN16_SIG1, TRAIN16_SIG2)
    y = data[:, [0, 1]]
    # either use columns 3:18 or append smag/leith as needed
    X = data[:, 3:18]
    return X, y

# --- Signature models, filter width 32 ---
def get_train_fw32_sig_model2A():
    data = load_sig_data(TRAIN32_SIG1, TRAIN32_SIG2)
    y = data[:, [0, 1]]
    X = data[:, [4, 5, 9, 10, 11, 12]]
    return X, y

def get_train_fw32_sig_model2B():
    data = load_sig_data(TRAIN32_SIG1, TRAIN32_SIG2)
    y = data[:, [0, 1]]
    X = data[:, 4:19]
    return X, y

# --- Source-term models, filter width 16 ---
def get_train_fw16_source_model2B():
    data = load_source_data(TRAIN16_SOURCE)
    y = -data[:, 0].reshape(-1, 1)   # note the leading minus in your original
    X = data[:, 1:17]
    return X, y

# --- Source-term models, filter width 32 ---
def get_train_fw32_source_model2B():
    data = load_source_data(TRAIN32_SOURCE)
    y = -data[:, 0].reshape(-1, 1)
    X = data[:, 1:17]
    return X, y
