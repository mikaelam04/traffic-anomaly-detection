# dataset.py

import scipy.io
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def load_traffic_data(mat_path):

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    mat = scipy.io.loadmat(mat_path)

    X_train_raw = mat['tra_X_tr']
    Y_train_raw = mat['tra_Y_tr'].T
    X_test_raw = mat['tra_X_te']
    Y_test_raw = mat['tra_Y_te'].T

    adj_matrix = mat['tra_adj_mat']

    # ==========================================================
    # CONVERT SPARSE → DENSE
    # ==========================================================

    X_train = np.array([X_train_raw[0, i].toarray() for i in range(X_train_raw.shape[1])])
    X_test = np.array([X_test_raw[0, i].toarray() for i in range(X_test_raw.shape[1])])

    # ==========================================================
    # COLUMN NAMES
    # ==========================================================

    feature_cols = (
        [f'traffic_t-{i}' for i in range(10, 0, -1)]
        + [f'weekday_{i}' for i in range(7)]
        + [f'hour_{i}' for i in range(24)]
        + ['dir_N', 'dir_S', 'dir_E', 'dir_W']
        + ['num_lanes']
        + ['road_name_1', 'road_name_2'])
    
    sensor_cols = [f'sensor_{i+1}' for i in range(36)]

    # ==========================================================
    # FLATTEN SENSOR DIMENSION
    # ==========================================================

    n_train, n_sensors, n_feats = X_train.shape
    n_test = X_test.shape[0]

    X_train_flat = X_train.reshape(-1, n_feats)
    X_test_flat = X_test.reshape(-1, n_feats)

    X_train_df = pd.DataFrame(X_train_flat, columns = feature_cols)
    X_test_df = pd.DataFrame(X_test_flat, columns = feature_cols)

    # ==========================================================
    # FIX DATASET COLUMN ISSUE
    # ==========================================================

    tmp = X_train_df['num_lanes'].copy()
    X_train_df['num_lanes'] = X_train_df['hour_22']
    X_train_df['hour_22'] = tmp

    tmp = X_test_df['num_lanes'].copy()
    X_test_df['num_lanes'] = X_test_df['hour_22']
    X_test_df['hour_22'] = tmp

    # ==========================================================
    # SCALE NUM_LANES
    # ==========================================================

    X_train_df['num_lanes'] = (X_train_df['num_lanes'] - 1) / 4
    X_test_df['num_lanes'] = (X_test_df['num_lanes'] - 1) / 4

    # ==========================================================
    # TARGET DATA
    # ==========================================================

    Y_train_df = pd.DataFrame(Y_train_raw, columns = sensor_cols)
    Y_test_df = pd.DataFrame(Y_test_raw, columns = sensor_cols)

    # ==========================================================
    # SCALE TARGETS
    # ==========================================================

    y_scaler = MinMaxScaler()

    Y_train_scaled = y_scaler.fit_transform(Y_train_df)
    Y_test_scaled = y_scaler.transform(Y_test_df)

    # ==========================================================
    # RESHAPE FOR LSTM
    # ==========================================================

    X_train_clean = X_train_df.values.reshape(n_train, n_sensors, n_feats)
    X_test_clean = X_test_df.values.reshape(n_test, n_sensors, n_feats)

    X_tr = torch.FloatTensor(X_train_clean.reshape(-1, 1, n_feats))
    Y_tr = torch.FloatTensor(Y_train_scaled.reshape(-1))
    X_te = torch.FloatTensor(X_test_clean.reshape(-1, 1, n_feats))
    Y_te = torch.FloatTensor(Y_test_scaled.reshape(-1))

    return (
        X_tr,
        Y_tr,
        X_te,
        Y_te,
        y_scaler,
        Y_train_scaled,
        Y_test_scaled,
        sensor_cols,
        n_train,
        n_test,
        n_sensors,
        n_feats,
        adj_matrix
    )