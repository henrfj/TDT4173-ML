import pandas as pd
import time
import random
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Specific tf libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout

# Train
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def plot_history(hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('MSLE')
    plt.yscale("log")
    plt.plot(hist['epoch'], hist['msle'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_msle'], label = 'Val Error')
    plt.legend()

def pre_process_numerical(features, Numerical_features, train, test, outliers_value=7,
 val_split=0.1, random_state=42):
    # Remove outlayers from training data
    no_outlayers = train[(np.abs(stats.zscore(train['price'])) < outliers_value)]

    # Training and validation data preprocessing
    labels = no_outlayers[features]
    labels = labels.fillna(labels.mean())
    targets = no_outlayers['price']

    # Test data preprocessing
    test_labels = test[features]
    test_labels = test_labels.fillna(test_labels.mean())

    # ADD R
    labels, test_labels = polar_coordinates(labels, test_labels)
    # ADD rel_height
    labels['rel_height'] = labels["floor"] / labels["stories"]
    test_labels['rel_height'] = test_labels["floor"] / test_labels["stories"]

    # Split
    train_labels, val_labels, train_targets, val_targets = train_test_split(
        labels, targets, test_size=val_split, shuffle= True, random_state=random_state)
    
    # Only normalize/scale the numerical data. Categorical data is kept as is.
    train_labels_n = train_labels.filter(Numerical_features)
    val_labels_n = val_labels.filter(Numerical_features)
    test_labels_n = test_labels.filter(Numerical_features)

    # Scale it.
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_labels_scaled = scaler.fit_transform(train_labels_n)
    val_labels_scaled = scaler.transform(val_labels_n)
    test_labels_scaled = scaler.transform(test_labels_n)

    # Re-enter proceedure
    training_norm_col = pd.DataFrame(train_labels_scaled, index=train_labels_n.index, columns=train_labels_n.columns) 
    train_labels.update(training_norm_col)

    val_norm_col = pd.DataFrame(val_labels_scaled, index=val_labels_n.index, columns=val_labels_n.columns) 
    val_labels.update(val_norm_col)

    testing_norm_col = pd.DataFrame(test_labels_scaled, index=test_labels_n.index, columns=test_labels_n.columns) 
    test_labels.update(testing_norm_col)

    # Drop the most correlated features.
    train_labels.drop(['longitude', 'latitude', 'area_kitchen', 'area_living', 'floor', 'stories'], inplace=True, axis=1)
    val_labels.drop(['longitude', 'latitude', 'area_kitchen', 'area_living', 'floor', 'stories'], inplace=True, axis=1)
    test_labels.drop(['longitude', 'latitude', 'area_kitchen', 'area_living', 'floor', 'stories'], inplace=True, axis=1)

    return train_labels, train_targets, val_labels, val_targets, test_labels

def get_cat_and_non_cat_data(metadata):
    categorical = []
    nonCategorical = []
    for _, feature in metadata.iterrows():
        if feature['type'] == 'categorical': categorical.append(feature['name'])
        else: nonCategorical.append(feature['name'])
    return nonCategorical, categorical

def polar_coordinates(labels, test):
    '''
    labels and input labels. Adds theta and also R to the dataframe copy.
    '''
    # Make a copy
    labels1_normed_r = labels.copy()
    test1_normed_r = test.copy()

    # Move origo to centre
    labels1_normed_r['latitude'] = labels1_normed_r['latitude'] - labels1_normed_r['latitude'].mean()
    labels1_normed_r['longitude'] = labels1_normed_r['longitude'] - labels1_normed_r['longitude'].mean()
    test1_normed_r['latitude'] = test1_normed_r['latitude'] -  test1_normed_r['latitude'].mean()
    test1_normed_r['longitude'] = test1_normed_r['longitude'] -  test1_normed_r['longitude'].mean()

    # Convert to polar coordinates
    labels1_normed_r['r'] =  np.sqrt(labels1_normed_r['latitude']**2 + labels1_normed_r['longitude']**2)
    labels1_normed_r['theta'] = np.arctan(labels1_normed_r['longitude']/labels1_normed_r['latitude'])
    test1_normed_r['r'] =  np.sqrt(test1_normed_r['latitude']**2 + test1_normed_r['longitude']**2)
    test1_normed_r['theta'] = np.arctan(test1_normed_r['longitude']/test1_normed_r['latitude'])

    return labels1_normed_r, test1_normed_r