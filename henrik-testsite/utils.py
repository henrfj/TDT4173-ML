import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
import numpy as np

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

def preProcess_numericalData(features, train, test, outliers_value=7, drop_nan=False):
    # Removing proce outlayers
    no_outlayers = train[(np.abs(stats.zscore(train[["price"]])) < outliers_value).all(axis=1)] 
    if drop_nan:
        no_outlayers = no_outlayers.dropna() #NB!

    # Labels and targets
    labels1 = no_outlayers[features]
    labels1 = labels1.fillna(labels1.mean())
    targets1= no_outlayers['price'] # Non nan values here.

    # Test
    test1 = test[features]
    test1 = test1.fillna(test1.mean())

    # Normalize
    normalized_labels1 = (labels1-labels1.min())/(labels1.max()-labels1.min())
    normalized_test1 = (test1-test1.min())/(test1.max()-test1.min())
    return normalized_labels1, normalized_test1, targets1


def polar_coordinates(labels, test):
    # Make a copy
    labels1_normed_r = labels.copy()
    test1_normed_r = test.copy()

    # Move origo to centre
    labels1_normed_r['latitude'] = labels1_normed_r['latitude'] -  labels1_normed_r['latitude'].mean()
    labels1_normed_r['longitude'] = labels1_normed_r['longitude'] -  labels1_normed_r['longitude'].mean()
    test1_normed_r['latitude'] = test1_normed_r['latitude'] -  test1_normed_r['latitude'].mean()
    test1_normed_r['longitude'] = test1_normed_r['longitude'] -  test1_normed_r['longitude'].mean()

    # Convert to polar coordinates
    labels1_normed_r['r'] =  np.sqrt(labels1_normed_r['latitude']**2 + labels1_normed_r['longitude']**2)
    labels1_normed_r['theta'] = np.arctan(labels1_normed_r['longitude']/labels1_normed_r['latitude'])
    test1_normed_r['r'] =  np.sqrt(test1_normed_r['latitude']**2 + test1_normed_r['longitude']**2)
    test1_normed_r['theta'] = np.arctan(test1_normed_r['longitude']/test1_normed_r['latitude'])

    return labels1_normed_r, test1_normed_r