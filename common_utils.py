import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def one_hot_encoder(train_df, test_df, cat_features, drop_old=True):
    '''
    Returns a copy of all three dfs, after one-hot encoding and !removing!
    their old cat_features.

    BUG! Some categories are only present in train not test or the other way around!
        - Then the encoding is made differently for the two!
    https://stackoverflow.com/questions/57946006/one-hot-encoding-train-with-values-not-present-on-test
    '''
    
    if(len(train_df.isna())!=0 or len(train_df.isna())!=0 or len(train_df.isna())!=0):
        assert ValueError

    train_labels = train_df.copy()
    test_labels = test_df.copy()

    encoded_features = []
    dfs=[train_labels, test_labels]
    for df in dfs:
        for feature in cat_features:
            encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
            n = df[feature].nunique()
            cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)

    n = len(cat_features)

    train_labels = pd.concat([train_labels, *encoded_features[ : n]], axis=1)
    test_labels = pd.concat([test_labels, *encoded_features[n : ]], axis=1)


    # Now drop the non-encoded ones!
    if drop_old:
        train_labels.drop(cat_features, inplace=True, axis=1)
        test_labels.drop(cat_features, inplace=True, axis=1)
    return train_labels, test_labels

def pre_process_numerical(features, Numerical_features, train, test,
                    outliers_value=7, val_split=0.1, random_state=42, scaler="none",
                    add_R="False", add_rel_height="False", droptable=[],
                    one_hot_encode=True, cat_features=[], drop_old=True):
    """
    Pre processes pandas dataframe, returns split datasets with preprocessing applied
    to numerical data.

        args:
            - features, numerical features: list of all features and all numerical ones.
            - train and test: train and test dataset. Train also has target "price".
            - outliers_value: removes data outside the range of mean+outliers_value*std
            - val_split: what percentage of data to use to validate.
            - scaler: none, minMax, or std. minMax scaled to range 1-0 and std scales around mean.
            - add_R if you want to add radius to dataset. add_rel_height to add rel height.
            - droptable: any features you want to drop at the end of preprocessing.
    """

    # Remove outlayers from training data
    no_outlayers = train[(np.abs(stats.zscore(train['price'])) < outliers_value)]

    # Training and validation data preprocessing
    labels = no_outlayers[features]
    labels = labels.fillna(labels.mean())
    targets = no_outlayers['price']

    # Test data preprocessing
    test_labels = test[features]
    test_labels = test_labels.fillna(test_labels.mean())

    if one_hot_encode:
        labels, test_labels = one_hot_encoder(labels, test_labels, cat_features, drop_old=drop_old)

    # ADD R
    if add_R:
        labels, test_labels = polar_coordinates(labels, test_labels)
    # ADD rel_height
    if add_rel_height:
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
    if scaler=="minMax":
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_labels_scaled = scaler.fit_transform(train_labels_n)
        val_labels_scaled = scaler.transform(val_labels_n)
        test_labels_scaled = scaler.transform(test_labels_n)
    elif scaler=="std":
        std_scale = preprocessing.StandardScaler().fit(train_labels_n)
        train_labels_scaled = std_scale.transform(train_labels_n)
        val_labels_scaled = std_scale.transform(val_labels_n)
        test_labels_scaled = std_scale.transform(test_labels_n)
    else:
        train_labels_scaled = train_labels_n
        val_labels_scaled = val_labels_n
        test_labels_scaled = test_labels_n
        

    # Re-enter proceedure
    training_norm_col = pd.DataFrame(train_labels_scaled, index=train_labels_n.index, columns=train_labels_n.columns) 
    train_labels.update(training_norm_col)

    val_norm_col = pd.DataFrame(val_labels_scaled, index=val_labels_n.index, columns=val_labels_n.columns) 
    val_labels.update(val_norm_col)

    testing_norm_col = pd.DataFrame(test_labels_scaled, index=test_labels_n.index, columns=test_labels_n.columns) 
    test_labels.update(testing_norm_col)

    # Drop the most correlated features.
    train_labels.drop(droptable, inplace=True, axis=1)
    val_labels.drop(droptable, inplace=True, axis=1)
    test_labels.drop(droptable, inplace=True, axis=1)

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
