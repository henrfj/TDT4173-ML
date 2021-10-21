import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from scipy import stats

import seaborn as sns
import matplotlib.pylab as plt

plt.style.use('ggplot')

import math


def root_mean_squared_log_error(y_true, y_pred):
    # Alternatively: sklearn.metrics.mean_squared_log_error(y_true, y_pred) ** 0.5
    assert (y_true >= 0).all()
    assert (y_pred >= 0).all()
    log_error = np.log1p(y_pred) - np.log1p(y_true)  # Note: log1p(x) = log(1 + x)
    return np.mean(log_error ** 2) ** 0.5


def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + 'Feature Importance')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def outlier_rejection(data, outlier_value):
    data = data[(np.abs(stats.zscore(data['price'])) < outlier_value)]


def polar_coordinates(data):
    # Make a copy
    data_normed_r = data.copy()

    # Move origin to centre
    data_normed_r['latitude'] = data_normed_r['latitude'] - data_normed_r['latitude'].mean()
    data_normed_r['longitude'] = data_normed_r['longitude'] - data_normed_r['longitude'].mean()

    # Convert to polar coordinates
    data_normed_r['r'] = np.sqrt(data_normed_r['latitude'] ** 2 + data_normed_r['longitude'] ** 2)
    data_normed_r['theta'] = np.arctan(data_normed_r['longitude'] / data_normed_r['latitude'])
    return data_normed_r


def penthouse_features(data_set):
    data = data_set.copy()
    # New col - rel_height
    data['rel_height'] = data["floor"] / data["stories"]
    data['penthouse'] = data["floor"] == data["stories"]
    data['high_up'] = (data['rel_height'] > 0.5) & (data['stories'] > 10)
    data['real_penthouse'] = (data['stories'] > 10) & (data['penthouse'])

    # Elevator
    data['penthouse_e'] = (data['penthouse']) & (data['elevator_passenger'])
    data['high_up_e'] = (data['high_up']) & (data['elevator_passenger'])
    data['real_penthouse_e'] = (data['real_penthouse']) & (data['elevator_passenger'])

    # Lacks elevator
    data['penthouse_e_w'] = (data['penthouse']) & (data['elevator_without'])
    data['high_up_e_w'] = (data['high_up']) & (data['elevator_without'])
    data['real_penthouse_e_w'] = (data['real_penthouse']) & (data['elevator_without'])
    return data


def getAllMetadata():
    metaData_apartment = pd.read_json('../data/apartments_meta.json')
    metaData_building = pd.read_json('../data/buildings_meta.json')
    metaData_apartment.at[0, 'name'] = 'id'
    metaData_building.at[0, 'name'] = 'building_id'
    metaData = pd.concat([metaData_apartment, metaData_building])
    return metaData


def normalize(data):
    # Only normalize/scale the numerical data. Categorical data is kept as is.
    metadata = getAllMetadata()
    (nonCategorical, categorical) = getNonCategoricalAndCategoricalFeatures(metadata)
    data_norm = data[nonCategorical]

    # Normalize Training Data
    std_scale = preprocessing.StandardScaler().fit(data_norm)
    data_scaled = std_scale.transform(data_norm)

    # Re-enter procedure
    data_norm_col = pd.DataFrame(data_scaled, index=data_norm.index, columns=data_norm.columns)
    data.update(data_norm_col.drop(['id'], axis=1,inplace=True))


def getNonCategoricalAndCategoricalFeatures(metadata):
    categorical = []
    nonCategorical = []
    for _, feature in metadata.iterrows():
        if feature['type'] == 'categorical': categorical.append(feature['name'])
        else: nonCategorical.append(feature['name'])
    return nonCategorical, categorical


def categorical_to_numerical(data, features):
    le = LabelEncoder()
    for feature in features:
        data[feature] = le.fit_transform(data[feature])


def impute_seller(data):
    data['seller'] = data.groupby("new").transform(lambda x: x.fillna(x.median()))['seller']


def impute_median(data, features):
    for feature in features:
        data[feature].fillna(data[feature].median(), inplace=True)


def impute_drop(data, features):
    for feature in features:
        data[feature].dropna(inplace=True)


def handle_NaN(data, test=False):
    all_features = ['seller', 'area_total', 'area_kitchen', 'area_living',
                    'floor', 'rooms', 'layout', 'ceiling', 'bathrooms_shared',
                    'bathrooms_private', 'windows_court', 'windows_street', 'balconies',
                    'loggias', 'condition', 'phones', 'building_id', 'new', 'latitude',
                    'longitude', 'district', 'street', 'address', 'constructed', 'material', 'stories',
                    'elevator_without', 'elevator_passenger', 'elevator_service', 'parking', 'garbage_chute',
                    'heating']
    if not test:
        all_features.append('price')

    data = data[all_features]

    categorical_to_numerical(data, ['street', 'address'])

    handled_features = ['new', 'seller', 'latitude', 'longitude']

    impute_median(data, ['new'])

    if not test:
        impute_drop(data, ['latitude', 'longitude'])
    else:
        impute_median(data, ['latitude', 'longitude'])
    impute_seller(data)

    unhandled_features = [item for item in all_features if item not in handled_features]
    impute_median(data, unhandled_features)


def add_features(data, radius=False, penthouse=False):
    if radius:
        data = polar_coordinates(data)
    if penthouse:
        data = penthouse_features(data)
    return data


def load_and_handle(apartment, building, test=False, split_before_handle=False):
    data = pd.merge(apartment, building, left_on='building_id', right_on='id')
    data.rename(columns={'id_x': 'id'}, inplace=True)
    data.drop('id_y', axis=1, inplace=True)
    if not test:
        outlier_rejection(data, 7)
    if split_before_handle:
        X = data.drop(['price'], axis=1)
        y = data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        handle_NaN(X_train, test=True)
        handle_NaN(X_test, test=True)
        X_train = add_features(X_train, radius=True, penthouse=True)
        X_test = add_features(X_test, radius=True, penthouse=True)
        normalize(X_train)
        normalize(X_test)
        return X_train, X_test, y_train, y_test
    handle_NaN(data, test)
    data = add_features(data, radius=True, penthouse=True)
    normalize(data)
    return data


def load_and_handle_train(split_before_handle=False):
    train_apartment = pd.read_csv('../data/apartments_train.csv')
    train_building = pd.read_csv('../data/buildings_train.csv')
    if split_before_handle:
        return load_and_handle(train_apartment, train_building, test=False, split_before_handle=True)
    data = load_and_handle(train_apartment, train_building, test=False, split_before_handle=False)
    return data


def load_and_handle_test():
    test_apartment = pd.read_csv('../data/apartments_test.csv')
    test_building = pd.read_csv('../data/buildings_test.csv')
    data = load_and_handle(test_apartment, test_building, test=True)
    return data


train = load_and_handle_train()
print(train)