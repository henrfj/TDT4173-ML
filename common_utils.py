import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from collections import Counter, defaultdict
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Specific tf libraries
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

def root_mean_squared_log_error(y_true, y_pred):
    # Alternatively: sklearn.metrics.mean_squared_log_error(y_true, y_pred) ** 0.5
    assert (y_true >= 0).all() 
    assert (y_pred >= 0).all()
    log_error = np.log1p(y_pred) - np.log1p(y_true)  # Note: log1p(x) = log(1 + x)
    return np.mean(log_error ** 2) ** 0.5

def categorical_to_numerical(data, features):
    le = LabelEncoder()
    for feature in features:
        data[feature] = le.fit_transform(data[feature])

def pre_process_numerical(features, numerical_features, train, test, metadata=[],
                    outliers_value=7, val_data=True, val_split=0.1, random_state=42, scaler="none",
                    add_R=False, add_rel_height=False, add_spacious=False, droptable=[],
                    one_hot_encode=True, cat_features=[], drop_old=True):
    """
    Pre processes pandas dataframe, returns split datasets with preprocessing applied
    to numerical data.

        args:
            - features, numerical features: list of all features and all numerical ones.
            - train and test: train and test dataset. Train also has target "price".
            - outliers_value: removes data outside the range of mean+outliers_value*std
            - val_data: bool to determine wheter or not you want validation data.
            - val_split: what percentage of data to use to validate.
            - scaler: none, minMax, or std. minMax scaled to range 1-0 and std scales around mean.
            - add_R if you want to add radius to dataset. add_rel_height to add rel height.
            - droptable: any features you want to drop at the end of preprocessing.
            Then for one-hot-encoding
            - Toogle it with "one_hot_encode"
            - Insert what cat-features you want to encode.
            - Drop_old is if you want to replace/delete the old categorical features,
            or keep them.
    """
    # Make a copy of data that is altered
    Numerical_features = numerical_features.copy()

    # Remove outlayers from training data
    no_outlayers = train[(np.abs(stats.zscore(train['price'])) < outliers_value)]

    # Training and validation data preprocessing
    labels = no_outlayers[features]
    labels = labels.fillna(labels.mean())
    targets = no_outlayers['price']

    # Test data preprocessing
    test_labels = test[features]
    test_labels = test_labels.fillna(test_labels.mean())

    if one_hot_encode and len(metadata):
        oneHotFeatures(metadata, labels, cat_features)
        oneHotFeatures(metadata, test_labels, cat_features)

    elif one_hot_encode:
        labels, test_labels = one_hot_encoder(labels, test_labels, cat_features, drop_old=drop_old)

    # Adding some new features
    # ADD R
    if add_R:
        labels, test_labels = polar_coordinates(labels, test_labels)
        Numerical_features.append("r")
    # ADD rel_height
    if add_rel_height:
        labels['rel_height'] = labels["floor"] / labels["stories"]
        test_labels['rel_height'] = test_labels["floor"] / test_labels["stories"]
        Numerical_features.append("rel_height")
    if add_spacious:
        labels['Spacious_rooms'] = labels['area_total'] /labels['rooms']
        test_labels['Spacious_rooms'] = test_labels['area_total'] /test_labels['rooms']
        Numerical_features.append("Spacious_rooms")


    # Split
    # TODO: dont split apartments of same building.
    
    if val_data:
        train_labels, val_labels, train_targets, val_targets = train_test_split(
            labels, targets, test_size=val_split, shuffle=True) #random_state=random_state
    else:
        train_labels = labels.copy()
        val_labels = labels.copy()
        train_targets = targets.copy()
        val_targets = targets.copy()


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
    elif scaler=="none":
        train_labels_scaled = train_labels_n
        val_labels_scaled = val_labels_n
        test_labels_scaled = test_labels_n
    else:
        assert ValueError, "Incorrect scaler"

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

    if val_data:
        return train_labels, train_targets, val_labels, val_targets, test_labels
    else:
        return train_labels, train_targets, test_labels


def one_hot_encoder(train_df, test_df, cat_features, drop_old=True):
    '''
    Returns a copy of all three dfs, after one-hot encoding and !removing!
    their old cat_features.

    NB! pd.get_dummies() does pretty much the same job!
    https://stackoverflow.com/questions/36285155/pandas-get-dummies
    BUG! Some categories are only present in train not test or the other way around!
        - Then the encoding is made differently for the two!
    https://stackoverflow.com/questions/57946006/one-hot-encoding-train-with-values-not-present-on-test
    '''
    # TODO: use Laures one-hot encoder.
    
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

def oneHotFeatures(metadata, data, features):
    values = []
    for feature in features:
        values.append(oneHotFeature(metadata, data, feature))
    return values

def oneHotFeature(metadata, data, feature):
    values = list(metadata.loc[metadata['name'] == feature]['cats'])[0]
    for i, value in enumerate(values):
        new_column = [1 if row == i else 0 for row in list(data[feature])]
        data[value] = new_column
    return values


def fillnaReg(df, X_features, y_feature):
    df = df.copy()
    df_temp = df[df[y_feature].notna()]
    if df_temp.shape[0] == 1: df_temp = df_temp.values.reshape(-1, 1)
    reg = LinearRegression().fit(df_temp[X_features], df_temp[y_feature])
    predict = reg.predict(df[X_features])
    df[y_feature] = np.where(df[y_feature]>0, df[y_feature], predict)
    return df


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

def lgbm_groupKFold(number_of_splits, model, X_train, y_train,
    eval_metric=None):  
    # y_train is log!!
    X_train = X_train.copy()
    y_train = y_train.copy()
    
    scores = []
    gkf = GroupKFold(n_splits=number_of_splits)
    groups = X_train["building_id"]
    best_score = 1
    i = 0
    
    for train_index, test_index in gkf.split(X_train, y_train, groups):
        X_train2, X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train2, y_test = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(
            X_train2,
            y_train2,
            eval_set=[(X_test, y_test)],
            eval_metric=eval_metric,
            verbose=False,
        )    
        prediction = np.exp(model.predict(X_test))
        score = root_mean_squared_log_error(prediction, np.exp(y_test))
        if score <  best_score:
            best_score = score
            best_model = model
            best_index = i
        scores.append(score)
        i += 1
    return scores, np.average(scores), best_model, best_index

def XGB_groupKFold(number_of_splits, model, X_train, y_train,
    eval_metric=None):  
    ''' y_train needs to be log. Model trains to predict logs now!'''
    X_train = X_train.copy()
    y_train = y_train.copy()
    
    scores = []
    gkf = GroupKFold(n_splits=number_of_splits)
    groups = X_train["building_id"]
    best_score = 1
    i = 0
    
    for train_index, test_index in gkf.split(X_train, y_train, groups):
        X_train2, X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train2, y_test = y_train.iloc[train_index], y_train.iloc[test_index]
        model.fit(
            X_train2,
            y_train2,
            eval_set=[(X_test, y_test)],
            eval_metric=eval_metric,
            early_stopping_rounds=15,
            verbose=False,
        )    
        prediction = np.exp(model.predict(X_test))
        score = root_mean_squared_log_error(prediction, np.exp(y_test))
        if score <  best_score:
            best_score = score
            best_model = model
            best_index = i
        scores.append(score)
        i += 1
    return scores, np.average(scores), best_model, best_index

def custom_asymmetric_eval(y_true, y_pred):
    loss = root_mean_squared_log_error(y_true,y_pred)
    return "custom_asymmetric_eval", np.mean(loss), False

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

# Attempt at homemade.
def rmsle_custom(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return K.sqrt(msle(y_true, y_pred))


def load_all_data(fraction_of_data=1, apartment_id='apartment_id'):
    # Metadata
    metaData_apartment = pd.read_json('../data/apartments_meta.json')
    metaData_building = pd.read_json('../data/buildings_meta.json')
    metaData = pd.concat([metaData_apartment, metaData_building])

    # Train
    train_apartment = pd.read_csv('../data/apartments_train.csv')
    train_building = pd.read_csv('../data/buildings_train.csv')
    train = pd.merge(train_apartment, train_building, left_on='building_id', right_on='id')
    train.rename(columns={'id_x' : apartment_id}, inplace=True)
    train.drop('id_y', axis=1, inplace=True)
    train = train.head(int(train.shape[0] * fraction_of_data))

    # Test
    test_apartment = pd.read_csv('../data/apartments_test.csv')
    test_building = pd.read_csv('../data/buildings_test.csv')
    test = pd.merge(test_apartment, test_building, left_on='building_id', right_on='id')
    test.rename(columns={'id_x' : apartment_id}, inplace=True)
    test.drop('id_y', axis=1, inplace=True)

    return train, test, metaData

def predict_and_store(model, test_labels, test_pd, path="default", exponential=False):
    '''
        Inputs
        - test_pd needs to be the original full test dataframe
    '''
    result = model.predict(test_labels)
    if exponential:
        result = np.exp(result)
    submission = pd.DataFrame()
    submission['id'] = test_pd['apartment_id']
    submission['price_prediction'] = result
    if len(submission['id']) != 9937:
        raise Exception("Not enough rows submitted!")
    submission.to_csv(path, index=False)


def create_ANN_model(dense_layers=[64, 64, 64], activation=tf.nn.leaky_relu,
                     dropout=[False, False, False], dropout_rate=0.2, optimizer='adam',
                      loss_function=rmsle_custom, metrics=['accuracy'], output_activation=True):
    # Model
    model = tf.keras.Sequential()
    for i, n in enumerate(dense_layers):
        model.add(Dense(n, activation=activation))
        if dropout[i]:
            model.add(Dropout(dropout_rate))
    
    if output_activation:
        model.add(Dense(1, activation=activation))
    else:
        model.add(Dense(1)) #Output

    # Optimized for reducing msle loss.
    model.compile(optimizer=optimizer, 
                loss=loss_function, #'msle', 'rmse', RMSLETF, rmsle_custom
                metrics=metrics) # metrics=['mse', 'msle'] metrics=[tf.keras.metrics.Accuracy()]

    return model

def csv_bagging(kaggle_scores, csv_paths, submission_path):
    # Making the acc dataframe
    d = {}
    for i, score in enumerate(kaggle_scores):
        d[i] = score
    acc = pd.DataFrame(
    d,
    index=[0]
    )
    acc = acc.T
    acc.columns = ['RMSLE']
    acc

    # Read dataframes, sort and store
    pd_predictions = []
    for path in csv_paths:
        pd_predictions.append(
            pd.read_csv(path).sort_values(by="id")
            )
    # Cast to numpy
    np_predictions = []
    for pred in pd_predictions:
        np_predictions.append(
            pred["price_prediction"].to_numpy().T
        )

    # Bagging
    avg_prediction = np.average(
        np_predictions,
        weights = 1 / acc['RMSLE'] ** 4,
        axis=0
        )
    
    result = avg_prediction
    submission = pd.DataFrame()
    submission['id'] = pd_predictions[0]['id']
    submission['price_prediction'] = result
    if len(submission['id']) != 9937:
        raise Exception("Not enough rows submitted!")
    
    submission.to_csv(submission_path, index=False)


