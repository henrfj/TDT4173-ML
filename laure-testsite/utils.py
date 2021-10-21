from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def fillnaReg(df, X_features, y_feature):
    df = df.copy()
    df_temp = df[df[y_feature].notna()]
    if df_temp.shape[0] == 1: df_temp = df_temp.values.reshape(-1, 1)
    reg = LinearRegression().fit(df_temp[X_features], df_temp[y_feature])
    predict = reg.predict(df[X_features])
    df[y_feature] = np.where(df[y_feature]>0, df[y_feature], predict)
    return df

def getAllMetadata():
    metaData_apartment = pd.read_json('../data/apartments_meta.json')
    metaData_building = pd.read_json('../data/buildings_meta.json')
    metaData_apartment.at[0, 'name'] = 'apartment_id'
    metaData_building.at[0, 'name'] = 'building_id'
    metaData = pd.concat([metaData_apartment, metaData_building])
    return metaData

def getAllTrainData():
    train_apartment = pd.read_csv('../data/apartments_train.csv')
    train_building = pd.read_csv('../data/buildings_train.csv')
    train = pd.merge(train_apartment, train_building.set_index('id'), how='left', left_on='building_id', right_on='id')
    return train

def getAllTestData():
    test_apartment = pd.read_csv('../data/apartments_test.csv')
    test_building = pd.read_csv('../data/buildings_test.csv')
    test = pd.merge(test_apartment, test_building.set_index('id'), how='left', left_on='building_id', right_on='id')
    return test

def getFeatureImportanceGraph(model, X_train):
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=X_train.columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("MDI or Gini Importance")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

def getNonCategoricalAndCategoricalFeatures(metadata):
    categorical = []
    nonCategorical = []
    for _, feature in metadata.iterrows():
        if feature['type'] == 'categorical': categorical.append(feature['name'])
        else: nonCategorical.append(feature['name'])
    return nonCategorical, categorical

def oneHotFeature(metadata, data, feature):
    values = list(metadata.loc[metadata['name'] == feature]['cats'])[0]
    for i, value in enumerate(values):
        new_column = [1 if row == i else 0 for row in list(data[feature])]
        data[value] = new_column
    return values