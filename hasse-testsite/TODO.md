# TODO

* Figure out why LGBM was so terrible

## Features
* Fill NaN with LinearRegression (Laure utils)
* Further handling of NaN values

**Create some new features**

Possibilities:
* total_bathrooms
* total windows
* No. of apartments in building
* Spatiousness (done)
* ..?
* Street: extract Boulevard/Park...
    * Make one hot encoding

Other feature engineering: 
* One hot encoding
    * Ordinal categorical data
* Normalization
* Categorical to numerical (for street and address)
* Street names


## Models
* Finetune models (nothing done so far)
* Further explore other options
    * SVR
    * XGBoost
* STACKING
* CROSS VALIDATION
* Tune models using sklearn

* Other normalization?

* Remove super correlated features

* RSMLE

* LGBM tuning
    * Regularization:
        * lambda_l1
        * num_leaves
        * subsample
        * feature_fraction
        * max_depth
        * max_bin
    * Training
        * num_iterations
        * early_stopping_rounds
        * categorical_feature (don't use one hot!)
        * learning_rate = eta


## Submissions

Prepare
* Catboost submission with 5 top features
* Submissions with around 8-10 features
* XGBoost submission on selected features
* Gradient boost on carefully selected features and include new one hot encoding



## Split boosting models
| Model          | Responsible Person          |
| :---           |    :----:             | 
| Catboost       | Hasse    | 
| LightGBM       | Hasse    | 
| Gradient Boost | Hasse    | 
| Adaboost       |  ??   | 
| XGBoost        | Henrik   | 
| SVR            | Laure    | 
| KNN            | Laure    | 
| Decision Tree  | Laure    | 
| Deep NN        | Henrik   | 

* BAGGING in common_utils