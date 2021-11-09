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
* STACKING

* Other normalization?

* Remove super correlated features

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

* Gradient Boosting Regressor Tuning
    * Tree-specific
        * min_samples_split
        * min_samples_leaf / min_weight_fraction_leaf
        * max_depth / max_leaf_nodes
        * max_features
    * Boosting
        * learning_rate
        * n_estimators
        * subsample
    * Miscellaneous
        * loss
        * init
        * random_state
        * verbose
        * warm_start
        * presort

    * General approach:
        1. Choose a relatively high learning rate. Generally the default value of 0.1 works but somewhere between 0.05 to 0.2 should work for different problems
        2. Determine the optimum number of trees for this learning rate. This should range around 40-70. Remember to choose a value on which your system can work fairly fast. This is because it will be used for testing various scenarios and determining the tree parameters.
        3. Tune tree-specific parameters for decided learning rate and number of trees. Note that we can choose different parameters to define a tree and I’ll take up an example here.
        4. Lower the learning rate and increase the estimators proportionally to get more robust models.



## Split boosting models
| Model          | Responsible Person          |
| :---           |    :----:             | 
| **Catboost**       | Hasse    | 
| **LightGBM**       | Hasse    | 
| **Gradient Boost** | Hasse    | 
| Adaboost       |  ??      | 
| XGBoost        | Henrik   | 
| SVR            | Laure    | 
| KNN            | Laure    | 
| Decision Tree  | Laure    | 
| Deep NN        | Henrik   | 
| AutoML         | Laure    | 

* BAGGING in common_utils