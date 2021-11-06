# TODO

## Deep learning
1. Find the 0.5 score ceiling problem. What went wrong?
   1. Using Grouped K-fold we get scores of 1.1 and 1.3 but get a 0.5-0.25 kaggle score
   2. Loss function that is used for training (custom_RMSLE) even gives 1.3 loss on val data...
   Despite the fact that training history shows it as 0.24-0.20 at last epoch...
2. Extract the best, 0.22 model! (done!)

## Gradient boosting
1. Hyperparameter tuning: needs to be done for each set of features?
2. Are there more hyperaparameters that can be tuned?
3. Avoid overfitting in the XGBoost k-fold function.

## Ensemble
1. Watch the part II lecture.
2. Stack some models! Wohoo :D
   1. Maybe area total is too important? Remove it and train models without.
   Then stack them with models that only use area total, or something like that.

## EDA
1. New features rule! Just see gradient boosting feature importance :O
   (Maybe area-total is too important?)
   1. Mansion: only apartment in the building, and area_total is in the top 10% percentile!
   2. Single family dwelling: Only apartment in the building, above average area total.
   3. Student dwelling/studio apartments: one room apartments: one with private and one with shared bathrooms.
2. Clean the data:
   1. Use other cols to fill nans, and use median not mean for categories.
      1. Mean for categories will just be useless data for the model anyways.
   2. 