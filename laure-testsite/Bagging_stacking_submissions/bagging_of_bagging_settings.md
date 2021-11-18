# bagging_of_bagging_3

```python
bags = [
    [
        {"path": main_path + 'Bag1/LGBM7.1.csv', 'score':0.16591},
        {"path": main_path + 'Bag1/sample_KNN_3.csv', 'score':0.33883}
    ],
    [
        {"path": main_path + 'Bag2/sample_RF_15.csv', 'score':0.20015},
        {"path": main_path + 'Bag2/Xgboost.csv', 'score':0.17781}
    ],
    [
        {"path": main_path + 'Bag4/deep_king_5_5.csv', 'score':0.20502},
        {"path": main_path + 'Bag3/CB5.1.csv', 'score':0.17140},
        {"path": main_path + 'Bag3/submission_gradientBoost.csv', 'score':0.19968}
    ]
]
```

# bagging_of_bagging_4

```python
bags = [
    [
        [
            {"path": main_path + 'Bag1/LGBM7.1.csv', 'score':0.16591},
            {"path": main_path + 'Bag1/sample_KNN_3.csv', 'score':0.33883}
        ],
        [
            {"path": main_path + 'Bag2/sample_RF_15.csv', 'score':0.20015},
            {"path": main_path + 'Bag2/Xgboost.csv', 'score':0.17781}
        ]
    ],
    [
        [
            {"path": main_path + 'Bag3/CB5.1.csv', 'score':0.17140},
            {"path": main_path + 'Bag3/submission_gradientBoost.csv', 'score':0.19968}
        ],
        [
            {"path": main_path + 'Bag4/deep_king_5_5.csv', 'score':0.20502},
            {"path": main_path + 'Bag4/GB5.1.csv', 'score':0.17172}
        ]
    ]
]
```

# bagging_of_bagging_5

```python
bags = [        
    [
        {"path": main_path + 'Bag1/LGBM7.1.csv', 'score':0.16591},
        {"path": main_path + 'Bag1/sample_KNN_3.csv', 'score':0.33883}
    ],
    [
        {"path": main_path + 'Bag2/sample_RF_15.csv', 'score':0.20015},
        {"path": main_path + 'Bag2/Xgboost.csv', 'score':0.17781}
    ],
    [
        {"path": main_path + 'Bag3/CB5.1.csv', 'score':0.17140},
        {"path": main_path + 'Bag3/submission_gradientBoost.csv', 'score':0.19968}
    ],
    [
        {"path": main_path + 'Bag4/deep_king_5_5.csv', 'score':0.20502},
        {"path": main_path + 'Bag4/GB5.1.csv', 'score':0.17172}
    ]        
]
```

# bagging_of_bagging_6

```python
bags = [   
    {"path": main_path + 'Bag1/LGBM7.1.csv', 'score':0.16591},
    {"path": main_path + 'Bag1/sample_KNN_3.csv', 'score':0.33883},
    {"path": main_path + 'Bag2/sample_RF_15.csv', 'score':0.20015},
    {"path": main_path + 'Bag2/Xgboost.csv', 'score':0.17781},
    {"path": main_path + 'Bag3/CB5.1.csv', 'score':0.17140},
    {"path": main_path + 'Bag3/submission_gradientBoost.csv', 'score':0.19968},
    {"path": main_path + 'Bag4/deep_king_5_5.csv', 'score':0.20502},
    {"path": main_path + 'Bag4/GB5.1.csv', 'score':0.17172}
]
```

# bagging_of_bagging_7
Weights = [6, 2]
```python
bags = [        
        [
            {"path": main_path + 'Bag1/LGBM7.1.csv', 'score':0.16591},
            {"path": main_path + 'Bag1/sample_KNN_3.csv', 'score':0.33883}
        ],
        [
            {"path": main_path + 'Bag2/sample_RF_15.csv', 'score':0.20015},
            {"path": main_path + 'Bag2/Xgboost.csv', 'score':0.17781}
        ],
        [
            {"path": main_path + 'Bag3/CB5.1.csv', 'score':0.17140},
            {"path": main_path + 'Bag3/submission_gradientBoost.csv', 'score':0.19968}
        ],
        [
            {"path": main_path + 'Bag4/deep_king_5_5.csv', 'score':0.20502},
            {"path": main_path + 'Bag4/GB5.1.csv', 'score':0.17172}
        ]        
    ]
```

# bagging_of_bagging_8
Weights = [4, 4]
```python
bags = [        
        [
            {"path": main_path + 'Bag1/LGBM7.1.csv', 'score':0.16591},
            {"path": main_path + 'Bag1/sample_KNN_3.csv', 'score':0.33883}
        ],
        [
            {"path": main_path + 'Bag2/sample_RF_15.csv', 'score':0.20015},
            {"path": main_path + 'Bag2/Xgboost.csv', 'score':0.17781}
        ],
        [
            {"path": main_path + 'Bag3/CB5.1.csv', 'score':0.17140},
            {"path": main_path + 'Bag3/submission_gradientBoost.csv', 'score':0.19968}
        ],
        [
            {"path": main_path + 'Bag4/deep_king_5_5.csv', 'score':0.20502},
            {"path": main_path + 'Bag4/GB5.1.csv', 'score':0.17172},
            {"path": main_path + 'Bag4/stacking_on_all_data_2.csv', 'score':0.16489}
        ]        
    ]
```

# bagging_of_bagging_9
Weights = [4, 4]
```python
bags = [        
        [
            {"path": main_path + 'Bag1/LGBM7.1.csv', 'score':0.16591},
            {"path": main_path + 'Bag1/sample_KNN_3.csv', 'score':0.33883}
        ],
        [
            {"path": main_path + 'Bag2/sample_RF_15.csv', 'score':0.20015},
            {"path": main_path + 'Bag2/Xgboost.csv', 'score':0.17781},
            {"path": main_path + 'Bag2/bagging_of_bagging_8.csv', 'score':0.16291}
        ],
        [
            {"path": main_path + 'Bag3/CB5.1.csv', 'score':0.17140},
            {"path": main_path + 'Bag3/submission_gradientBoost.csv', 'score':0.19968},
            {"path": main_path + 'Bag3/bagging_of_bagging_4.csv', 'score':0.16415}
        ],
        [
            {"path": main_path + 'Bag4/deep_king_5_5.csv', 'score':0.20502},
            {"path": main_path + 'Bag4/GB5.1.csv', 'score':0.17172},
            {"path": main_path + 'Bag4/stacking_on_all_data_2.csv', 'score':0.16489}
        ]        
    ]
```