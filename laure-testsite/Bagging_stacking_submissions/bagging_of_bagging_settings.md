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