# Features based on catboost
Selected features = `['area_total','r','floor','district','parking']`

| Model          | All Features          | Selected Features Adaboost  | Selected Features Gradient Boost  |
| :---           |    :----:             |          :---:       |         ---:       |
| Catboost       | 0.2263792791606827    | 0.2506950121943363  |                     |
| LightGBM       | 0.2470070706304115    | 0.2646902188546250  |                     |
| Gradient Boost | 0.1811148103176012    | 0.2598197783068658  | 0.20183055476440248 |
| Adaboost       | 2.0818536706130706    | 1.7092256548384503  |                     |

All of them get worse when selecting less features, except for Adaboost (which has the worst score anyway).

But based on submissions made on 14.10.2021 the public leaderboard score becomes BETTER with less features.

Best submission up till now (20.10.2021) was using Gradient Boost. So we'll make another submission with Gradient Boost, but on only a few features. Select based on own feature importance:

Selected features = `['area_total','r','constructed','district','theta']`

Note, the feature importance in `feature_selection.ipynb` is `['area_total','r','district','latitude','constructed','theta']` but since r and theta already implement `latitude` we'll ignore this one.
