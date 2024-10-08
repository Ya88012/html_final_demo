import pandas as pd

_1 = pd.read_csv('./same_season_test_data.csv')
_2 = pd.read_csv('./same_season_test_label.csv').drop(['id'], axis = 1)

_ = pd.concat([_1, _2], axis = 1)

print(_)
_.to_csv('./same_season_test_dataset.csv', index = False)

__1 = pd.read_csv('./2024_test_data.csv')
__2 = pd.read_csv('./2024_test_label.csv').drop(['id'], axis = 1)

__ = pd.concat([__1, __2], axis = 1)

print(__)
__.to_csv('./2024_test_dataset.csv', index = False)
