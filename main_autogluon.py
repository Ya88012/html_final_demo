import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from pathlib import Path
import json

# train_file_path = f'./mixed_year/train_dataset.csv'
# test_file_path = f'./mixed_year/test_dataset.csv'
# train_file_path = f'./pre_dataset/train_dataset.csv'
# test_file_path = f'./pre_dataset/test_dataset.csv'
# train_file_path = f'./slight_mix_dataset/train_dataset.csv'
# test_file_path = f'./slight_mix_dataset/test_dataset.csv'
# train_file_path = f'./pre_dataset/slight_train_dataset.csv'
# test_file_path = f'./pre_dataset/slight_test_dataset.csv'
train_file_path = f'./pre_dataset/train_dataset.csv'
test_file_path = f'./pre_dataset/test_dataset.csv'
test_file_path_2024 = f'./pre_dataset/test_dataset_2024.csv'

train_ds = TabularDataset(train_file_path)
test_ds = TabularDataset(test_file_path)
test_ds_2024 = TabularDataset(test_file_path_2024)

# train_ds = train_ds.drop(['Unnamed: 0'], axis = 1)
# train_ds = train_ds.drop(['date'], axis = 1)
train_ds = train_ds.sample(frac = 1)
# test_ds = test_ds.drop(['date'], axis = 1)

label = 'home_team_win'

predictor = TabularPredictor(label = label).fit(train_ds, holdout_frac = 0.2)

eval_result = predictor.evaluate(test_ds)
y_pred = predictor.predict(test_ds)
eval_result_2024 = predictor.evaluate(test_ds_2024)
y_pred_2024 = predictor.predict(test_ds_2024)


print('eval_result:')
print(eval_result)
print('eval_result_2024:')
print(eval_result_2024)

Path('./result').mkdir(parents = True, exist_ok = True)
with open( './result/hopefully_autogluon_result.txt', mode = 'w+' ) as f:
    # print(eval_result, file = f)
    json.dump(eval_result, f, indent = 4)
    json.dump(eval_result_2024, f, indent = 4)
