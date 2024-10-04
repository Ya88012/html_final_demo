import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from pathlib import Path

# train_file_path = f'./dataset/train_dataset.csv'
train_file_path = f'./my_dataset/train_dataset.csv'
# test_file_path = f'./dataset/test_dataset.csv'
test_file_path = f'./my_dataset/test_dataset.csv'
test_label_path = f'./my_dataset/test_label.csv'

train_ds = TabularDataset(train_file_path).drop('date', axis = 1)
test_ds = pd.read_csv(test_file_path)
test_label = pd.read_csv(test_label_path).drop('id', axis = 1)
label = 'home_team_win'

t = pd.concat([test_ds, test_label], axis = 1)

predictor = TabularPredictor(label = label).fit(train_ds)
# predictor = TabularPredictor.load('./AutogluonModels/ag-20241004_114459')

# y_pred = predictor.predict(test_ds)
eval_result = predictor.evaluate(t)

print(eval_result)
Path('./result').mkdir(parents = True, exist_ok = True)
with open( './result/autogluon_result.txt', mode = 'w+' ) as f:
    print(eval_result, file = f)
