from autogluon.tabular import TabularDataset, TabularPredictor
from pathlib import Path

train_file_path = f'./dataset/train_dataset.csv'
test_file_path = f'./dataset/test_dataset.csv'

train_ds = TabularDataset(train_file_path).drop('Unnamed: 0', axis = 1)
test_ds = TabularDataset(test_file_path)
label = 'home_team_win'

'''
Do preprocessing.
'''

predictor = TabularPredictor(label = label).fit(train_ds)

y_pred = predictor.predict(test_ds)
eval_result = predictor.evaluate(test_ds)
print(eval_result)
Path('./result').mkdir(parents = True, exist_ok = True)
with open( './result/autogluon_result.txt', mode = 'w+' ) as f:
    print(eval_result, file = f)
