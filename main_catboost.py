import pandas as pd
from catboost import CatBoostClassifier, Pool
from pathlib import Path

train_file_path = f'./dataset/pre_train_dataset.csv'
test_file_path = f'./dataset/pre_test_dataset.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

model = CatBoostClassifier(
    iterations = 10,
    task_type = 'GPU',
    devices = '0',
)

label = 'home_team_win'
train_pool = Pool(train_df.drop(label, axis = 1), label = train_df[label])
test_pool = Pool(test_df.drop(label, axis = 1), label = test_df[label])

model.fit(train_pool)

metrics = model.eval_metrics(
    test_pool,
    metrics = ['Accuracy', 'BalancedAccuracy', 'AUC', 'MCC', 'F1', 'Precision', 'Recall'], 
)

# print(metrics)

output_dict = {}
print('final_result:')
for m, v in metrics.items():
    print(f'{m}: {v[-1]}')
    output_dict[m] = v[-1]

Path('./result').mkdir(parents = True, exist_ok = True)
with open( './result/catboost_result.txt', mode = 'w+' ) as f:
    print(output_dict, file = f)