import pandas as pd
from catboost import CatBoostClassifier, Pool
from pathlib import Path
import json

# train_file_path = f'./mixed_year/train_dataset.csv'
# test_file_path = f'./mixed_year/test_dataset.csv'
# train_file_path = f'./pre_dataset/train_dataset.csv'
# test_file_path = f'./pre_dataset/test_dataset.csv'
train_file_path = f'./slight_mix_dataset/train_dataset.csv'
test_file_path = f'./slight_mix_dataset/test_dataset.csv'
train_file_path = f'./pre_dataset/slight_train_dataset.csv'
test_file_path = f'./pre_dataset/slight_test_dataset.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

model = CatBoostClassifier(
    iterations = 1000,
)

label = 'home_team_win'
train_pool = Pool(train_df.drop([label], axis = 1), label = train_df[label])
test_pool = Pool(test_df.drop([label], axis = 1), label = test_df[label])

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
with open( './result/slight_catboost_result.txt', mode = 'w+' ) as f:
    # print(output_dict, file = f)
    json.dump(output_dict, f, indent = 4)