import pandas as pd
from catboost import CatBoostClassifier, Pool
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

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

label = 'home_team_win'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
test_df_2024 = pd.read_csv(test_file_path_2024)

X_train, X_val, y_train, y_val = train_test_split(
    train_df.drop([label], axis = 1), train_df[label], test_size = 0.2, random_state = 1126)

model = CatBoostClassifier(
    iterations = 2000,
    depth = 6,
    learning_rate = 1e-3,
    use_best_model = True,
    random_seed = 1126,
)

train_pool = Pool(X_train, label = y_train)
val_pool = Pool(X_val, label = y_val)

test_pool = Pool(test_df.drop([label], axis = 1), label = test_df[label])
test_pool_2024 = Pool(test_df_2024.drop([label], axis = 1), label = test_df_2024[label])

model.fit(train_pool, eval_set = val_pool)

metrics = model.eval_metrics(
    test_pool,
    metrics = ['Accuracy', 'BalancedAccuracy', 'AUC', 'MCC', 'F1', 'Precision', 'Recall'], 
)
metrics_2024 = model.eval_metrics(
    test_pool_2024,
    metrics = ['Accuracy', 'BalancedAccuracy', 'AUC', 'MCC', 'F1', 'Precision', 'Recall'], 
)

# print(metrics)
output_dict = {}
print('final_result:')
for m, v in metrics.items():
    print(f'{m}: {v[-1]}')
    output_dict[m] = v[-1]
output_dict_2024 = {}
print('final_result_2024:')
for m, v in metrics_2024.items():
    print(f'{m}: {v[-1]}')
    output_dict_2024[m] = v[-1]

Path('./result').mkdir(parents = True, exist_ok = True)
with open( './result/hopefully_catboost_result.txt', mode = 'w+' ) as f:
    # print(output_dict, file = f)
    json.dump(output_dict, f, indent = 4)
    json.dump(output_dict_2024, f, indent = 4)