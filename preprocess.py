from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import pandas as pd
from pathlib import Path

# train_file_path = f'./mixed_year/train_dataset.csv'
# test_file_path = f'./mixed_year/test_dataset.csv'
train_file_path = f'./mixed_year/slight_train_dataset.csv'
test_file_path = f'./mixed_year/slight_test_dataset.csv'

Path('./pre_dataset').mkdir(parents = True, exist_ok = True)
# pre_train_file_path = f'./pre_dataset/train_dataset.csv'
# pre_test_file_path = f'./pre_dataset/test_dataset.csv'
pre_train_file_path = f'./pre_dataset/slight_train_dataset.csv'
pre_test_file_path = f'./pre_dataset/slight_test_dataset.csv'

train_ds = TabularDataset(train_file_path)
# train_ds = train_ds.drop(['Unnamed: 0'], axis = 1)
test_ds = pd.read_csv(test_file_path)

train_gen = AutoMLPipelineFeatureGenerator()
pre_train_ds = train_gen.fit_transform(X = train_ds)
pre_train_ds.to_csv(pre_train_file_path, index = False)

test_gen = AutoMLPipelineFeatureGenerator()
pre_test_ds = test_gen.fit_transform(X = test_ds)
pre_test_ds.to_csv(pre_test_file_path, index = False)

print(pre_train_ds)
print(pre_test_ds)
