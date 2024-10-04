from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

train_file_path = f'./dataset/train_dataset.csv'
test_file_path = f'./dataset/test_dataset.csv'

pre_train_file_path = f'./dataset/pre_train_dataset.csv'
pre_test_file_path = f'./dataset/pre_test_dataset.csv'

train_ds = TabularDataset(train_file_path).drop('Unnamed: 0', axis = 1)
test_ds = TabularDataset(test_file_path)

train_gen = AutoMLPipelineFeatureGenerator()
pre_train_ds = train_gen.fit_transform(X = train_ds)
pre_train_ds.to_csv(pre_train_file_path, index = False)

test_gen = AutoMLPipelineFeatureGenerator()
pre_test_ds = test_gen.fit_transform(X = test_ds)
pre_test_ds.to_csv(pre_test_file_path, index = False)

print(pre_train_ds)
print(pre_test_ds)
