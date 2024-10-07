import pandas as pd

from sklearn.model_selection import train_test_split

# train_file_path = '../dataset/train_dataset.csv'
# test_file_path = '../dataset/test_dataset.csv'
train_file_path = f'../slight_mix_dataset/train_dataset.csv'
test_file_path = f'../slight_mix_dataset/test_dataset.csv'

train_df = pd.read_csv(train_file_path).drop(['Unnamed: 0'], axis = 1)
test_df = pd.read_csv(test_file_path)

df = pd.concat([train_df, test_df], ignore_index = True, sort = False)

train_sets = {}
test_sets = {}

df['year'] = pd.to_datetime(df['date']).dt.year
for year in df['year'].unique():
    year_data = df[df['year'] == year]
    train, test = train_test_split(year_data, test_size = 0.3, random_state = 1126)
    train_sets[year] = train
    test_sets[year] = test

train_df = pd.concat(train_sets.values()).drop(['year', 'date'], axis = 1)
test_df = pd.concat(test_sets.values()).drop(['year', 'date'], axis = 1)

print(train_df)
print(test_df)

train_df.to_csv('./slight_train_dataset.csv', index = False)
test_df.to_csv('./slight_test_dataset.csv', index = False)
