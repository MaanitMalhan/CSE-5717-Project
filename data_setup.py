import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
electric_data = pd.read_csv('energy_dataset.csv', parse_dates=['time'])

# set time column to datetime
electric_data['time'] = pd.to_datetime(electric_data['time'], utc=True).dt.tz_convert(None)

# drop columns with all NaN values or all 0 values
cols_to_keep = ['time'] + [
    col for col in electric_data.columns 
    if col != 'time' and not (electric_data[col].isna().all() or (electric_data[col].fillna(0) == 0).all())
]
electric_data = electric_data[cols_to_keep]

# drop rows with NaN values
electric_data = electric_data.dropna()

# set time column as index
electric_data.set_index('time', inplace=True)

# enforce time continuity
full_index = pd.date_range(start=electric_data.index.min(), end=electric_data.index.max(), freq='H')
electric_data = electric_data.reindex(full_index)
electric_data = electric_data.dropna()

# separate feature and target columns, also drop price and load statistics
target_column = 'price actual'
feature_columns = [col for col in electric_data.columns if col != target_column and col != 'price day ahead']

# standardize features and target
feature_means = electric_data[feature_columns].mean()
feature_stds = electric_data[feature_columns].std()
electric_data[feature_columns] = (electric_data[feature_columns] - feature_means) / feature_stds

# standardize target
target_mean = electric_data[target_column].mean()
target_std = electric_data[target_column].std()
electric_data[target_column] = (electric_data[target_column] - target_mean) / target_std

# save normalization statistics
feature_means.to_csv("feature_means.csv")
feature_stds.to_csv("feature_stds.csv")
pd.Series({'mean': target_mean, 'std': target_std}).to_csv("target_stats.csv")

# create empty model dataset
electric_model_dataset = []
length = len(electric_data)

# create model dataset
for i in range(length - 12):
    if i % 12 != 0:
        continue
    target_price = electric_data.iloc[i + 12][target_column]
    window = electric_data.iloc[i:i+12][feature_columns].to_numpy()
    electric_model_dataset.append((target_price, window))
    print(f"{i + 1} / {length - 24}\r", end='')

# save model dataset
electric_model_dataset = np.array(electric_model_dataset, dtype=object)
np.save('electric_model_dataset.npy', electric_model_dataset, allow_pickle=True)
