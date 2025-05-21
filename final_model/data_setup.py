import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

# Load data
energy = pd.read_csv('energy_dataset.csv')
weather = pd.read_csv('weather_features.csv')

# Process energy
energy['time'] = pd.to_datetime(energy['time'], utc=True) + pd.DateOffset(hours=1)
energy = energy.drop(['generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead'], axis=1)

useless_feats = [col for col in energy.columns if energy[col].max() == 0.0]
energy = energy.drop(useless_feats, axis=1)
energy = energy.drop(['forecast solar day ahead', 'forecast wind onshore day ahead', 'total load forecast', 'price day ahead'], axis=1)

# Process weather
weather = weather.rename(columns={'dt_iso': 'time'})
weather = weather.drop_duplicates(subset=['time', 'city_name'], keep='first')
weather['time'] = pd.to_datetime(weather['time'], utc=True) + pd.DateOffset(hours=1)
weather = weather.drop(['rain_3h', 'snow_3h'], axis=1)
weather = weather.drop(['weather_id', 'weather_main', 'weather_icon'], axis=1)
weather.loc[weather.pressure > 1050, 'pressure'] = np.nan
weather.loc[weather.pressure < 950, 'pressure'] = np.nan
weather.loc[weather.wind_speed > 25, 'wind_speed'] = np.nan
weather['pressure'] = weather['pressure'].interpolate(method='linear', limit_direction='forward')
weather['wind_speed'] = weather['wind_speed'].interpolate(method='linear', limit_direction='forward')
weather = weather.drop(['temp_min', 'temp_max'], axis=1)

# Merge weather by city
weathergy = energy.copy()
for city, df_city in weather.groupby('city_name'):
    df_city = df_city.add_suffix(f'_{city}')
    df_city = df_city.rename(columns={f'time_{city}': 'time'})
    df_city = df_city.drop(f'city_name_{city}', axis=1)
    weathergy = pd.merge(weathergy, df_city, on='time', how='outer')

# Interpolate missing values
null_columns = weathergy.columns[weathergy.isnull().any()]
for col in null_columns:
    weathergy[col] = weathergy[col].interpolate(method='polynomial', order=5)

# Encode categorical features
categorical_feats = weathergy.select_dtypes(include=['object']).columns
lab_enc = LabelEncoder()
for feat in categorical_feats:
    weathergy[feat] = lab_enc.fit_transform(weathergy[feat].astype(str))

# Feature merges
weathergy['generation fossil coal'] = weathergy['generation fossil brown coal/lignite'] + weathergy['generation fossil hard coal']
weathergy = weathergy.drop(['generation fossil brown coal/lignite', 'generation fossil hard coal'], axis=1)

cities = [' Barcelona', 'Bilbao', 'Madrid', 'Seville', 'Valencia']
weathergy['temp_spain'] = weathergy[[f'temp_{c}' for c in cities]].mean(axis=1)
weathergy = weathergy.drop([f'temp_{c}' for c in cities], axis=1)

weathergy['pressure_Madrid_Valencia'] = weathergy['pressure_Madrid'] + weathergy['pressure_Valencia']
weathergy['pressure_Madrid_Valencia'] /= 2
weathergy = weathergy.drop(['pressure_Madrid', 'pressure_Valencia'], axis=1)

weathergy['humidity_Madrid_Seville'] = weathergy['humidity_Madrid'] + weathergy['humidity_Seville']
weathergy['humidity_Madrid_Seville'] /= 2
weathergy = weathergy.drop(['humidity_Madrid', 'humidity_Seville'], axis=1)

# Drop low-correlation features
low_corr_feats = ['humidity_ Barcelona', 'clouds_all_ Barcelona', 'humidity_Bilbao', 'rain_1h_Madrid',
                  'rain_1h_Seville', 'rain_1h_Valencia', 'weather_description_Valencia']
weathergy = weathergy.drop(low_corr_feats, axis=1)

# Temporal features
weathergy['hour'] = weathergy['time'].dt.hour
weathergy['weekday'] = weathergy['time'].dt.dayofweek
weathergy['month'] = weathergy['time'].dt.month

weathergy['business'] = weathergy['hour'].apply(lambda h: 2 if 8 <= h <= 14 or 16 <= h <= 21 else (1 if 14 < h <= 16 else 0))
weathergy['weekend'] = weathergy['weekday'].apply(lambda w: 2 if w == 6 else (1 if w == 5 else 0))
weathergy = weathergy.drop(['hour', 'weekday'], axis=1)

# Prepare for LSTM
weathergy.index = list(range(len(weathergy)))
weathergy_time = weathergy.drop('time', axis=1)
scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit_transform(weathergy_time.drop('price actual', axis=1))
scaled_df = pd.DataFrame(scaled_features, columns=weathergy_time.columns[:-1])

# LSTM sequence preparation
window_size = 24
X, Y = [], []
for i in range(len(scaled_df) - window_size):
    X.append(scaled_df.iloc[i:i+window_size].values)
    Y.append(weathergy_time['price actual'].iloc[i + window_size])

X = np.array(X)
Y = np.array(Y)

np.save('X_lstm.npy', X)
np.save('y_lstm.npy', Y)
print("Saved LSTM data")
