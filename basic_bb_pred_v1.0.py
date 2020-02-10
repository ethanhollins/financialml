import Constants
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime as dt

from DataLoader import DataLoader
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

'''
Data Preprocessing
'''

dl = DataLoader()
df_gbpusd_h4 = dl.get(Constants.GBPUSD, Constants.FOUR_HOURS, start=dt.datetime(year=2010, month=1, day=1))
df_gbpusd_d = dl.get(Constants.GBPUSD, Constants.DAILY, start=dt.datetime(year=2010, month=1, day=1))
# df_eurusd_h4 = dl.get(Constants.EURUSD, Constants.FOUR_HOURS, start=dt.datetime(year=2015, month=1, day=1))
# df_eurusd_d = dl.get(Constants.EURUSD, Constants.DAILY, start=dt.datetime(year=2015, month=1, day=1))

data = df_gbpusd_h4[['bid_close']]
label_data = df_gbpusd_d[['bid_open', 'bid_close']]

print(data.head())
print(data.shape)
print(label_data.tail())
print(label_data.shape)

X = []
y = []
num_bars = 20
for i in range(len(label_data.index)):
	key = label_data.index[i]
	if key in data.index:
		try:
			idx = int(data.index.get_loc(key))
		except:
			continue
	elif key - 100800 in data.index:
		try:
			idx = int(data.index.get_loc(key - 100800))
		except:
			continue
	else:
		continue

	if idx >= num_bars and i+1 < label_data.shape[0]:
		temp = data.iloc[idx-num_bars:idx]
		scaler = MinMaxScaler()
		scaler.fit(temp)
		temp = scaler.transform(temp)

		X.append(temp)
		y.append(0 if label_data.iloc[i+1][0] <= label_data.iloc[i+1][1] else 1)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print()
# print(X[0])
print('X shape: {}'.format(X.shape))
# print(y[-5:])
print('y shape: {}'.format(y.shape))

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.3, random_state=1
)

'''
Data Model Creation
'''

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(num_bars, len(data.keys()))),
	# keras.layers.Dropout(0.1),
	keras.layers.Dense(256, activation='relu'),
	keras.layers.Dropout(0.1),
	keras.layers.Dense(256, activation='relu'),
	keras.layers.Dropout(0.1),
	keras.layers.Dense(256, activation='relu'),
	keras.layers.Dropout(0.1),
	keras.layers.Dense(2, activation='softmax')
])

model.summary()

model.compile(
	optimizer='adam',
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=20)
test_loss, test_acc = model.evaluate(X_test, y_test)

print('{} - {}'.format(test_loss, test_acc))
