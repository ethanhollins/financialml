"""

This model attempts to predict how many pips the market
will move in a direction before a reversal based on
candlestick analysis data.

Created by: Ethan Hollins

	v2 - Feeding only train data that has higher than
		 20 (pip_threshold) pip move

"""
from DataLoader import DataLoader
import Constants
import datetime as dt
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class timeit(object):
	def __init__(self):
		self.start = time.time()
		print('Started timer.')
	
	def end(self):
		print('Finished in {:.2f}s\n'.format(time.time() - self.start))

timer = timeit()
timer.end()

'''
Data Preprocessing
'''

dl = DataLoader()
df = dl.get(Constants.GBPUSD, Constants.ONE_HOUR, start=dt.datetime(year=2010, month=1, day=1))
df = df.drop(columns=[k for k in df.keys() if k.startswith('ask')])

print(df.head(5))

'''
Feature Engineering
'''

# Hyper parameters
num_candles = 48
time_series = 24
pip_threshold = 10

learning_rate = 0.005
momentum = 0.1

def convertToPips(x):
	return round(x * 1000, 2)

def getCandlestickData(candle):
	if candle[3] >= candle[0]:
		return [
			1, # Bullish
			convertToPips(candle[1] - candle[3]), # Wick up
			convertToPips(candle[2] - candle[0]), # Wick down
			convertToPips(candle[3] - candle[0]) # Body size
		]
	else:
		return [
			0, # Bearish
			convertToPips(candle[1] - candle[0]), # Wick up
			convertToPips(candle[2] - candle[3]), # Wick down
			convertToPips(candle[0] - candle[3]) # Body size
		]

def getPipMove(data):
	long_move = 0
	short_move = 0
	close = data[0][3]
	for candle in data:
		high_move = convertToPips(candle[1] - close)
		low_move = convertToPips(candle[2] - close)
		if high_move > long_move: long_move = high_move
		if low_move < short_move: short_move = low_move
	return long_move if long_move > abs(short_move) else short_move

def getTrainData():
	X = []
	y = []
	for i in range(time_series, df.shape[0]-1):
		pip_move = getPipMove(df.values[i+1:i+1+time_series])
		if pip_move >= pip_threshold or pip_move <= -pip_threshold:
			temp_candlesticks = []
			for j in range(num_candles-1,-1,-1):
				temp_candlesticks.append(
					getCandlestickData(df.values[i-j])
				)

			X.append(temp_candlesticks)
			y.append(pip_move)
	return np.array(X), np.array(y)

timer = timeit()
X, y = getTrainData()
timer.end()

print('X example:\n{}'.format(X[0]))
print('X shape: {}'.format(X.shape))
print('y example: {}'.format(y[:5]))
print('y shape: {}\n'.format(y.shape))

'''
Split and normalize data
'''

# Normalize by mean and std
def normalize(i):
	return (i - np.mean(i)) / np.std(i)

def denormalize(i, mean, std):
	return i * std + mean

X_norm = np.copy(X)
y_norm = np.copy(y)
X_norm[:,:,1:] = normalize(X_norm[:,:,1:])
y_norm = normalize(y_norm)

# Visualize normalized data
print('X normalized:\n{}\n'.format(X_norm[0]))
print('y normalized:\n{}'.format(y_norm[:5]))

print('y denormalized: {}\n'.format(
	denormalize(y_norm[:5], np.mean(y), np.std(y))
))

# Visualize data
print(
	'Average Move: {:.2f}\n'\
	'Average Long Move: {:.2f}\n'\
	'Average Short Move: {:.2f}\n'.format(
		np.mean(y),
		np.mean(list(filter(lambda x: x > 0, y))),
		np.mean(list(filter(lambda x: x < 0, y)))
	)
)

timer = timeit()
X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X_norm, y_norm, test_size=0.3, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5, random_state=1)
timer.end()

print('Train shape: {}\nTest shape: {}\nVal shape: {}\n'.format(
	X_train.shape, X_test.shape, X_val.shape
))

"""
Build model
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=64, dropout=0.1, return_sequences=True))
model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
model.add(tf.keras.layers.LSTM(units=64))
model.add(tf.keras.layers.Dense(1))

model.compile(
	optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
	loss=tf.keras.losses.mean_squared_error
)

timer = timeit()
history = model.fit(X_train, y_train, batch_size=1000, epochs=20, validation_data=(X_val, y_val))
timer.end()

# Plot model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

test_loss = model.evaluate(X_test, y_test)
print('Test Loss: {:.4f}'.format(test_loss))

# Plot histogram of predictions
predictions = model.predict(X_test)
predictions = denormalize(predictions[:,0], np.mean(y), np.std(y))
print(predictions)

plt.hist(predictions)
plt.title('Pip Predictions')
plt.ylabel('Count')
plt.xlabel('Pips')
plt.show()
