import datetime as dt
import matplotlib
import pandas as pd
import itertools
import tensorflow as tf
import time
import shap
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc

import Constants
from DataLoader import DataLoader

class MeasureTime():
	def __init__(self):
		self.start = time.time()

	def kill(self):
		print('Time elapsed: ' + time.strftime("%H:%M:%S", time.gmtime(time.time() - self.start)))

notebook_timer = MeasureTime()
notebook_timer.kill()

'''
Data Preprocessing and Visualization
'''

products = [
	Constants.GBPUSD, Constants.EURUSD, Constants.AUDUSD, 
	# Constants.EURJPY, Constants.USDCAD, Constants.NZDUSD,
	# Constants.USDJPY, Constants.USDCHF
]

# Get EURUSD One Hour OHLC dataset
dl = DataLoader()

all_df = []
for product in products:
	df = dl.get(product, Constants.FOUR_HOURS, start=dt.datetime(year=2005, month=1, day=1))
	df = df.drop(columns=[k for k in df.keys() if k.startswith('ask')])
	all_df.append(df)

print(all_df[0].head(5))

# Graphs OHLC data
def graph_data_ohlc(dataset):
	fig = plt.figure()
	ax1 = plt.subplot2grid((1,1), (0,0))
	openp = dataset[:,[0]]
	highp = dataset[:,[1]]
	lowp = dataset[:,[2]]
	closep = dataset[:,[3]]
	date = range(len(closep))

	ohlc = []
	for x in date:
		ohlc.append((date[x], openp[x], highp[x], lowp[x], closep[x]))
		x+=1
	candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#77d879', colordown='#db3f3f')
	for label in ax1.xaxis.get_ticklabels():
		label.set_rotation(45)
	ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
	ax1.grid(True)
	plt.xlabel('Candle')
	plt.ylabel('Price')
	plt.title('Candlestick sample representation')

	plt.subplots_adjust(left=0.09, bottom=0.20, right=0.94, top=0.90, wspace=0.2, hspace=0)
	plt.show()

# graph_data_ohlc(eurusd_h1.tail(1000).values)

'''
Feature Engineering
'''

def ohlc_to_candlestick(conversion_array):
	candlestick_data = [0]*4

	if conversion_array[3]>conversion_array[0]:
		candle_type = 1
		wicks_up = abs(conversion_array[1]-conversion_array[3])
		wicks_down = abs(conversion_array[2]-conversion_array[0])
		body_size = abs(conversion_array[3]-conversion_array[0])

	else:
		candle_type = 0
		wicks_up = abs(conversion_array[1]-conversion_array[0])
		wicks_down = abs(conversion_array[2]-conversion_array[3])
		body_size = abs(conversion_array[0]-conversion_array[3])

	candlestick_data[0] = candle_type
	candlestick_data[1] = round(wicks_up*10000,2)
	candlestick_data[2] = round(wicks_down*10000,2)
	candlestick_data[3] = round(body_size*10000,2)

	return candlestick_data

def X_Y_candlestick_generator(data, lookback, MinMax=False):
	if MinMax: scaler = preprocessing.MinMaxScaler()
	X = []
	X_raw = []
	y = []
	for i in range(len(data)-lookback):
		temp_list = []
		temp_list_raw = []
		for candle in data[i:i+lookback]:
			candlestick = ohlc_to_candlestick(candle)
			temp_list.append(candlestick)
			temp_list_raw.append(candle)

		if MinMax:
			temp_list = scaler.fit_transform(temp_list)

		X.append(temp_list)
		X_raw.append(temp_list_raw)

		candlestick_prediction = ohlc_to_candlestick(data[i+lookback])
		pred = candlestick_prediction[0]
		y.append(pred)

	return np.array(X), np.array(y), np.array(X_raw)

cell_timer = MeasureTime()
lookback = 5
X = []
y = []
X_raw = []
for df in all_df:
	temp_X, temp_y, temp_X_raw = X_Y_candlestick_generator(df.values[1:], lookback, MinMax=True)
	X.append(temp_X)
	y.append(temp_y)
	X_raw.append(temp_X_raw)

X = np.concatenate(X)
y = np.concatenate(y)
X_raw = np.concatenate(X_raw)
cell_timer.kill()

# print('Visualize data:')
# print(X[:2])
# print(X[-2:])
# print(y[:5])
# print(X_raw[:2])
# print(X_raw[-2:])
# print('--|\n')

print('\nShape of X: {}\nShape of y: {}\nShape of X_raw: {}'.format(
	X.shape, y.shape, X_raw.shape
))

print('Examples:\n{}\n{}\n{}'.format(X[1], y[1], X_raw[1]))

unique, counts = np.unique(y, return_counts=True)
predictions_type = dict(zip(unique, counts))
print('Bull: {} percent: {:0.2%}'.format(predictions_type[1], predictions_type[1]/len(y)))
print('Bear: {} percent: {:0.2%}'.format(predictions_type[0], predictions_type[0]/len(y)))
print('Total: {}'.format(len(y)))

# Visualization of how model works
# for i in range(5):
# 	x = i+1000
# 	if y[x] == 1: print('Prediction Bullish')
# 	if y[x] == 0: print('Prediction Bearish')
# 	graph_data_ohlc(X_raw[x])

'''
Build the Model
'''

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=32, dropout=0.1, return_sequences=True, input_shape=(None, X.shape[-1])))
model.add(tf.keras.layers.LSTM(units=64, dropout=0.1, return_sequences=True))
model.add(tf.keras.layers.LSTM(units=64, dropout=0.1))
model.add(tf.keras.layers.Dense(units=1, activation='relu'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.summary()

# Split data into train and test sets
cell_timer = MeasureTime()

X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5, random_state=1)

X_train_raw, X_val_and_test_raw = train_test_split(X_raw, test_size=0.3)
X_val_raw, X_test_raw = train_test_split(X_val_and_test_raw, test_size=0.5)

cell_timer.kill()

print('Training data: ' + 'X Input shape: ' + str(X_train.shape) + ', ' + 'y Output shape: ' + str(y_train.shape))
print('Validation data: ' + 'X Input shape: ' + str(X_val.shape) + ', ' + 'y Output shape: ' + str(y_val.shape))
print('Test data: ' + 'X Input shape: ' + str(X_test.shape) + ', ' + 'y Output shape: ' + str(y_test.shape))

# Train model
cell_timer = MeasureTime()
history = model.fit(X_train, y_train, batch_size=1000, epochs=25, validation_data=(X_val, y_val))
cell_timer.kill()

# Chart 1 - Model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Chart 2 - Model Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Test model
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Make some predictions
cell_timer = MeasureTime()
counter = 0
won = 0
lost = 0
predictions = model.predict(X_test)
alpha_distance = 0.40


for i in range(len(predictions)):
	pred = predictions[i][0]
	if pred > (1-alpha_distance):
		# if y_test[i] == 1: print('Answer: [Bullish]\n Guess:[Bullish]')
		# elif y_test[i] == 0: print('A: [Bearish]\n Guess:[Bullish]')

		if y_test[i] == 1:
			won+=1
		else:
			lost+=1

	elif pred < alpha_distance:
		# if y_test[i] == 1: print('Answer: [Bullish]\n Guess:[Bearish]')
		# elif y_test[i] == 0: print('A: [Bearish]\n Guess:[Bearish]')

		if y_test[i] == 0:
			won+=1
		else:
			lost+=1

print('Won: {} Lost: {}'.format(won, lost))
print('Success rate: {:.2%}'.format(won/(won+lost)))
cell_timer.kill()

