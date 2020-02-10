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

# Get EURUSD One Hour OHLC dataset
dl = DataLoader()
eurusd_h1 = dl.get(Constants.EURUSD, Constants.ONE_HOUR, start=dt.datetime(year=2010, month=1, day=1))
eurusd_h1 = eurusd_h1.drop(columns=[k for k in eurusd_h1.keys() if k.startswith('ask')])
print(eurusd_h1.head(5))

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

# Group candlestick data into sets of `lookback` size
def group_ohlc_generator(data, lookback):
	arr = []
	for i in range(len(data)-lookback):
		temp_list = []
		for candle in data[i:i+lookback]:
			temp_list.append(candle)
		arr.append(temp_list)
	return np.array(arr)

lookback = 5

cell_timer = MeasureTime()
three_dim_sequence = group_ohlc_generator(eurusd_h1[1:].values, lookback)
cell_timer.kill()

# print(three_dim_sequence[:2])
print(three_dim_sequence.shape)

# Visualize candlestick sets

# counter=0
# for candle in three_dim_sequence[1000:1005]:
# 	counter += 1
# 	print('Step ' + str(counter))
# 	graph_data_ohlc(candle)

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

# Test feature engineering
# ohlc = three_dim_sequence[1000:1010][5][1]
# print(ohlc)
# candle = ohlc_to_candlestick(ohlc)
# print(candle)

def candlestick_generator(data, lookback):
	arr = []
	for i in range(len(data)-lookback):
		temp_list = []
		for candle in data[i:i+lookback]:
			candlestick = ohlc_to_candlestick(candle)
			temp_list.append(candlestick)
		arr.append(temp_list)
	return np.array(arr)

cell_timer = MeasureTime()
three_dim_sequence_candle = candlestick_generator(eurusd_h1.values[1:], lookback)
cell_timer.kill()

# print(three_dim_sequence_candle[:2])
print(three_dim_sequence_candle.shape)

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
			temp_list = scaler.fit_transform(templist)

		X.append(temp_list)
		X_raw.append(temp_list_raw)

		candlestick_prediction = ohlc_to_candlestick(data[i+lookback])
		pred = candlestick_prediction[0]
		y.append(pred)

	return np.array(X), np.array(y), np.array(X_raw)

cell_timer = MeasureTime()
X, y, X_raw = X_Y_candlestick_generator(eurusd_h1.values[1:], lookback)
cell_timer.kill()

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
model.add(tf.keras.layers.LSTM(units=64, dropout=0.1))
# model.add(layers.LSTM(units=128, dropout=0.1))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model.summary()

# Split data into train and test sets
cell_timer = MeasureTime()

X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_val_and_test, y_val_and_test, test_size=0.5)

X_train_raw, X_val_and_test_raw = train_test_split(X_raw, test_size=0.3)
X_val_raw, X_test_raw = train_test_split(X_val_and_test_raw, test_size=0.5)

cell_timer.kill()

print('Training data: ' + 'X Input shape: ' + str(X_train.shape) + ', ' + 'y Output shape: ' + str(y_train.shape))
print('Validation data: ' + 'X Input shape: ' + str(X_val.shape) + ', ' + 'y Output shape: ' + str(y_val.shape))
print('Test data: ' + 'X Input shape: ' + str(X_test.shape) + ', ' + 'y Output shape: ' + str(y_test.shape))

# Train model
cell_timer = MeasureTime()
history = model.fit(X_train, y_train, batch_size=500, epochs=20, validation_data=(X_val, y_val))
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
alpha_distance = 0.35

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

# def get_feature_importance(model, X_train_dataset, feature_names):
# 	pred_x = model.predict(X_train_dataset)

# 	random_ind = np.random.choice(X_train.shape[0], 1000, replace=False)
# 	data = X_train[random_ind[:500]]
# 	e = shap.DeepExplainer(model, data)
# 	test1 = X_train[random_ind[500:1000]]
# 	shap_val = e.shap_values(test1)
# 	shap_val = np.array(shap_val)
# 	shap_val = np.reshape(shap_val, (int(shap_val.shape[1]), int(shap_val.shape[2]), int(shap_val.shape[3])))
# 	shap_abs = np.absolute(shap_val)
# 	sum_0 = np.sum(shap_abs, axis=0)
# 	x_pos = [i for i, _ in enumerate(f_names)]

# 	plt.figure(figsize=(10,6))

# 	plt1 = plt.subplot(4,1,1)
# 	plt1.barh(x_pos, sum_0[2])
# 	plt1.set_yticks(x_pos)
# 	plt1.set_yticklabels(feature_names)
# 	plt1.set_title('features of last candle')

# 	plt2 = plt.subplot(4,1,2,sharex=plt1)
# 	plt2.barh(x_pos, sum_0[1])
# 	plt2.set_yticks(x_pos)
# 	plt2.set_yticklabels(feature_names)
# 	plt2.set_title('features of last candle -1')

# 	plt3 = plt.subplot(4,1,3, sharex=plt1)
# 	plt3.barh(x_pos, sum_0[0])
# 	plt3.set_yticks(x_pos)
# 	plt3.set_ytickslabels(feature_names)
# 	plt3.set_title('features of last candle -2')

# 	plt.tight_layout()
# 	plt.show()

# cell_timer = MeasureTime()
# features_list = ['candle type', 'wicks up', 'wicks down', 'body size']
# get_feature_importance(model, X_train, features_list)
# cell_timer.kill()



