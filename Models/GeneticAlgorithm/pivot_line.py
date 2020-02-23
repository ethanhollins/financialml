from DataLoader import DataLoader
from numba import jit
import GA
import Constants
import numpy as np
import datetime as dt
import time
import pandas as pd
import tensorflow as tf

class timeit(object):
	def __init__(self):
		self.start = time.time()
		print('Timer module started.')
	def end(self):
		print('Timer module finished %.2f'%(time.time() - self.start))

'''
Data Preprocessing
'''

dl = DataLoader()

timer = timeit()
df_m1 = dl.get(Constants.GBPUSD, Constants.ONE_MINUTE, start=dt.datetime(2015,1,1))
df_m1 = df_m1[['bid_high', 'bid_low', 'bid_close']]
df_d = dl.get(Constants.GBPUSD, Constants.DAILY, start=dt.datetime(2015,1,1))
df_d = df_d[['bid_high', 'bid_low', 'bid_close']]
timer.end()

# Visualize 1 minute data
print('M1:\n%s'%df_m1.head(5))
print(df_m1.shape)

# Visualize Daily data
print('\nD:\n%s'%df_d.head(5))
print(df_d.shape)
print()

'''
Feature Engineering
'''

# Convert price to pips
def convertToPips(x):
	return np.around(x * 1000, 2)

@jit
def getRsiData(close, period):
	X = np.zeros((close.shape[0]-period), dtype=np.float32)
	gain = np.zeros((close.shape[0]-period), dtype=np.float32)
	loss = np.zeros((close.shape[0]-period), dtype=np.float32)
	
	for i in range(period, close.shape[0]):
		gain_sum = 0.0
		loss_sum = 0.0
		if i > period:
			p_gain = gain[i-period-1]
			p_loss = loss[i-period-1]

			chng = close[i] - close[i-1]
			if chng >= 0:
				gain_sum += chng
			else:
				loss_sum += np.absolute(chng)

			gain_avg = (p_gain * (period-1) + gain_sum)/period
			loss_avg = (p_loss * (period-1) + loss_sum)/period

		else:
			for j in range(0, i):
				if j != 0:
					chng = close[j] - close[j-1]

					if chng >= 0:
						gain_sum += chng
					else:
						loss_sum += np.absolute(chng)

			gain_avg = gain_sum / period
			loss_avg = loss_sum / period

		gain[i-period] = gain_avg
		loss[i-period] = loss_avg

		if loss_avg == 0.0:
			X[i-period] = 100
		else:
			X[i-period] = 100 - (100 / (1 + gain_avg/loss_avg))
	return X

@jit
def getCciData(high, low, close, period):
	X = np.zeros((close.shape[0]-period), dtype=np.float32)
	
	for i in range(period, close.shape[0]):
		# Calculate Typical price
		c_typ = (high[i] + low[i] + close[i])/3.0
		typ_sma = 0.0
		for j in range(i-period, i):
			typ_sma += (high[j] + low[j] + close[j])/3.0

		typ_sma /= period
		
		# Calculate Mean Deviation
		mean_dev = 0.0
		for j in range(i-period, i):
			mean_dev += np.absolute(
				((high[j] + low[j] + close[j])/3.0) - typ_sma
			)

		mean_dev /= period
		const = .015

		if mean_dev == 0:
			X[i-period] = 0
			continue

		X[i-period] = (c_typ - typ_sma) / (const * mean_dev)
	return X

rsi_period = 10
cci_period = 5
timer = timeit()
rsi_data = getRsiData(df_m1.values[:,2], rsi_period)
cci_data = getCciData(
	df_m1.values[:,0], df_m1.values[:,1], df_m1.values[:,2],
	cci_period
)

m1_off = max(rsi_period, cci_period)

df_m1_feat = pd.DataFrame(data={
	'RSI':rsi_data[m1_off-rsi_period:],
	'CCI':cci_data[m1_off-cci_period:]
}, index= df_m1.index.values[m1_off:])
timer.end()

print(df_m1_feat.head(5))
print(df_m1_feat.shape)

# Get distance from previous `lookup` num cpps to current day close
def distFromCpp(data, lookup):
	X = []
	for i in range(lookup+1, data.shape[0]):
		c_lookup = []
		for l in range(lookup-1,-1,-1):
			high = data[i-1-l][0]
			low = data[i-1-l][1]
			close = data[i-1-l][2]
			cpp_dist = round(data[i][2] - (high + low + close)/3,5)
			c_lookup.append(convertToPips(cpp_dist))
		X.append(c_lookup)
	return np.array(X)

lookup = 3

timer = timeit()
cpp_data = distFromCpp(df_d.values, lookup)
timer.end()

# Visualize Central Pivot Point data
print('CPP Data:\n%s'%cpp_data[:5])


def getTimestampRanges(timestamps, start_off, end_off):
	X = []
	for ts in timestamps:
		time = dl.convertTimestampToTime(ts)
		time = dl.convertTimezone(time, 'Europe/London')
		start_time = time + dt.timedelta(hours=start_off)
		end_time = time + dt.timedelta(hours=end_off)
		start_time = dl.convertTimezone(start_time, 'Australia/Melbourne')
		end_time = dl.convertTimezone(end_time, 'Australia/Melbourne')
		X.append([
			dl.convertTimeToTimestamp(start_time),
			dl.convertTimeToTimestamp(end_time),
		])
	return X

def normalize(x):
	return (x - np.mean(x)) / np.std(x)

def getTrainData(m1_data, m1_lookup, cpp_data, ts_ranges):
	cpp_data = normalize(cpp_data)
	m1_data.values[:,0] = normalize(m1_data.values[:,0])
	m1_data.values[:,1] = normalize(m1_data.values[:,1])

	X = [[] for i in range(m1_data.shape[1]+1)]
	for i in range(len(ts_ranges)):
		ts_r = ts_ranges[i]
		cpp = cpp_data[i]
		data = m1_data[(m1_data.index > ts_r[0]) & (m1_data.index < ts_r[1])].values
		if data.shape[0] > 0:
			X[0] += ([cpp] * (data.shape[0]-m1_lookup))
			for m1_i in range(data.shape[1]):
				for m1_j in range(m1_lookup, data.shape[0]):
					X[m1_i+1].append(data[m1_j-m1_lookup:m1_j,m1_i])
	X = [np.array(x, dtype=np.float64) for x in X]
	return [x.reshape(x.shape[0], x.shape[1], 1) for x in X]

timer = timeit()
ts_ranges = getTimestampRanges(df_d.index.values, 8, 16)
timer.end()

timer = timeit()
train_data = getTrainData(df_m1_feat, 10, cpp_data, ts_ranges[lookup+1:])
timer.end()

print('\nTrain Data:')
for i in range(len(train_data)):
	print('{} - {}'.format(
		train_data[i].shape,
		train_data[i][0]
	))

class MultiInputDenseModel(tf.keras.Model):
	def __init__(self, layers, input_size):
		super(MultiInputDenseModel, self).__init__()
		self.dense_layers = []
		for i in range(len(layers)):
			self.dense_layers.append(tf.keras.layers.Dense(layers[i]))
		if len(self._layers) % 2 == 0:
			self.dense_layers.append(tf.keras.layers.Dense(input_size))
		self.dense_out = tf.keras.layers.Dense(1)

	def call(self, inpt):
		mul_layers = []

		for i in range(len(inpt)):
			mul_layers.append(
				self.dense_layers[i](tf.transpose(inpt[i]))
			)

		x = tf.transpose(mul_layers[0])
		for i in range(1, len(mul_layers)):
			if i % 2 != 0:
				x = tf.matmul(x, mul_layers[i])
				x = tf.nn.relu(x)
			else:
				x = tf.matmul(x, tf.transpose(mul_layers[i]))
				x = tf.nn.relu(x)
		if len(inpt) < len(self._layers):
			x = tf.matmul(x, tf.transpose(self._layers[-1]))

		x = self.dense_out(tf.transpose(x))
		return tf.nn.tanh(x)

class CustomModel(GA.GeneticAlgorithmModel):
	
	def __init__(self):
		super().__init__()

	def __call__(self):
		super().__call__()
		return CustomModel.run()

	def generateModel(self, mean, std):
		return MultiInputDenseModel([16,16], mean, std)

	def run():
		return


# Test Model
# for _ in range(20):
result = []
model = MultiInputDenseModel([16,16,16], 1)
for i in range(train_data[0].shape[0]):
	x = model([
		train_data[0][i],
		train_data[1][i],
		train_data[2][i]
	])

	result.append(x)

timer = timeit()
print(np.mean(result))
timer.end()




