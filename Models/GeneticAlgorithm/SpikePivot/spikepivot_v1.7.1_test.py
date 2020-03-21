import sys
sys.path.append('../')

from DataLoader import DataLoader
from numba import jit
import GA
import Constants
import numpy as np
import datetime as dt
import time
import pandas as pd
import bt
import json

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

start_data = dt.datetime(2017,10,1)
start = dt.datetime(2018,1,1)
start_ts = dl.convertTimeToTimestamp(start)
end = dt.datetime(2019,1,1)

data_split = 0.5

df = dl.get(Constants.GBPUSD, Constants.DAILY, start=start_data, end=end)
# df.values[:,:4] = df[['bid_open', 'bid_high', 'bid_low', 'bid_close']].values
# Visualize data
print('\nData:\n%s'%df.head(5))
print(df.shape)
print()

'''
Feature Engineering
'''

# Convert price to pips
@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

def normalize(x, mean, std):
	return (x - mean) / std

def restrictTimePeriod(data):
	X = []
	for i in range(data.shape[0]):
		ts = data.index.values[i]
		time = dl.convertTimestampToTime(ts)
		london_time = dl.convertTimezone(time, 'Europe/London')
		if 8 <= time.hour < 20:
			X.append(data.values[i])
	return np.array(X, dtype=np.float32)

@jit
def isLongSpike(data, threshold):
	s_idx = int((data.shape[0]-1)/2)
	s_high = convertToPips(data[s_idx, 1]) - threshold

	for j in range(data.shape[0]):
		if j == s_idx:
			continue
		c_high = convertToPips(data[j, 1])
		if c_high > s_high:
			return False
	return True

@jit
def isShortSpike(data, threshold):
	s_idx = int((data.shape[0]-1)/2)
	s_low = convertToPips(data[s_idx, 2]) + threshold

	for j in range(data.shape[0]):
		if j == s_idx:
			continue
		c_low = convertToPips(data[j, 2])
		if c_low < s_low:
			return False
	return True

# @jit
def getTrainData(data, timestamps):
	spike_threshold = 1.0
	lookback = 3
	c_data = np.array([0,0], dtype=np.float32) # LONG DIST, SHORT DIST

	# Spike LONG, Swing LONG, Spike SHORT, Swing SHORT, Current High, Current Low
	ad_data = [0,0,0,0, max(data[:lookback,1]), min(data[:lookback,2])] 
	X = []

	for i in range(lookback, data.shape[0]):
		c_data = np.copy(c_data)

		ad_data[4] = data[i,1] if data[i,1] > ad_data[4] else ad_data[4]
		ad_data[5] = data[i,2] if data[i,2] < ad_data[5] else ad_data[5]

		if c_data[0]:
			c_data[0] = convertToPips((data[i,3] - ad_data[0]) * (ad_data[1] / ad_data[0]))

		if c_data[1]:
			c_data[1] = convertToPips((ad_data[2] - data[i,3]) * (ad_data[2] / ad_data[3]))

		# Get Current Spike Info
		if isLongSpike(
			data[i+1-lookback:i+1],
			spike_threshold
		):
			s_idx = int((lookback-1)/2)
			ad_data[0] = data[i-s_idx, 1]
			ad_data[1] = ad_data[5]
			ad_data[5] = min(data[i+1-lookback:i+1, 2])
 
			c_data[0] = convertToPips((data[i,3] - ad_data[0]) * (ad_data[1] / ad_data[0]))

		if isShortSpike(
			data[i+1-lookback:i+1],
			spike_threshold
		):
			s_idx = int((lookback-1)/2)
			ad_data[2] = data[i-s_idx, 2]
			ad_data[3] = ad_data[4] 
			ad_data[4] = max(data[i+1-lookback:i+1, 1])

			c_data[1] = convertToPips((ad_data[2] - data[i,3]) * (ad_data[2] / ad_data[3]))

		# time = dl.convertTimestampToTime(timestamps[i])
		# london_time = dl.convertTimezone(time, 'Europe/London')
		# if 8 <= time.hour < 20:
		X.append(c_data)
		# else:
		# 	X.append(np.array([-999,-999], dtype=np.float32))

	return X

timer = timeit()
train_data = np.array(getTrainData(np.round(df.values[:,4:], decimals=5), df.index.values), dtype=np.float32)
timer.end()
train_data_off = train_data.shape[0] - df.loc[df.index >= start_ts].values.shape[0]
train_data = train_data[train_data_off:]

print('Train Data:\n%s\n' % train_data[:5])

train_size = int(train_data.shape[0] * data_split)
period_off = 3

X_train = train_data[:train_size]
y_train = df.values[train_data_off+period_off:train_data_off+period_off+train_size].astype(np.float32)
X_val = train_data[train_size:]
y_val = df.values[train_data_off+period_off+train_size:].astype(np.float32)

mean = np.mean(X_train)
std = np.std(X_train)
X_train_norm = normalize(X_train, mean, std)
X_val_norm = normalize(X_val, mean, std)

print('Train Data: {} {}'.format(X_train.shape, y_train.shape))
print('Val Data: {} {}'.format(X_val.shape, y_val.shape))

'''
Create Genetic Model
'''

@jit(forceobj=True)
def model_run(inpt, W1, W2, W3, b1, b2, b3):
	x = np.matmul(inpt, W1) + b1
	x = bt.relu(x)		
	x = np.matmul(x, W2) + b2
	x = bt.relu(x)		
	x = np.matmul(x, W3) + b3

	x[:,:2] = bt.sigmoid(x[:,:2])

	t_x = np.copy(x[:,2])
	x[:,2] = (t_x - t_x.min()) / (t_x.max() - t_x.min())
	x[:,2] = np.sum(x[:,2])/x[:,2].size
	x[:,2] *= (250-30) + 30
	
	t_x = np.copy(x[:,3])
	x[:,3] = (t_x - t_x.min()) / (t_x.max() - t_x.min())
	x[:,3] = np.sum(x[:,3])/x[:,3].size
	x[:,3] *= (250-30) + 30
	return x

class BasicDenseModel(object):
	def __init__(self, in_size, layers):
		self._layers = layers
		self.W = []
		self.b = []
		
		self.createWb(in_size)

	def createWb(self, in_size):
		# Input layer
		self.W.append(
			np.random.normal(size=(in_size, self._layers[0]))
		)
		self.b.append(
			np.random.normal(size=(self._layers[0]))
		)

		# Hidden Layers
		for i in range(1, len(self._layers)):
			self.W.append(
				np.random.normal(size=(self._layers[i-1], self._layers[i]))
			)
			self.b.append(
				np.random.normal(size=(self._layers[i]))
			)

	def __call__(self, inpt):
		return model_run(inpt, *self.W, *self.b)

class GeneticPlanModel(GA.GeneticAlgorithmModel):
	def __init__(self, threshold=0.5):
		super().__init__()
		self.threshold = threshold

	def __call__(self, X, y, training=False):
		super().__call__(X, y)
		if training:
			results = bt.start(
				GeneticPlanModel.run, y.astype(np.float32), self.threshold,
				self._model(X)
			)
			results = [
				results[0], # Return
				results[4], # Drawdown
				results[1],	# Gain
				results[2],	# Loss
			]
			self.train_results = results
		else:
			results = bt.start(
				GeneticPlanModel.run, y.astype(np.float32), self.threshold,
				self._model(X)
			)
			results = [
				results[0], # Return
				results[4], # Drawdown
				results[1],	# Gain
				results[2],	# Loss
			]
			self.val_results = results
		return self.getPerformance(*results)

	def getPerformance(self, ret, dd, gain, loss):
		if loss == 0:
			gpr = 10.0
		else:
			gpr = (gain / loss) + 1
		return (ret) - pow(max(dd-2, 0), 2) - pow(max(gpr-2,0), 2)

	def generateModel(self, model_info):
		return BasicDenseModel(2, [32, 32, 4])

	def newModel(self):
		return GeneticPlanModel(self.threshold)

	def getWeights(self):
		return (
			[np.copy(i) for i in self._model.W]
			+ [np.copy(i) for i in self._model.b]
		)

	def setWeights(self, weights):
		self._model.W = [np.copy(i) for i in weights[:len(self._model.W)]]
		self._model.b = [np.copy(i) for i in weights[len(self._model.W):]]

	def __str__(self):
		return  ' (Train) Perf: {:.2f}\t% Ret: {:.2f}%\t% DD: {:.2f}%\tGPR: {:.2f}\n' \
				'   (Val) Perf: {:.2f}\t% Ret: {:.2f}%\t% DD: {:.2f}%\tGPR: {:.2f}\n'.format(
			self.getPerformance(*self.train_results),
			self.train_results[0], self.train_results[1],
			(self.train_results[2] / self.train_results[3]) if self.train_results[3] != 0 else 0,
			self.getPerformance(*self.val_results),
			self.val_results[0], self.val_results[1],
			(self.val_results[2] / self.val_results[3]) if self.val_results[3] != 0 else 0
		)

	@jit
	def run(i, positions, ohlc, result, data, threshold, out):
		c_dir = bt.get_direction(positions, 0)
		sl = 100.0
		tp_increment = 61.0
		# sl = min(max(out[i][2], 30), 250)
		# tp_increment = min(max(out[i][3], 30), 250)
		# print(sl)
		# print(tp_increment)

		if c_dir == bt.BUY:
			if out[i][0] > threshold:
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, bt.SELL, sl, 0, sl)

		elif c_dir == bt.SELL:
			if out[i][1] > threshold:
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, bt.BUY, sl, 0, sl)

		else:
			if out[i][0] > out[i][1]:
				if out[i][0] > threshold:
					positions = bt.create_position(positions, ohlc[i], bt.SELL, sl, 0, sl)

			else:
				if out[i][1] > threshold:
					positions = bt.create_position(positions, ohlc[i], bt.BUY, sl, 0, sl)

		if bt.get_num_positions(positions) > 0:
			pos_sl = bt.get_sl(positions, 0)
			profit = bt.get_profit(positions, 0, ohlc[i])
			profit_multi = np.floor(profit/tp_increment)
			new_sl = -tp_increment * (profit_multi-1)
			if pos_sl > new_sl:
				if -new_sl >= tp_increment*2:
					positions = bt.modify_sl(positions, 0, new_sl)
				elif -new_sl >= tp_increment*1.5:
					positions = bt.modify_sl(positions, 0, 0)

		return positions, result, data
		
'''
Run Saved Model
'''

gpm = GeneticPlanModel(threshold=0.5)
with open('./saved/v1.7.1_d_18_50g/0.json', 'r') as f:
	info = json.load(f)
	weights = [np.array(i, dtype=np.float32) for i in info['weights']]

gpm.setModel(gpm.generateModel(None))
gpm.setWeights(weights)

print(gpm(X_train_norm, y_train, training=True))
print(gpm(X_val_norm, y_val, training=False))
print(gpm)
