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

data_split = 0.7
stoprange = 200.0
num_years = end.year - start.year

df_h4 = dl.get(Constants.GBPUSD, Constants.FOUR_HOURS, start=start_data, end=end)
df_m = dl.get(Constants.GBPUSD, Constants.ONE_MINUTE, start=start_data, end=end)
df_m = df_m.loc[df_m.index >= df_h4.index[1]]
df_h4 = df_h4.loc[df_h4.index[1:]]

# Visualize data
print('\nData (H4):\n%s'%df_h4.head(5))
print(df_h4.shape)
print('\nData (M1):\n%s'%df_m.head(5))
print(df_m.shape)
print()

print(dl.convertTimestampToTime(df_h4.index.values[0]))
print(dl.convertTimestampToTime(df_m.index.values[0]))
raise SystemExit()
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
	x = bt.sigmoid(x)
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
		results = bt.start(
			GeneticPlanModel.run, y.astype(np.float32), self.threshold,
			self._model(X)
		)
		results = [
			results[0], # Return
			results[4], # Drawdown
			results[1],	# Gain
			results[2],	# Loss
			results[5], # Wins
			results[6], # Losses
		]
		if training:
			self.train_results = results
		else:
			self.val_results = results

		return self.getPerformance(*results, training=training)

	def getPerformance(self, ret, dd, gain, loss, wins, losses, training=False):
		if loss == 0:
			gpr = 0.0
		else:
			gpr = (gain / loss) + 1

		dd_mod = pow(max(dd-2, 0), 2)
		gpr_mod = pow(max(gpr-3,0), 2)

		if training:
			min_trades = (num_years * 60) * data_split
		else:
			min_trades = (num_years * 60) * (1 - data_split)

		num_trades = wins + losses
		trades_mod = pow(min_trades - num_trades, 2) if num_trades < min_trades else 0

		return ret - dd_mod - gpr_mod - trades_mod

	def generateModel(self, model_info):
		return BasicDenseModel(2, [16, 16, 2])

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
		return  ' (Train) Perf: {:.2f}\t% Ret: {:.2f}%\t% DD: {:.2f}%\tGPR: {:.2f}\tTrades: {}\n' \
				'   (Val) Perf: {:.2f}\t% Ret: {:.2f}%\t% DD: {:.2f}%\tGPR: {:.2f}\tTrades: {}\n'.format(
			self.getPerformance(*self.train_results),
			self.train_results[0], self.train_results[1],
			(self.train_results[2] / self.train_results[3]) if self.train_results[3] != 0 else 0,
			(self.train_results[4] + self.train_results[5]),
			self.getPerformance(*self.val_results),
			self.val_results[0], self.val_results[1],
			(self.val_results[2] / self.val_results[3]) if self.val_results[3] != 0 else 0,
			(self.val_results[4] + self.val_results[5])
		)

	@jit
	def run(i, positions, ohlc, result, data, stats, threshold, out):
		c_dir = bt.get_direction(positions, 0)
		sl = stoprange

		if c_dir == bt.BUY:
			if out[i][0] > threshold:
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, stats, bt.SELL, sl, 0, sl)

		elif c_dir == bt.SELL:
			if out[i][1] > threshold:
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, stats, bt.BUY, sl, 0, sl)

		else:
			if out[i][0] > out[i][1]:
				if out[i][0] > threshold:
					positions = bt.create_position(positions, ohlc[i], bt.SELL, sl, 0, sl)

			else:
				if out[i][1] > threshold:
					positions = bt.create_position(positions, ohlc[i], bt.BUY, sl, 0, sl)

		if bt.get_num_positions(positions) > 0:
			pos_sl = bt.get_sl(positions, 0)
			risk = bt.get_risk(positions, 0)
			profit = bt.get_profit(positions, 0, ohlc[i])
			profit_multi = np.floor(profit/risk)
			new_sl = -risk * (profit_multi-1)
			if pos_sl > new_sl:
				if -new_sl >= risk*2:
					positions = bt.modify_sl(positions, 0, new_sl)
				elif -new_sl >= risk*1.5:
					positions = bt.modify_sl(positions, 0, 0)

		return positions, result, data, stats
		
'''
Create Genetic Algorithm
'''

bt.recompile_all()

crossover = GA.PreserveBestCrossover(preserve_rate=0.5)
mutation = GA.PreserveBestMutation(mutation_rate=0.02, preserve_rate=0.5)
ga = GA.GeneticAlgorithm(
	crossover, mutation,
	survival_rate=0.2
)

ga.setSeed(1)
ga.saveBest(10, 'v1.7.0_d_2017_100g_70p_16x16_60min', {'mean': float(mean), 'std': float(std)})

def generate_models(num_models):
	models = []
	for i in range(num_models):
		models.append(GeneticPlanModel(threshold=0.5))
	return models

num_models = 5000
ga.fit(
	models=generate_models(num_models),
	train_data=(X_train_norm, y_train),
	val_data=(X_val_norm, y_val),
	generations=100
)


