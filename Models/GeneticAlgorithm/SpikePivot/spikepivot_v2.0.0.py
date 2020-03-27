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

start = dt.datetime(2019,1,1)
end = dt.datetime(2020,1,1)

data_split = 0.7
stoprange = 200.0
num_years = end.year - start.year

df_h4 = dl.get(Constants.GBPUSD, Constants.FOUR_HOURS, start=start, end=end)
df_m = dl.get(Constants.GBPUSD, Constants.FIVE_MINUTES, start=start, end=end)

# Add Offset
df_h4.index = df_h4.index + dl.getPeriodOffsetSeconds(Constants.FOUR_HOURS)
df_m.index = df_m.index + dl.getPeriodOffsetSeconds(Constants.FIVE_MINUTES)

'''
Feature Engineering
'''

# Convert price to pips
@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

def normalize(x, mean, std):
	return (x - mean) / std

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
def getTrainData(data):
	spike_threshold = 1.0
	lookback = 3
	c_data = np.array([[0,0],[0,0]], dtype=np.float32) # LONG DIST, SHORT DIST

	# Spike LONG, Swing LONG, Spike SHORT, Swing SHORT, Current High, Current Low
	ad_data = [0,0,0,0, max(data[:lookback,1]), min(data[:lookback,2])] 
	X = []

	for i in range(lookback, data.shape[0]):
		c_data = np.copy(c_data)

		# Get Current Spike Info
		if isLongSpike(
			data[i+1-lookback:i+1],
			spike_threshold
		):
			s_idx = int((lookback-1)/2)
			ad_data[0] = data[i-s_idx, 1]
			ad_data[1] = ad_data[5]

			body = convertToPips(data[i-s_idx, 3] - data[i-s_idx, 0])

			if body >= 0:
				wick = convertToPips(data[i-s_idx, 1] - data[i-s_idx, 3])
			else:
				wick = convertToPips(data[i-s_idx, 1] - data[i-s_idx, 0])

			c_data[0] = [body, wick]

		if isShortSpike(
			data[i+1-lookback:i+1],
			spike_threshold
		):
			s_idx = int((lookback-1)/2)
			ad_data[2] = data[i-s_idx, 2]
			ad_data[3] = ad_data[4]

			body = convertToPips(data[i-s_idx, 0] - data[i-s_idx, 3])

			if body >= 0:
				wick = convertToPips(data[i-s_idx, 3] - data[i-s_idx, 2])
			else:
				wick = convertToPips(data[i-s_idx, 0] - data[i-s_idx, 2])

			c_data[1] = [body, wick]

		ad_data[4] = data[i,1] if data[i,1] > ad_data[4] else ad_data[4]
		ad_data[5] = data[i,2] if data[i,2] < ad_data[5] else ad_data[5]

		X.append(c_data)

	return X

timer = timeit()
train_data = np.array(getTrainData(np.round(df_h4.values[:,4:], decimals=5)), dtype=np.float32)
timer.end()
print('Train Data:\n%s\n' % train_data[:5])
print(train_data.shape)

train_size = int(train_data.shape[0] * data_split)

period_off = 3
train_start_ts = df_h4.index[period_off]
val_start_ts = df_h4.index[period_off+train_size]

X_train = train_data[:train_size-1]
X_val = train_data[train_size:-1]

mean = np.mean(X_train)
std = np.std(X_train)
X_train_norm = normalize(X_train, mean, std)
X_val_norm = normalize(X_val, mean, std)

'''
Charts processing
'''

@jit
def processChart(i, charts, timestamps, chart, chart_ts):
	for j in range(chart_ts.shape[0]):
		ts = chart_ts[j]
		idx = (np.abs(timestamps - ts)).argmin()
		charts[i][idx] = chart[j]
	return charts

# Train y data

y_train_h4 = df_h4[period_off:period_off+train_size].values
y_train_h4_ts = df_h4[period_off:period_off+train_size].index.values

y_train_m = df_m.loc[(df_m.index >= train_start_ts) & (df_m.index < val_start_ts)].values
y_train_ts_m = df_m.loc[(df_m.index >= train_start_ts) & (df_m.index < val_start_ts)].index.values

y_train_ts = np.sort(np.unique(np.concatenate(
	(y_train_h4_ts, y_train_ts_m)
)))

unprocessed_charts = [y_train_h4, y_train_m]
unprocessed_ts = [y_train_h4_ts, y_train_ts_m]
y_train = np.zeros((len(unprocessed_charts), y_train_ts.shape[0], y_train_m.shape[1]))

for i in range(len(unprocessed_charts)):
	chart = unprocessed_charts[i]
	ts = unprocessed_ts[i]
	y_train = processChart(i, y_train, y_train_ts, chart, ts)

print('\nTrain Chart Data:\n%s'%y_train[:,:5])

# Check 4 Hour candles are correctly ordered
# idx = np.unique(np.nonzero(y_train[0])[0])
# print(y_train[0, idx])

# Validation y data

y_val_h4 = df_h4[period_off+train_size:].values
y_val_h4_ts = df_h4[period_off+train_size:].index.values

y_val_m = df_m.loc[df_m.index >= val_start_ts].values
y_val_ts_m = df_m.loc[df_m.index >= val_start_ts].index.values

y_val_ts = np.sort(np.unique(np.concatenate(
	(y_val_h4_ts, y_val_ts_m)
)))

unprocessed_charts = [y_val_h4, y_val_m]
unprocessed_ts = [y_val_h4_ts, y_val_ts_m]
y_val = np.zeros((len(unprocessed_charts), y_val_ts.shape[0], y_val_m.shape[1]))

for i in range(len(unprocessed_charts)):
	chart = unprocessed_charts[i]
	ts = unprocessed_ts[i]
	y_val = processChart(i, y_val, y_val_ts, chart, ts)

# Visualize data
print('\nVal Chart Data:\n%s'%y_val[:,:5])

num_train_h4 = np.unique(np.nonzero(y_train[0])[0]).shape
num_val_h4 = np.unique(np.nonzero(y_val[0])[0]).shape

print('\nTrain Data: {} {} H4: {}'.format(X_train.shape, y_train.shape, num_train_h4))
print('Val Data: {} {} H4: {}'.format(X_val.shape, y_val.shape, num_val_h4))

raise SystemExit()

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
	def run(i, j, positions, charts, result, data, stats, threshold, out):
		

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
ga.saveBest(10, 'v2.0.0', {'mean': float(mean), 'std': float(std)})

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


