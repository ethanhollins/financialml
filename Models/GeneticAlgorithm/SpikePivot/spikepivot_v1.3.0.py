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
import tensorflow as tf
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

df = dl.get(Constants.GBPUSD, Constants.THIRTY_MINUTES, start=dt.datetime(2018,1,1))
print(df.values[:,:4])
df.values[:,:4] = df[['bid_open', 'bid_high', 'bid_low', 'bid_close']].values
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

@jit
def getTrainData(data):
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

		X.append(c_data)

	return X

timer = timeit()
train_data = np.array(getTrainData(np.round(df.values[:,4:], decimals=5)), dtype=np.float32)
timer.end()

print('Train Data:\n%s\n' % train_data[:5])

train_size = int(df.shape[0] * 0.7)
period_off = 3

X_train = train_data[:train_size]
y_train = df.values[period_off:train_size+period_off].astype(np.float32)
X_val = train_data[train_size:]
y_val = df.values[train_size+period_off:].astype(np.float32)

mean = np.mean(X_train)
std = np.std(X_train)
X_train = normalize(X_train, mean, std)
X_val = normalize(X_val, mean, std)

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
		]
		if training:
			self.train_results = results
		else:
			self.val_results = results
		return self.getPerformance(*results)

	def getPerformance(self, ret, dd, gain, loss):
		if loss == 0:
			gpr = 0.0
		else:
			gpr = (gain / loss) + 1
		return (ret) - pow(dd, 2)

	def generateModel(self, model_info):
		return BasicDenseModel(2, [128, 128, 2])

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

	# TODO:
	# 	Run model from inside run function
	#	Switch inputs based on current position
	@jit#(forceobj=True)
	def run(i, positions, ohlc, result, data, threshold, out):
		c_dir = bt.get_direction(positions, 0)
		sl = 200.0
		# print('{} - {}'.format(timestamps[i], out[i]))

		if c_dir == bt.BUY:
			if out[i][0] > threshold:
				# print('SELL (S&R) {} - {} | {:.2f}'.format(timestamps[i], ohlc[i], result * 55))
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, bt.SELL, sl, 0, sl)

		elif c_dir == bt.SELL:
			if out[i][1] > threshold:
				# print('BUY (S&R) {} - {} | {:.2f}'.format(timestamps[i], ohlc[i], result * 55))
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, bt.BUY, sl, 0, sl)

		else:
			if out[i][0] > out[i][1]:
				if out[i][0] > threshold:
					# print('SELL (REG) {} - {} | {:.2f}'.format(timestamps[i], ohlc[i], result * 55))
					positions = bt.create_position(positions, ohlc[i], bt.SELL, sl, 0, sl)

			else:
				if out[i][1] > threshold:
					# print('BUY (REG) {} - {} | {:.2f}'.format(timestamps[i], ohlc[i], result * 55))
					positions = bt.create_position(positions, ohlc[i], bt.BUY, sl, 0, sl)

		return positions, result, data
		
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
# ga.save(0, 3899, 'v1.3.0', {'mean': float(mean), 'std': float(std)})

def generate_models(num_models):
	models = []
	for i in range(num_models):
		models.append(GeneticPlanModel(threshold=0.5))
	return models

num_models = 5000
ga.fit(
	models=generate_models(num_models),
	train_data=(X_train, y_train),
	val_data=(X_val, y_val),
	generations=20
)

'''
Run Saved Model
'''
# gpm = GeneticPlanModel(threshold=0.5)
# with open('./saved/v1.3.0/0.json', 'r') as f:
# 	info = json.load(f)
# 	weights = [np.array(i, dtype=np.float32) for i in info['weights']]

# timestamps = df.index.values[period_off:]
# print(dl.convertTimestampToTime(timestamps[0]))
# gpm.setModel(gpm.generateModel(None))
# gpm.setWeights(weights)
# print(gpm(X_train, y_train, training=True))
# print(gpm(X_val, y_val))
# print(gpm)
	



