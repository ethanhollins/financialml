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

df = dl.get(Constants.GBPUSD, Constants.FOUR_HOURS, start=dt.datetime(2017,1,1))
df = df[['bid_open', 'bid_high', 'bid_low', 'bid_close']]

# Visualize data
print('\nH4:\n%s'%df.head(5))
print(df.shape)
print()

'''
Feature Engineering
'''

# Get donchian data
def getDonchUpDown(high, low, period):
	X = []
	last_high = 0
	last_low = 0
	for i in range(period, high.shape[0]):
		c_high = 0.
		c_low = 0.

		for j in range(i-period, i):
			if c_high == 0 or high[j] > c_high:
				c_high = high[j]
			if c_low == 0 or low[j] < c_low:
				c_low = low[j]

		if last_high != 0 and last_low != 0:
			x = []
			if c_high > last_high:
				x.append(1)
			elif c_high == last_high:
				x.append(0)
			elif c_high < last_high:
				x.append(-1)

			if c_low > last_low:
				x.append(1)
			elif c_low == last_low:
				x.append(0)
			elif c_low < last_low:
				x.append(-1)

			X.append(x)
		last_high = c_high
		last_low = c_low

	return np.array(X, dtype=np.float32)

period = 4
timer = timeit()
train_data = getDonchUpDown(
	df.values[:,1],
	df.values[:,2],
	period
)
timer.end()

print('Donch Data:\n%s'%train_data[:5])
# print('OHLC Data:\n%s'%df.values[period+1:period+1+5])
print(train_data.shape)

train_size = int(df.shape[0] * 0.7)
period_off = period+1

X_train = train_data[:train_size]
y_train = df.values[period_off:train_size+period_off].astype(np.float32)
X_val = train_data[train_size:]
y_val = df.values[train_size+period_off:].astype(np.float32)

print('Train Data: {} {}'.format(X_train.shape, y_train.shape))
print('Val Data: {} {}'.format(X_val.shape, y_val.shape))

'''
Create Genetic Model
'''

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
		x = np.matmul(inpt, self.W[0]) + self.b[0]
		for i in range(1, len(self.W)):
			x = bt.relu(x)		
			x = np.matmul(x, self.W[i]) + self.b[i]
		return bt.sigmoid(x)

# Convert price to pips
@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

class GeneticPlanModel(GA.GeneticAlgorithmModel):
	def __init__(self, sl, tp, threshold=0.5):
		super().__init__()
		self.sl = sl
		self.tp = tp
		self.threshold = threshold

	def __call__(self, X, y, training=False):
		super().__call__(X, y)
		results = bt.start(GeneticPlanModel.run, y.astype(np.float32), self._model(X), self.threshold, self.sl, self.tp)
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
		return (ret * gpr) - pow(dd, 2)

	def generateModel(self, model_info):
		return BasicDenseModel(2, [16, 16, 2])

	def setModel(self, model):
		self._model = model

	def newModel(self):
		return GeneticPlanModel(self.sl, self.tp, self.threshold)

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

	# @jit
	def run(i, positions, ohlc, result, data, out, threshold, sl, tp):
		# Long Entry
		if out[i][1] > (1 - threshold):
			c_dir = bt.get_direction(positions, 0)
			if not c_dir:
				positions = bt.create_position(positions, ohlc[i], bt.BUY, sl, tp, sl)
			elif c_dir != bt.BUY:
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, bt.BUY, sl, 0, sl)

		# Short Entry
		if out[i][0] < threshold:
			c_dir = bt.get_direction(positions, 0)
			if not c_dir:
				positions = bt.create_position(positions, ohlc[i], bt.SELL, sl, tp, sl)
			elif c_dir != bt.SELL:
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, bt.SELL, sl, 0, sl)

		return positions, result, data
		

# Test Dense Model
bdm = BasicDenseModel(2, [16, 16, 2])

print("Dense Model weights and biases:")
print([i.shape for i in bdm.W])
print([i.shape for i in bdm.b])

print("Test Dense model:")

for i in range(3):
	bdm = BasicDenseModel(2, [32, 32, 2])
	print(bdm(np.array([[-1,0],[1,1],[0,0]])))

# gpm = GeneticPlanModel(130., 1000.)

# print("\nTest Genetic Plan Model:")
# print(gpm(X_train, y_train))
# gpm = GeneticPlanModel(130., 1000.)
# print(gpm(X_val, y_val))

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

# Saved Seeds: 20
# ga.setSeed(1)
# ga.save(0,4999, 'donch_v1.2.0')

# def generate_models(num_models):
# 	models = []
# 	for i in range(num_models):
# 		models.append(GeneticPlanModel(130., 0., threshold=0.5))
# 	return models

# num_models = 5000
# ga.fit(
# 	models=generate_models(num_models),
# 	train_data=(X_train, y_train),
# 	val_data=(X_val, y_val),
# 	generations=2
# )

'''
Run Saved Model
'''

timestamps = df.index.values[period_off+y_train.shape[0]:]
print(dl.convertTimestampToTime(timestamps[0]))
gpm = GeneticPlanModel(130., 1000.)
with open('./saved/donch_v1.2.0/0.json', 'r') as f:
	info = json.load(f)
	weights = [np.array(i, dtype=np.float32) for i in info['weights']]

gpm.setModel(gpm.generateModel(None))
gpm.setWeights(weights)
print(gpm(X_train, y_train, training=True))
print(gpm(X_val, y_val))
print(gpm)
	









