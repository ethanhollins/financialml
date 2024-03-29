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

df = dl.get(Constants.GBPUSD, Constants.THIRTY_MINUTES, start=dt.datetime(2019,1,1), end=dt.datetime(2019,6,1))

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

# Get donchian data
def getDonchUpDown(high, low, period, lookup):
	X = []
	last_high = 0
	last_low = 0
	for i in range(period, high.shape[0]):
		c_high = 0.
		c_low = 0.
		x = []

		for j in range(i+1-period, i+1):
			if c_high == 0 or high[j] > c_high:
				c_high = high[j]
			if c_low == 0 or low[j] < c_low:
				c_low = low[j]

		if last_high and last_low:
			x.append(convertToPips(c_high - last_high))
			x.append(convertToPips(c_low - last_low))

			if len(X) > 0:
				X.append(X[-1][-(lookup-1):] + [x])
			else:
				X.append([x])

		last_high = c_high
		last_low = c_low

	return np.array(X[lookup-1:], dtype=np.float32)

period = 4
timer = timeit()
train_data = getDonchUpDown(
	df.values[:,5],
	df.values[:,6],
	period,
	5 # lookup
)
timer.end()

print('Donch Data:\n%s'%train_data[:2])
print(train_data.shape)

train_size = int(df.shape[0] * 0.3)
period_off = df.shape[0] - train_data.shape[0]

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
	for i in range(inpt.shape[1]):
		c_i = inpt[:,i,:]
		c_i = c_i.reshape(c_i.shape[0], 1, c_i.shape[1])
		x = np.matmul(c_i, W1)
		x = bt.relu(x)

		if i+1 == inpt.shape[1]:
			break

		x = np.matmul(x.reshape(x.shape[0], x.shape[2], x.shape[1]), np.ones([1,c_i.shape[2]]))
		x = bt.relu(x.reshape(x.shape[0], x.shape[2], x.shape[1]))
		W1 = x
	
	x = np.matmul(x, W2) + b2
	x = bt.relu(x)

	x = np.matmul(x, W3) + b3
	x = bt.sigmoid(x).reshape(x.shape[0], x.shape[2])

	return x

class BasicRnnModel(object):
	def __init__(self, in_shape, layers):
		self._layers = layers
		self.W = []
		self.b = []
		
		self.createWb(in_shape)

	def createWb(self, in_shape):
		# RNN Layer
		self.W.append(
			np.random.normal(size=(in_shape[2], self._layers[0]))
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
			GeneticPlanModel.run, y.astype(np.float32), self.threshold, self._model(X)
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
		return ret - pow(max(dd-3, 0), 2) - pow(max(gpr-3,0), 2)

	def generateModel(self, model_info):
		return BasicRnnModel(X_train.shape, [16, 16, 2])

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
		sl = 80.0

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

		return positions, result, data
		

'''
Create Genetic Algorithm
'''

bt.recompile_all()

crossover = GA.PreserveBestCrossover(preserve_rate=0.5)
mutation = GA.PreserveBestMutation(mutation_rate=0.05, preserve_rate=0.5)
ga = GA.GeneticAlgorithm(
	crossover, mutation,
	survival_rate=0.2
)

ga.setSeed(1)
# ga.save(16, 1552, 'v1.3.1', {'mean': float(mean), 'std': float(std)})
# ga.save(16, 2, 'v1.3.1', {'mean': float(mean), 'std': float(std)})
ga.saveBest(10, 'v1.7.0_4', {'mean': float(mean), 'std': float(std)})

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
	generations=50
)











