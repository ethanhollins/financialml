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

df = dl.get(Constants.GBPUSD, Constants.ONE_HOUR, start=dt.datetime(2017,1,1))
df = df[['bid_open', 'bid_high', 'bid_low', 'bid_close']]

# Visualize data
print('\nH4:\n%s'%df.head(5))
print(df.shape)
print()

'''
Feature Engineering
'''
@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

def normalize(x):
	return (x - np.mean(x)) / np.std(x)

# Produce SMA train data
@jit
def getSmaDiff(data, periods, lookup):
	X = []
	for i in range(periods.max()+lookup, data.shape[0]):
		c_lookup = []
		for j in range(i-lookup, i):
			p_diff = []
			for p_i in range(len(periods)-1):
				p_x = periods[p_i]
				for p_y in periods[p_i+1:]:
					x = np.sum(data[j+1-p_x:j+1])/p_x
					y = np.sum(data[j+1-p_y:j+1])/p_y
					p_diff.append(1 if x > y else 0)
			c_lookup.append(p_diff)
		X.append(c_lookup)
	return np.array(X)

lookup = 1
periods = [1,2,5,10]
timer = timeit()
X = getSmaDiff(df.values, np.array(periods), lookup)
# X = normalize(X)
X = X.reshape(
	X.shape[0],
	X.shape[1] * X.shape[2]
)

# Visualize train data
print('X: {}'.format(X[-5:]))
timer.end()

train_size = int(0.7 * X.shape[0])
off = max(periods) + lookup

X_train = X[:train_size]
y_train = df.values[off:train_size+off]
X_val = X[train_size:]
y_val = df.values[train_size+off:]

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
	x[:,2] *= (250-50) + 50
	
	t_x = np.copy(x[:,3])
	x[:,3] = (t_x - t_x.min()) / (t_x.max() - t_x.min())
	x[:,3] = np.sum(x[:,3])/x[:,3].size
	x[:,3] *= (1300-30) + 30
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

# Convert price to pips
@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

class GeneticPlanModel(GA.GeneticAlgorithmModel):
	def __init__(self, max_pos=2, threshold=0.5):
		super().__init__()
		self.max_pos = max_pos
		self.threshold = threshold

	def __call__(self, X, y, training=False):
		super().__call__(X, y)
		results = bt.start(
			GeneticPlanModel.run, y.astype(np.float32), self._model(X), self.threshold, self.max_pos
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
		return (ret * gpr) - pow(dd, 2)

	def generateModel(self, model_info):
		return BasicDenseModel(X_train.shape[1], [64, 64, 4])

	def newModel(self):
		return GeneticPlanModel(self.max_pos, self.threshold)

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
	def run(i, positions, ohlc, result, data, out, threshold, max_pos):
		sl = min(max(out[i][2], 50), 250)
		tp = min(max(out[i][3], 30), 1300)
			# Long Entry
		if out[i][1] > (1 - threshold):
			c_dir = bt.get_direction(positions, 0)
			if not c_dir:
				positions = bt.create_position(positions, ohlc[i], bt.BUY, sl, tp)
			elif c_dir == bt.BUY:
				if bt.get_num_positions(positions) < max_pos:
					positions = bt.create_position(positions, ohlc[i], bt.BUY, sl, tp)
			elif c_dir == bt.SELL:
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, bt.BUY, sl, tp)


		# Short Entry
		if out[i][0] < threshold:
			c_dir = bt.get_direction(positions, 0)
			if not c_dir:
				positions = bt.create_position(positions, ohlc[i], bt.SELL, sl, tp)
			elif c_dir == bt.SELL:
				if bt.get_num_positions(positions) < max_pos:
					positions = bt.create_position(positions, ohlc[i], bt.SELL, sl, tp)
			elif c_dir == bt.BUY:
				positions, result = bt.stop_and_reverse(positions, ohlc[i], result, bt.SELL, sl, tp)

		return positions, result, data
		

# Test Dense Model
bdm = BasicDenseModel(2, [16, 16, 4])

print("Dense Model weights and biases:")
print([i.shape for i in bdm.W])
print([i.shape for i in bdm.b])

# print("Test Dense model:")

# for i in range(3):
# 	bdm = BasicDenseModel(2, [32, 32, 4])
# 	print(bdm(np.array([[-1,0],[1,1],[0,0]])))

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

def generate_models(num_models):
	models = []
	for i in range(num_models):
		models.append(GeneticPlanModel(1, threshold=0.5))
	return models

num_models = 5000
ga.fit(
	models=generate_models(num_models),
	train_data=(X_train, y_train),
	val_data=(X_val, y_val),
	generations=10
)













