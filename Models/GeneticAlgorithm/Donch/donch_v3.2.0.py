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
df = df[['bid_high', 'bid_low', 'bid_close']]

# Visualize data
print('\nH4:\n%s'%df.head(5))
print(df.shape)
print()

'''
Feature Engineering
'''

# Convert price to pips
@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

def normalize(x):
	return (x - np.mean(x)) / np.std(x)

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

	return np.array(X)

donch_period = 4
timer = timeit()
train_data = getDonchUpDown(
	df.values[:,0],
	df.values[:,1],
	donch_period
)
timer.end()

print('Donch Data:\n%s'%train_data[-5:])
print(train_data.shape)

train_size = int(df.shape[0] * 0.7)
period_off = donch_period+1

X_train = train_data[:train_size]
y_train = df.values[period_off:train_size+period_off]
X_val = train_data[train_size:]
y_val = df.values[train_size+period_off:]

print('Train Data: {} {}'.format(X_train.shape, y_train.shape))
print('Val Data: {} {}'.format(X_val.shape, y_val.shape))

'''
Create Genetic Model
'''

class BasicDenseModel(object):
	def __init__(self, in_size1, layers1, in_size2, layers2):
		self._layers1 = layers1
		self.W1 = []
		self.b1 = []

		self._layers2 = layers2
		self.W2 = []
		self.b2 = []
		
		self.createWb1(in_size1)
		self.createWb2(in_size2)

	def createWb1(self, in_size):
		# Input layer
		self.W1.append(
			np.random.normal(size=(in_size, self._layers1[0]))
		)
		self.b1.append(
			np.random.normal(size=(self._layers1[0]))
		)

		# Hidden Layers
		for i in range(1, len(self._layers1)):
			self.W1.append(
				np.random.normal(size=(self._layers1[i-1], self._layers1[i]))
			)
			self.b1.append(
				np.random.normal(size=(self._layers1[i]))
			)

	def createWb2(self, in_size):
		# Input layer
		self.W2.append(
			np.random.normal(size=(in_size, self._layers2[0]))
		)
		self.b2.append(
			np.random.normal(size=(self._layers2[0]))
		)

		# Hidden Layers
		for i in range(1, len(self._layers2)):
			self.W2.append(
				np.random.normal(size=(self._layers2[i-1], self._layers2[i]))
			)
			self.b2.append(
				np.random.normal(size=(self._layers2[i]))
			)

	def __call__(self, inpt1):
		x = np.matmul(inpt1, self.W1[0]) + self.b1[0]
		for i in range(1, len(self.W1)):
			x = tf.nn.relu(x)		
			x = np.matmul(x, self.W1[i]) + self.b1[i]

		y = (self.W2[0] * ((25+250)/2)) + self.b2[0]
		for i in range(1, len(self.W2)):
			y = tf.nn.relu(y)		
			y = np.matmul(y, self.W2[i]) + self.b2[i]

		return [tf.nn.sigmoid(x).numpy(), np.clip(np.sum(y), 25, 500)]

class GeneticPlanModel(GA.GeneticAlgorithmModel):
	def __init__(self, threshold=0.5):
		super().__init__()
		self.threshold = threshold

	def __call__(self, X, y, training=False):
		super().__call__(X, y)
		out = self._model(X)
		results = GeneticPlanModel.run(out[0], out[1], y, self.threshold, 130)
		if training:
			self.train_results = results
		else:
			self.val_results = results
		return self.getPerformance(*results)

	def getPerformance(self, result, dd, gain, loss):
		if loss == 0:
			gpr = 0
		else:
			gpr = (gain / loss) + 1
		return (result * gpr) - pow(dd, 2)

	def generateModel(self, model_info):
		return BasicDenseModel(2, [16, 16, 2], 1, [32, 1])

	def newModel(self):
		return GeneticPlanModel(self.threshold)

	def getWeights(self):
		return (
			[np.copy(i) for i in self._model.W1]
			+ [np.copy(i) for i in self._model.b1]
			+ [np.copy(i) for i in self._model.W2]
			+ [np.copy(i) for i in self._model.b2]
		)

	def setWeights(self, weights):
		off = 0
		off += len(self._model.W1)
		self._model.W1 = [np.copy(i) for i in weights[:off]]
		self._model.b1 = [np.copy(i) for i in weights[off:off+len(self._model.b1)]]
		off += len(self._model.b1)
		self._model.W2 = [np.copy(i) for i in weights[off:off+len(self._model.W2)]]
		off += len(self._model.W2)
		self._model.b2 = [np.copy(i) for i in weights[off:]]

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
	def run(out1, tp, y, threshold, sl):
		pos_open = 0
		pos_dir = 0
		result = 0.0

		max_ret = 0.0
		dd = 0.0
		gain = 0.0
		loss = 0.0

		for i in range(out1.shape[0]):

			if pos_open != 0:
				if pos_dir == 1:
					if convertToPips(y[i][1] - pos_open) <= -sl:
						loss += 1.0
						result -= 1.0
						pos_open = 0

					elif convertToPips(y[i][0] - pos_open) >= tp:
						gain += (tp/sl)
						result += (tp/sl)
						pos_open = 0
				else:
					if convertToPips(pos_open - y[i][0]) <= -sl:
						loss += 1.0
						result -= 1.0
						pos_open = 0

					elif convertToPips(pos_open - y[i][1]) >= tp:
						gain += (tp/sl)
						result += (tp/sl)
						pos_open = 0

			if out1[i][1] > threshold:
				if pos_open == 0 or pos_dir != 1:
					if pos_open != 0:
						ret = convertToPips(pos_open - y[i][2])/sl
						if ret >= 0:
							gain += ret
						else:
							loss += abs(ret)
						result += ret
					pos_open = y[i][2]
					pos_dir = 1
			if out1[i][0] < threshold:
				if pos_open == 0 or pos_dir != 0:
					if pos_open != 0:
						ret = convertToPips(y[i][2] - pos_open)/sl
						if ret >= 0:
							gain += ret
						else:
							loss += abs(ret)
						result += ret
					pos_open = y[i][2]
					pos_dir = 0

			if result > max_ret:
				max_ret = result
			elif (max_ret - result) > dd:
				dd = (max_ret - result)

		return [result, dd, gain, loss]

# Test Dense Model
bdm = BasicDenseModel(2, [16, 16, 2], 2, [16, 16, 1])

print("Dense Model weights and biases:")
print([i.shape for i in bdm.W1])
print([i.shape for i in bdm.b1])
print([i.shape for i in bdm.W2])
print([i.shape for i in bdm.b2])

print("Test Dense model:")

gpm = GeneticPlanModel()

print("\nTest Genetic Plan Model:")
print(gpm(X_train, y_train))

'''
Create Genetic Algorithm
'''

crossover = GA.PreserveBestCrossover(preserve_rate=0.5)
mutation = GA.PreserveBestMutation(mutation_rate=0.02, preserve_rate=0.5)
ga = GA.GeneticAlgorithm(
	crossover, mutation,
	survival_rate=0.2
)

def generate_models(num_models):
	models = []
	for i in range(num_models):
		models.append(GeneticPlanModel(threshold=0.4))
	return models

num_models = 5000
ga.fit(
	models=generate_models(num_models),
	train_data=(X_train, y_train),
	val_data=(X_val, y_val),
	generations=10
)













