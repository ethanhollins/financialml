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

period = 4
timer = timeit()
train_data = getDonchUpDown(
	df.values[:,0],
	df.values[:,1],
	period
)
timer.end()

print('Donch Data:\n%s'%train_data[-5:])
print(train_data.shape)

train_size = int(df.shape[0] * 0.7)
period_off = period+1

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
		x = np.matmul(inpt, self.W[0])
		x = tf.nn.relu(x)
		for i in range(1, len(self.W)):
			x = tf.nn.relu(x)		
			x = np.matmul(x, self.W[i]) + self.b[i]
		return tf.nn.sigmoid(x).numpy()

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
		return GeneticPlanModel.run(self._model(X), y, self.threshold, self.sl, self.tp)

	def generateModel(self, mean, std):
		return BasicDenseModel(2, [16, 16, 2])

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

	@jit
	def run(out, y, threshold, sl, tp):
		pos_open = 0
		pos_dir = 0
		result = 0.0

		for i in range(out.shape[0]):

			if pos_open != 0:
				if pos_dir == 1:
					if convertToPips(y[i][1] - pos_open) <= -sl:
						result = result - sl
						pos_open = 0
						
					# elif convertToPips(y[i][0] - pos_open) >= tp:
					# 	result = result + tp
					# 	pos_open = 0
				else:
					if convertToPips(pos_open - y[i][0]) <= -sl:
						result = result - sl
						pos_open = 0

					# elif convertToPips(pos_open - y[i][1]) >= tp:
					# 	result = result + tp
					# 	pos_open = 0

			if out[i][1] > threshold:
				if pos_open == 0 or pos_dir != 1:
					if pos_open != 0:
						result = result + convertToPips(pos_open - y[i][2])
					pos_open = y[i][2]
					pos_dir = 1
			if out[i][0] < threshold:
				if pos_open == 0 or pos_dir != 0:
					if pos_open != 0:
						result = result + convertToPips(y[i][2] - pos_open)
					pos_open = y[i][2]
					pos_dir = 0

		return result

# Test Dense Model
bdm = BasicDenseModel(2, [16, 16, 2])

print("Dense Model weights and biases:")
print([i.shape for i in bdm.W])
print([i.shape for i in bdm.b])

print("Test Dense model:")

for i in range(3):
	bdm = BasicDenseModel(2, [32, 32, 2])
	print(bdm(np.array([[-1,0],[1,1],[0,0]])))

gpm = GeneticPlanModel(130., 1000.)

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
		models.append(GeneticPlanModel(130., None, threshold=0.4))
	return models

num_models = 5000
ga.fit(
	models=generate_models(num_models),
	train_data=(X_train, y_train),
	val_data=(X_val, y_val),
	generations=10
)













