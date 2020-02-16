"""

Performance driven deep learning model. Model uses
genetic algorithm concepts to find a solution.

Created by: Ethan Hollins

"""

from DataLoader import DataLoader
from numba import jit
from sklearn.preprocessing import MinMaxScaler
import Constants
import datetime as dt
import tensorflow as tf
import numpy as np
import math
import time

tf.keras.backend.set_floatx('float64')

class timeit(object):
	def __init__(self):
		print('\nTimer module started.')
		self.start = time.time()
	def end(self):
		print('Timer module finished: {:.2f}s'.format(time.time() - self.start))

'''
Data Preprocessing
'''

# Load GBPUSD ohlc data
dl = DataLoader()
df = dl.get(Constants.GBPUSD, Constants.ONE_HOUR, start=dt.datetime(2017,1,1))
df = df[['bid_close']]

print(df.head(5))
print(df.shape)

@jit
def convertToPips(x):
	return np.around(x * 1000, 2)

# Produce SMA train data
@jit
def getSmaDiff(data, periods, lookup):
	X = []
	for i in range(periods.max()+lookup, data.shape[0]):
		c_lookup = []
		for j in range(lookup):
			p_diff = []
			for p_i in range(len(periods)-1):
				p_x = periods[p_i]
				for p_y in periods[p_i+1:]:
					x = np.sum(data[i-j+1-p_x:i-j+1])/p_x
					y = np.sum(data[i-j+1-p_y:i-j+1])/p_y
					p_diff.append(convertToPips(x) - convertToPips(y))
			c_lookup.append(p_diff)
		X.append(c_lookup)
	return np.array(X)

# Test SMA function
# timer = timeit()
# sma_diff = getSmaDiff(df.values, [20,50,100], 5)
# print('Sample:\n{}\n'.format(sma_diff[:5]))
# print('Shape: {}'.format(sma_diff.shape))
# timer.end()

# Generate training data
def generateTrainData(data):
	return getSmaDiff(data, np.array([20,50,100]), 5), data[:,0]

def normalize(x):
	return (x - np.mean(x)) / np.std(x)

timer = timeit()
X, y = generateTrainData(df.values)
X = normalize(X)

# Visualize train data
print('X: {}'.format(X[-5:]))
print('y: {}'.format(y[-5:]))
timer.end()

# X = tf.convert_to_tensor(X, tf.float32)
# y = tf.convert_to_tensor(y, tf.float32)

'''
Build Genetic Algorithm model
'''

class SimplePipReturnEvaluator(object):

	def __init__(self, threshold=0.5):
		self.threshold = threshold

	def __call__(self, y, out):
		return SimplePipReturnEvaluator.run(y, out.numpy(), self.threshold)

	@jit
	def run(y, out, threshold):
		result = 0.0
		c_dir = -1
		last_dir = -1
		last_entry = 0.0

		for i in range(out.shape[0]):
			c_out = out[i][0]

			if c_out > (1.0 - threshold):
				c_dir = 1
			elif c_out < threshold:
				c_dir = 0
			else:
				c_dir = -1

			if c_dir != -1:
				if last_dir != -1:
					if last_dir != c_dir:
						if last_dir == 1:
							result += (y[i] - last_entry)
						else:
							result += (last_entry - y[i])
						last_dir = c_dir
						last_entry = y[i]
				else:
					last_dir = c_dir
					last_entry = y[i]

		return convertToPips(result)

class GeneticAlgorithmModel(object):

	def __init__(self, model, evaluator):
		self._model = model
		self._evaluator = evaluator

	def __call__(self, X, y, training=False):
		return self._evaluator(y, self._model(X))

	def mutate(self, weights, mutation_rate, mean, std):
		for i in range(len(weights)):
			if type(weights[i]) == np.ndarray:
				weights[i] = self.mutate(weights[i], mutation_rate, mean, std)
			else:
				if np.random.uniform() <= mutation_rate:
					weights[i] = np.random.normal(mean, std)
		return weights

	def getModel(self):
		return self._model

	def setModel(self, model):
		self._model = model

	def getWeights(self):
		return self._model.weights

	def setWeights(self, weights):
		self._model.set_weights(weights)

class GeneticAlgorithmController(object):
	# TODO: Add optimizer functionality to performance selection process

	def __init__(self, evaluator, num_models=100, survival_rate=0.2, mutation_rate=0.01):
		self._evaluator = evaluator
		self._num_models = num_models
		self._survival_rate = survival_rate
		self._mutation_rate = mutation_rate

	def add(self, model):
		self._models.append(model)

	def fit(self, train_data, val_data=None, batch_size=0, generations=20, run=True):
		assert type(batch_size) == int
		self._mean = np.mean(train_data[0])
		self._std = np.std(train_data[0])

		train_data = (tf.convert_to_tensor(train_data[0]), tf.convert_to_tensor(train_data[1]))


		if batch_size:
			X = [train_data[0][i*batch_size:i*batch_size+batch_size] for i in math.ceil(train_data[0].shape[0]/batch_size)]
			y = [train_data[0][i*batch_size:i*batch_size+batch_size] for i in math.ceil(train_data[0].shape[0]/batch_size)]
			self._train_data = (X, y)
		else:
			self._train_data = ([train_data[0]], [train_data[1]])

		if val_data:
			self._val_data = (tf.convert_to_tensor(val_data[0]), tf.convert_to_tensor(val_data[1]))
		else:
			self._val_data = None

		self.generateModels()

		if run:
			assert type(generations) == int
			self.run(generations)

	def run(self, generations=20):
		for generation in range(generations):
			performance_l = self()
			print(sorted(np.array(performance_l)[:,0], reverse=True))
			self.optimize(np.array(performance_l))

	def __call__(self):
		performance_l = []
		for mod in self._models:
			batch = []
			for i in range(len(self._train_data[0])):
				batch.append(
					mod(self._train_data[0][i], self._train_data[1][i])
				)
			performance_l.append(batch)
			print('.', end='', flush=True)
		print()
		return performance_l

	def generateModels(self):
		self._models = []
		for i in range(self._num_models):
			model = self.generateKerasModel()
			self._models.append(GeneticAlgorithmModel(model, self._evaluator))
		print('Models initialized.')

	def generateKerasModel(self):
		model = tf.keras.models.Sequential()
		# model.add(tf.keras.layers.LSTM(
		# 	16, return_sequences=True, 
		# 	kernel_initializer=tf.keras.initializers.RandomNormal(mean=np.mean(X), stddev=np.std(X))
		# ))
		model.add(tf.keras.layers.LSTM(
			16, kernel_initializer=tf.keras.initializers.RandomNormal(mean=np.mean(X), stddev=np.std(X))
		))
		model.add(tf.keras.layers.Dense(
			1, activation='sigmoid',
			kernel_initializer=tf.keras.initializers.RandomNormal(mean=np.mean(X), stddev=np.std(X))
		))
		return model

	def select(self, p):
		# for i in range(len(p)):
		# 	if p[i] == 0.0:
		# 		print(self._models[i].getWeights())

		p_scaled = (p - p.min()) / (p.max() - p.min())
		p_sorted = sorted(enumerate(p_scaled), key=lambda x: x[1], reverse=True)
		p_selected = []
		selected = []
		idx = 0
		while len(selected) < int(len(self._models)*self._survival_rate):
			i, v = p_sorted[idx%len(p_sorted)]
			if np.random.uniform() <= v and not i in [x[0] for x in p_selected]:
				p_selected.append((i,v))
				selected.append(self._models[i])
			idx+=1
		return selected, p_selected

	# def crossover(self):
	# 	indices = np.arange(len(self._models))
	# 	result = []
	# 	for i in indices:
	# 		mates = np.random.choice(np.delete(indices, i), int(1/self._survival_rate), replace=False)
	# 		for m in mates:
	# 			new_weights = GeneticAlgorithmController.crossover_xy(
	# 				[i.numpy() for i in self._models[i].getModel().weights], 
	# 				[i.numpy() for i in self._models[m].getModel().weights]
	# 			)
				
	# 			clone_keras = tf.keras.models.clone_model(self._models[i].getModel())
	# 			clone_keras.build(input_shape=self._train_data[0][0].shape)

	# 			new_model = GeneticAlgorithmModel(clone_keras, self._evaluator)
	# 			new_model.setWeights(new_weights)

	# 			result.append(new_model)
	# 	self._models = result

	def crossover(self, p):
		p_sorted = sorted(enumerate([i[1] for i in p]), key=lambda x: x[1], reverse=True)
		indices = np.arange(len(self._models))
		result = [self._models[x[0]] for x in p_sorted[:int(len(p_sorted))]]
		i=0
		while len(result) < self._num_models:
			mates = np.random.choice(np.delete(indices, i), int(1/self._survival_rate), replace=False)
			for m in mates:
				new_weights = GeneticAlgorithmController.crossover_xy(
					[i.numpy() for i in self._models[i].getModel().weights], 
					[i.numpy() for i in self._models[m].getModel().weights]
				)
				
				clone_keras = tf.keras.models.clone_model(self._models[i].getModel())
				clone_keras.build(input_shape=self._train_data[0][0].shape)

				new_model = GeneticAlgorithmModel(clone_keras, self._evaluator)
				new_model.setWeights(new_weights)

				result.append(new_model)
			i = (i+1)%len(self._models)
		self._models = result

	def crossover_xy(x, y):
		for i in range(len(x)):
			for j in range(x[i].shape[0]):
				if np.random.uniform() >= 0.5:
					x[i][j] = y[i][j]
		return x

	def mutate(self):
		l = int(len(self._models)*self._survival_rate)
		for model in self._models[int(l*0.2):]:
			weights = [i.numpy() for i in model.getWeights()]
			new_weights = model.mutate(weights, self._mutation_rate, self._mean, self._std)
			model.setWeights(new_weights)

	def optimize(self, performance):
		self._models, p_selected = self.select(performance)
		self.crossover(p_selected)
		if self._mutation_rate:
			self.mutate()

	def load(self):
		return

	def save(self):
		return
		
# Test Evaluator module
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(
	16, return_sequences=True, 
	kernel_initializer=tf.keras.initializers.RandomNormal(mean=np.mean(X), stddev=np.std(X))
))
model.add(tf.keras.layers.LSTM(
	16, kernel_initializer=tf.keras.initializers.RandomNormal(mean=np.mean(X), stddev=np.std(X))
))
model.add(tf.keras.layers.Dense(
	1, activation='sigmoid',
	kernel_initializer=tf.keras.initializers.RandomNormal(mean=np.mean(X), stddev=np.std(X))
))

print('\nTesting evaluator.')
evaluator = SimplePipReturnEvaluator(threshold=0.4)
timer = timeit()
result = evaluator(y, model(X))
print('Evaluator Result: {}'.format(result))
timer.end()

# Test Model module
print('\nTesting model.')
timer = timeit()
test_model = GeneticAlgorithmModel(model, evaluator)
test_model(X, y)
timer.end()

print('\nTesting controller.')
controller = GeneticAlgorithmController(evaluator, num_models=300)
controller.fit(train_data=(X, y), generations=10)




