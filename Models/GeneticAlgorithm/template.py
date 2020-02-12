"""

Performance driven deep learning model. Model uses
genetic algorithm concepts to find a solution.

Created by: Ethan Hollins

"""

from DataLoader import DataLoader
import Constants
import datetime as dt
import tensorflow as tf
import math


'''
Data Preprocessing
'''

# Load GBPUSD ohlc data
dl = DataLoader()
df = dl.get(Constants.GBPUSD, Constants.ONE_HOUR, start=dt.datetime(2010,1,1))
df = df.drop(columns=[k for k in df.keys() if k.startswith('ask')])

print(df.head(5))
print(df.shape)



'''
Build Genetic Algorithm model
'''

def SimpleBuySellEvaluator(object):

	def __init__(threshold=0.5):
		self.threshold = threshold
	
	# @tf.function
	def __call__(X, y, layers):
		assert layers[-1].units == 1

		result = tf.Variable(0.0, tf.float32)
		last_dir = tf.Variable(-1, tf.int32)
		last_entry = tf.Variable(0.0, tf.float32)
		c_dir = tf.Variable(-1, tf.int32)

		for i in range(X.shape[0]):
			x = tf.Variable(X[i])
			for layer in layers:
				x.assign(layer(x))

			if x[0] > (1.0 - self.threshold):
				c_dir.assign(1)
			elif x[0] < self.threshold:
				c_dir.assign(0)
			else:
				c_dir.assign(-1)

			if c_dir != -1:
				if last_dir != -1:
					if last_dir != c_dir:
						if last_entry[0] == 1:
							result.assign(result + (y[i] - last_entry))
						else:
							result.assign(result + (last_entry - y[i]))
						last_dir.assign(c_dir)
						last_entry.assign(y[i])
				else:
					last_dir.assign(c_dir)
					last_entry.assign(y[i])

		return tf.constant(result, tf.float32)

class GeneticAlgorithmModel(object):

	def __init__(self, evaluator, layers=[]):
		self._evaluator = evaluator
		self._layers = layers

	def add(self, layer):
		self.layers.append(layer)

	@tf.function
	def __call__(self, X, y, training=False):
		return self._evaluator(X, y, layers)

class GeneticAlgorithmController(object):

	def __init__(self, models=[], survival_rate=0.5, mutation_rate=0.01):
		self._models = models
		self._survival_rate = tf.constant(survival_rate, tf.float32)
		self._mutation_rate = tf.constant(mutation_rate, tf.float32)

	def add(self, model):
		self._models.append(model)

	def fit(self, train_data, val_data=None, batch_size=0, epochs=20, run=True):
		assert type(batch_size) == int
		assert type(train_data) == set
		train_data = (tf.constant(train_data), tf.constant(train_data))

		if batch_size:
			X = [train_data[0][i*batch_size:i*batch_size+batch_size] for i in math.ceil(train_data[0].shape[0]/batch_size)]
			y = [train_data[0][i*batch_size:i*batch_size+batch_size] for i in math.ceil(train_data[0].shape[0]/batch_size)]
			self._train_data = (X, y)
		else:
			self._train_data = ([train_data[0]], [train_data[1]])

		if validation_data:
			assert type(validation_data) == set
			self._val_data = (tf.constant(validation_data[0]), tf.constant(validation_data[1]))
		else:
			self._val_data = None

		assert type(epochs) == int
		self._epochs = epochs

		if run:
			self.run()

	def run(self):
		for epoch in self._epochs:
			performance_l = self()
			self.optimize(performance_l)

	def __call__(self):
		performance_l = []
		for model in self._models:
			performance_l.append(model(self._train_data[0], self._train_data[1]))
		return performance_l

	@tf.function
	def select(self, performance):
		# TODO: select _models based on random choice by chance performance

	@tf.function
	def crossover(self):
		# TODO: _models crossover
		return

	@tf.function
	def mutate(self):
		# TODO: _models mutate
		return

	def optimize(self, performance):
		self._models = self.select(performance)
		self.crossover()
		if self._mutation_rate:
			self.mutate()

	def load(self):
		return

	def save(self):
		return
		
	


	





