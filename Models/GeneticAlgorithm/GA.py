from numba import jit
import tensorflow as tf
import numpy as np
import math

'''
Genetic Algorithm Model
'''

class GeneticAlgorithmModel(object):

	def __init__(self, model, evaluator):
		self._model = model
		self._evaluator = evaluator

	def __call__(self, X, y, training=False):
		return self._evaluator(y, self._model(X))

	def getModel(self):
		return self._model

	def setModel(self, model):
		self._model = model

	def getWeights(self):
		return self._model.weights

	def setWeights(self, weights):
		self._model.set_weights(weights)

'''
Genetic Algorithm Main Controller
'''

class GeneticAlgorithm(object):
	# TODO: Add optimizer functionality to performance selection process

	def __init__(self, 
		evaluator, crossover_opt, mutation_opt=None,
		model_generator=generic_lstm_model_generator,
		num_models=100, survival_rate=0.5
	):
		self._evaluator = evaluator
		self._crossover_opt = crossover_opt
		self._mutation_opt = mutation_opt
		self._model_generator = model_generator
		self._num_models = num_models
		self._survival_rate = survival_rate

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

		self.build()
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
	
	def build(self):
		self._crossover_opt.build(
			self._survival_rate, self._num_models,
			self._evaluator, self._train_data[0][0].shape
		)

		self._mutation_opt.build(
			self._survival_rate, self._mean, self._std
		)

	def generateModels(self):
		self._models = []
		for i in range(self._num_models):
			model = self._model_generator(self._mean, self._std)
			self._models.append(GeneticAlgorithmModel(model, self._evaluator))
		print('Models initialized.')

	def select(self, fit):
		fit_scaled = (fit - fit.min()) / (fit.max() - fit.min())
		fit_sorted = sorted(enumerate(fit_scaled), key=lambda x: x[1], reverse=True)
		fit_selected = []
		selected = []
		idx = 0
		while len(selected) < int(len(self._models)*self._survival_rate):
			i, v = fit_sorted[idx%len(fit_sorted)]
			if np.random.uniform() <= v and not i in [x[0] for x in fit_selected]:
				fit_selected.append((i,v))
				selected.append(self._models[i])
			idx+=1
		return selected, [i[1] for i in fit_selected]

	def optimize(self, fit):
		selected = self.select(fit)
		self._crossover_opt(selected)
		if self._mutation_opt and self._mutate_func:
			self._mutation_opt(self._models)

	def load(self):
		return

	def save(self):
		return

'''
GA Evaluators
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

'''
GA Optimizers
'''

### Crossover optimizers

class GenericCrossover(object):
	def __init__(self, crossover_func=generic_crossover):
		self._crossover_func = crossover_func

	def build(survival_rate, num_models, evaluator, data_shape):
		self._survival_rate = survival_rate
		self._num_models = num_models
		self._evaluator = evaluator
		self._data_shape = data_shape

	def __call__(self, models):
		models = [i[0] for i in models]

		indices = np.arange(len(models))
		result = []
		for i in indices:
			mates = np.random.choice(np.delete(indices, i), int(1/self._survival_rate), replace=False)
			for m in mates:
				new_weights = self._crossover_func(
					[i.numpy() for i in models[i].getModel().weights], 
					[i.numpy() for i in models[m].getModel().weights]
				)
				
				clone_keras = tf.keras.models.clone_model(models[i].getModel())
				clone_keras.build(input_shape=self._data_shape)

				new_model = GeneticAlgorithmModel(clone_keras, self._evaluator)
				new_model.setWeights(new_weights)

				result.append(new_model)
		return result

class PreserveBestCrossover(object):
	def __init__(self, preserve_rate=0.2, crossover_func=generic_crossover):
		self._preserve_rate = preserve_rate
		self._crossover_func = crossover_func
		assert 0.0 <= self._survival_rate <= 1.0
		assert 0.0 <= self._preserve_rate <= 1.0

	def build(self, survival_rate, num_models, evaluator, data_shape):
		self._survival_rate = survival_rate
		self._num_models = num_models
		self._evaluator = evaluator
		self._data_shape = data_shape

	def __call__(self, models):
		fitness = [i[1] for i in models]
		models = [i[0] for i in models]

		fit_sorted = sorted(enumerate(fitness), key=lambda x: x[1], reverse=True)
		indices = np.arange(len(models))
		result = [models[x[0]] for x in fit_sorted[:int(len(fit_sorted) * self._preserve_rate)]]
		i=0
		while len(result) < self._num_models:
			mates = np.random.choice(np.delete(indices, i), int(1/self._survival_rate), replace=False)
			for m in mates:
				new_weights = self._crossover_func(
					[i.numpy() for i in models[i].getModel().weights], 
					[i.numpy() for i in models[m].getModel().weights]
				)
				
				clone_keras = tf.keras.models.clone_model(models[i].getModel())
				clone_keras.build(input_shape=self._data_shape)

				new_model = GeneticAlgorithmModel(clone_keras, self._evaluator)
				new_model.setWeights(new_weights)

				result.append(new_model)
			i = (i+1)%len(models)
		return result

### Crossover Functions

def generic_crossover(x, y):
	for i in range(len(x)):
		for j in range(x[i].shape[0]):
			if np.random.uniform() >= 0.5:
				x[i][j] = y[i][j]
	return x

### Mutation Optimizers

class GenericMutation(object):
	def __init__(self, mutation_rate=0.01, mutate_func=generic_mutate):
		self._mutation_rate = mutation_rate
		self._mutate_func = mutate_func

	def build(self, survival_rate, mean, std):
		self._survival_rate = survival_rate
		self._mean = mean
		self._std = std

	def __call__(self, models):
		for model in models:
			weights = [i.numpy() for i in model.getWeights()]
			new_weights = self._mutate_func(weights, self._mutation_rate, self._mean, self._std)
			model.setWeights(new_weights)

class PreserveBestMutation(object):
	def __init__(self, mutation_rate=0.01, mutate_func=generic_mutate):
		self._mutation_rate = mutation_rate
		self._mutate_func = mutate_func

	def build(self, survival_rate, mean, std):
		self._survival_rate = survival_rate
		self._mean = mean
		self._std = std

	def __call__(self, models):
		l = int(len(models)*self._survival_rate)
		for model in models[int(l*0.2):]:
			weights = [i.numpy() for i in model.getWeights()]
			new_weights = self._mutate_func(weights, self._mutation_rate, self._mean, self._std)
			model.setWeights(new_weights)

### Mutate Functions

def generic_mutate(weights, mutation_rate, mean, std):
	for i in range(len(weights)):
		if type(weights[i]) == np.ndarray:
			weights[i] = GenericMutate(weights[i], mutation_rate, mean, std)
		else:
			if np.random.uniform() <= mutation_rate:
				weights[i] = np.random.normal(mean, std)
	return weights

### Keras Model Generators

def generic_lstm_model_generator(mean, std):
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.LSTM(
		16, kernel_initializer=tf.keras.initializers.RandomNormal(mean=mean, stddev=std)
	))
	model.add(tf.keras.layers.Dense(
		1, activation='sigmoid',
		kernel_initializer=tf.keras.initializers.RandomNormal(mean=mean, stddev=std)
	))
	return model



