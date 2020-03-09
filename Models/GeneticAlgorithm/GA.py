from numba import jit
import numpy as np
import math
import sys
import time
import json
import os

'''
Generic functions
'''

@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

'''
GA Optimizers
'''

### Crossover Functions

def generic_crossover(x, y):
	for i in range(len(x)):
		if type(x[i]) == np.ndarray:
			x[i] = generic_crossover(x[i], y[i])
		else:
			if np.random.random() >= 0.5:
				x[i] = y[i]
	return x

def perfect_crossover(x, y):
	for i in range(len(x)):
		if len(x[i].shape) == 1:
			idx = np.random.choice(x[i].size, size=int(x[i].size/2)+1)
			x[i][idx] = y[i][idx]
		else:
			x[i] = perfect_crossover(x[i], y[i])
	return x

def layer_crossover(x, y):
	idx = np.random.choice(len(x), size=int(len(x)/2)+1)
	for i in idx:
		x[i] = y[i]
	return x

### Crossover optimizers

class GenericCrossover(object):
	def __init__(self, crossover_func=generic_crossover):
		self._crossover_func = crossover_func

	def build(survival_rate, num_models):
		self._survival_rate = survival_rate
		self._num_models = num_models

	def __call__(self, models):
		models = [i for i in models[0]]

		indices = np.arange(len(models))
		result = []
		for i in indices:
			mates = np.random.choice(np.delete(indices, i), int(1/self._survival_rate), replace=False)
			for m in mates:
				i_weights = models[i].getModel().get_weights()
				m_weights = models[m].getModel().get_weights()

				new_weights = self._crossover_func(
					i_weights, m_weights
				)

				new_model = models[i].newModel()
				gen_model = new_model.generateModel(
					models[i].getModelInfo()
				)
				new_model.setModel(gen_model)
				new_model.setWeights(new_weights)

				result.append(new_model)
				if len(result) >= self._num_models:
					return result
		return result


class PreserveBestCrossover(object):
	def __init__(self, preserve_rate=0.2, crossover_func=generic_crossover):
		self._preserve_rate = preserve_rate
		self._crossover_func = crossover_func
		assert 0.0 <= self._preserve_rate <= 1.0

	def build(self, survival_rate, num_models):
		self._survival_rate = survival_rate
		self._num_models = num_models

	def __call__(self, models):
		fitness = [i[0] for i in models[1]]
		models = [i for i in models[0]]

		fit_sorted = sorted(enumerate(fitness), key=lambda x: x[1], reverse=True)
		indices = np.arange(len(models))
		result = [models[x[0]] for x in fit_sorted[:int(len(fit_sorted) * self._preserve_rate)]]
		i=0
		while len(result) < self._num_models:
			mates = np.random.choice(np.delete(indices, i), int(1/self._survival_rate), replace=False)
			for m in mates:
				i_weights = models[i].getWeights()
				m_weights = models[m].getWeights()

				new_weights = self._crossover_func(
					i_weights, m_weights
				)

				new_model = models[i].newModel()
				new_model.setModelInfo(models[i].getModelInfo())
				gen_model = new_model.generateModel(
					new_model.getModelInfo()
				)
				new_model.setModel(gen_model)
				new_model.setWeights(new_weights)

				result.append(new_model)
				if len(result) >= self._num_models:
					return result
			i = (i+1)%len(models)
		return result

### Mutate Functions

def generic_mutate(weights, mutation_rate, mean, std):
	for i in range(len(weights)):
		if type(weights[i]) == np.ndarray:
			weights[i] = generic_mutate(weights[i], mutation_rate, mean, std)
		else:
			if np.random.uniform() <= mutation_rate:
				weights[i] = np.random.normal(mean, std)
	return weights

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
			weights = model.getWeights()
			new_weights = self._mutate_func(weights, self._mutation_rate, self._mean, self._std)
			model.setWeights(new_weights)


class PreserveBestMutation(object):
	def __init__(self, mutation_rate=0.01, preserve_rate=0.2, mutate_func=generic_mutate):
		self._mutation_rate = mutation_rate
		self._preserve_rate = preserve_rate
		self._mutate_func = mutate_func

	def build(self, survival_rate, mean, std):
		self._survival_rate = survival_rate
		self._mean = mean
		self._std = std

	def __call__(self, models):
		l = int(len(models)*self._survival_rate)
		for model in models[int(l*self._preserve_rate):]:
			weights = model.getWeights()
			new_weights = self._mutate_func(weights, self._mutation_rate, self._mean, self._std)
			model.setWeights(new_weights)


'''
Genetic Algorithm Model
'''

class GeneticAlgorithmModel(object):

	def __init__(self):
		self._model = None

	def __call__(self, X, y, training=False):
		if not self._model:
			self._model_info = []
			try:
				self._model_info = [
					np.mean(X), np.std(X),
					X.shape
				]
			except:
				pass
			self._model = self.generateModel(self.getModelInfo())

	def generateModel(self, model_info):
		raise Exception('Generate Model function not found.')
		return

	def getModel(self):
		return self._model

	def setModel(self, model):
		self._model = model

	def getModelInfo(self):
		return self._model_info

	def setModelInfo(self, model_info):
		self._model_info = model_info

	def getWeights(self):
		return self._model.get_weights()

	def setWeights(self, weights):
		self._model.set_weights(weights)

	def newModel(self):
		return GeneticAlgorithmModel()

class SimplePipReturnModel(GeneticAlgorithmModel):

	def __init__(self, threshold=0.5):
		super().__init__()
		self._threshold = threshold

	def __call__(self, X, y, training=False):
		super().__call__(X, y, training)
		return SimplePipReturnModel.run(y, self._model(X).numpy(), self._threshold)

	def newModel(self):
		return SimplePipReturnModel(self._threshold)

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
Genetic Algorithm Main Controller
'''

class GeneticAlgorithm(object):
	# TODO: Add optimizer functionality to performance selection process

	def __init__(self, 
		crossover_opt, mutation_opt=None,
		survival_rate=0.5
	):
		self._crossover_opt = crossover_opt
		self._mutation_opt = mutation_opt
		self._survival_rate = survival_rate
		self._save = []
		self._save_best = None

	def add(self, model):
		self._models.append(model)

	def fit(self, models, train_data, val_data=None, batch_size=0, generations=20, run=True):
		self._models = models
		self._num_models = len(models)
		
		assert type(batch_size) == int
		self._mean = np.mean(train_data[0])
		self._std = np.std(train_data[0])

		train_data = (train_data[0], train_data[1])


		if batch_size:
			X = [train_data[0][i*batch_size:i*batch_size+batch_size] for i in math.ceil(train_data[0].shape[0]/batch_size)]
			y = [train_data[0][i*batch_size:i*batch_size+batch_size] for i in math.ceil(train_data[0].shape[0]/batch_size)]
			self._train_data = (X, y)
		else:
			self._train_data = ([train_data[0]], [train_data[1]])

		if val_data:
			self._val_data = (val_data[0], val_data[1])
		else:
			self._val_data = None

		self.build()

		if run:
			assert type(generations) == int
			self.run(generations)

	def run(self, generations=20):
		for generation in range(generations):
			train_fit, val_fit = self(gen=generation)
			self.report(generation, train_fit, val_fit)
			self._onSave(generation)

			if generation+1 == generations:
				self._onSaveBest(train_fit, val_fit)
			else:
				self.optimize(np.array(train_fit))

		print('Training completed.')

	def __call__(self, gen=0):
		train_fit = []
		val_fit = []

		num_models = len(self._models)
		start = time.time()
		print()
		for i in range(num_models):
			mod = self._models[i]
			batch = []
			for j in range(len(self._train_data[0])):
				batch.append(
					mod(self._train_data[0][j], self._train_data[1][j], training=True)
				)
			train_fit.append(batch)

			if self._val_data:
				val_fit.append(mod(*self._val_data))

			progress = int((i+1)/num_models * 20.0)
			print('({}) Progress: [{}{}] - {}/{} '.format(
				gen+1,
				'='*(progress), 
				('>' if 20-progress > 0 else '')+(' '*int(20-progress-1)), 
				i+1, num_models
			), end='' if i == num_models-1 else '\r', flush=True)

		print('{:.2f}s'.format(
			time.time() - start
		))
		return train_fit, val_fit
	
	def build(self):
		self._crossover_opt.build(
			self._survival_rate, self._num_models
		)

		self._mutation_opt.build(
			self._survival_rate, self._mean, self._std
		)

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
		start = time.time()
		print(' Performing Selection...', end='\r', flush=True)
		selected = self.select(fit)
		print(' Performing Selection... {:.2f}s'.format(time.time() - start))
		
		start = time.time()
		print(' Performing Crossover...', end='\r', flush=True)
		self._models = self._crossover_opt(selected)
		print(' Performing Crossover... {:.2f}s'.format(time.time() - start))

		if self._mutation_opt:
			start = time.time()
			print(' Performing Mutation...', end='\r', flush=True)
			self._mutation_opt(self._models)
			print(' Performing Mutation... {:.2f}s'.format(time.time() - start))

	def report(self, gen, train_fit, val_fit):
		train_arr = np.array(train_fit)[:,0]
		train_best = np.amax(train_arr)
		train_median = np.sort(train_arr)[::-1][int(len(train_arr)/2)]

		if len(val_fit) > 0:
			val_best = np.amax(val_fit)
			val_median = np.sort(val_fit)[::-1][int(len(val_fit)/2)]

		print('\n Train | Best: {:.2f}\tMedian: {:.2f}'.format(
			train_best, train_median
		))
		
		if len(val_fit) > 0:
			print(' Val   | Best: {:.2f}\tMedian: {:.2f}\n'.format(
				val_best, val_median
			))

		for i in sorted(enumerate(train_arr), key=lambda x: x[1], reverse=True)[:5]:
			print(' ({}):\n{}'.format(i[0], self._models[i[0]]))

		if len(val_fit) > 0:
			print(' Mid Point:')
			mid = (train_arr + np.array(val_fit)) / 2.0
			count = 0
			for i in sorted(enumerate(mid), key=lambda x: x[1], reverse=True):
				if count >= 10:
					break

				if i[1] != 0.0:
					print(' ({}):\n{}'.format(i[0], self._models[i[0]]))
					count += 1

	def setSeed(self, seed):
		np.random.seed(seed)

	def load(self):
		return

	def save(self, gen, model_num, name, data={}):
		self._save.append((gen, model_num, name, data))

	def saveBest(self, num_models, name, data={}):
		self._save_best = (num_models, name, data)

	def _onSave(self, gen):
		for i in self._save:
			if i[0] == gen:
				print('Saving...\n%s' % self._models[i[1]])
				path = os.path.join('saved', '{}'.format(i[2]))
				if not os.path.exists(path):
					os.makedirs(path)
				path_weights = os.path.join(path, '{}.json'.format(len(os.listdir(path))))
				
				weights = {'weights': [x.tolist() for x in self._models[i[1]].getWeights()]}
				weights.update(i[3])
				with open(path_weights, 'w') as f:
					f.write(json.dumps(weights, indent=2))

	def _onSaveBest(self, train_fit, val_fit):
		if self._save_best:
			train_arr = np.array(train_fit)[:,0]
			mid = (train_arr + np.array(val_fit)) / 2.0
			count = 0
			for i in sorted(enumerate(mid), key=lambda x: x[1], reverse=True):
				if count >= self._save_best[0]:
					break

				print('Saving...\n%s' % self._models[i[0]])
				path = os.path.join('saved', '{}'.format(self._save_best[1]))
				if not os.path.exists(path):
					os.makedirs(path)
				path_weights = os.path.join(path, '{}.json'.format(len(os.listdir(path))))
				
				weights = {'weights': [x.tolist() for x in self._models[i[0]].getWeights()]}
				weights.update(self._save_best[2])
				with open(path_weights, 'w') as f:
					f.write(json.dumps(weights, indent=2))

				count += 1