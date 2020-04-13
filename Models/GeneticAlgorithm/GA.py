from numba import jit
import numpy as np
import math
import sys
import time
import json
import os

try:
	import cupy as cp
except:
	pass

import bt

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

	def build(self, survival_rate, num_models):
		self._survival_rate = survival_rate
		self._num_models = num_models

	def __call__(self, models):
		models = [i for i in models[0]]

		indices = np.arange(len(models))
		i=0
		result = []
		while len(result) < self._num_models:
			# Random 10%
			if len(result) < int(len(models) * 0.1):
				new_model = models[0].newModel()
				new_model.setModelInfo(models[0].getModelInfo())
				gen_model = new_model.generateModel(
					new_model.getModelInfo()
				)
				new_model.setModel(gen_model)
				result.append(new_model)
			# Crossover remaining
			else:
				mates = np.random.choice(np.delete(indices, i), int(1/self._survival_rate), replace=False)
				for m in mates:
					i_weights = models[i].getWeights()
					m_weights = models[m].getWeights()

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
				i = (i+1)%len(models)
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
			# Random 10%
			if len(result) < int(len(fit_sorted) * (self._preserve_rate + 0.1)):
				new_model = models[0].newModel()
				new_model.setModelInfo(models[0].getModelInfo())
				gen_model = new_model.generateModel(
					new_model.getModelInfo()
				)
				new_model.setModel(gen_model)
				result.append(new_model)
			# Crossover remaining
			else:
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
		self._model_info = None

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
		survival_rate=0.5, is_gpu=False
	):
		self._crossover_opt = crossover_opt
		self._mutation_opt = mutation_opt
		self._survival_rate = survival_rate
		self._save = []
		self._save_best_train = None
		self._save_best_mp = None
		self.is_gpu = is_gpu

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
			if self.is_gpu:
				X_gpu = [train_data[0][i*batch_size:i*batch_size+batch_size] for i in math.ceil(train_data[0].shape[0]/batch_size)]
				self._train_data_gpu = X_gpu

			X = [train_data[0][i*batch_size:i*batch_size+batch_size] for i in math.ceil(train_data[0].shape[0]/batch_size)]
			y = [train_data[0][i*batch_size:i*batch_size+batch_size] for i in math.ceil(train_data[0].shape[0]/batch_size)]
			self._train_data = (X, y)
		else:
			if self.is_gpu:
				self._train_data_gpu = [cp.asarray(train_data[0])]

			self._train_data = ([train_data[0]], [train_data[1]])

		if val_data:
			if self.is_gpu:
				self._val_data_gpu = cp.asarray(val_data[0])

			self._val_data = (val_data[0], val_data[1])
		else:
			self._val_data = None

		self.build()

		if run:
			assert type(generations) == int
			self.run(generations)

	def run(self, generations=20):
		start = time.time()
		for generation in range(generations):
			train_fit, val_fit = self(gen=generation)
			self.report(generation, train_fit, val_fit)
			self._onSave(generation)

			if generation+1 == generations:
				self._onSaveBestTrain(train_fit)
				self._onSaveBestMP(train_fit, val_fit)
			else:
				self.optimize(np.array(train_fit))

		print('Training completed. {:.2f}s'.format(time.time() - start))

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
				if self.is_gpu:
					batch.append(
						mod(self._train_data_gpu[j], self._train_data[1][j], training=True)
					)
				else:
					batch.append(
						mod(self._train_data[0][j], self._train_data[1][j], training=True)
					)
			train_fit.append(batch)

			if self._val_data:
				if self.is_gpu:
					val_fit.append(mod(self._val_data_gpu, self._val_data[1]))
				else:
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
		if fit.max() != fit.min():	
			fit_scaled = (fit - fit.min()) / (fit.max() - fit.min())
		else:
			fit_scaled = fit / fit.min()

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

		for i in sorted(enumerate(train_arr), key=lambda x: x[1], reverse=True)[:3]:
			print(' ({}):\n{}'.format(i[0], self._models[i[0]]))

		if len(val_fit) > 0:
			print(' Mid Point:')
			mid = (train_arr + np.array(val_fit)) / 2.0
			count = 0
			for i in sorted(enumerate(mid), key=lambda x: x[1], reverse=True):
				if count >= 1:
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

	def saveBestTrain(self, num_models, name, data={}):
		self._save_best_train = (num_models, name, data)

	def saveBestMP(self, num_models, name, data={}):
		self._save_best_mp = (num_models, name, data)

	def _onSave(self, gen):
		for i in self._save:
			if i[0] == gen:
				print('Saving...\n%s' % self._models[i[1]])
				path = os.path.join('saved', '{}'.format(i[2]))
				if not os.path.exists(path):
					os.makedirs(path)
				path_weights = os.path.join(path, '{}.json'.format(len(os.listdir(path))))
				
				info = {'weights': [x.tolist() for x in self._models[i[1]].getWeights()]}
				info.update(i[3])
				try:
					info.update(self._models[i[1]].save())
				except:
					pass
				with open(path_weights, 'w') as f:
					f.write(json.dumps(info, indent=2))

	def _onSaveBestTrain(self, train_fit):
		if self._save_best_train:
			train_arr = np.array(train_fit)[:,0]
			count = 0
			for i in sorted(enumerate(train_arr), key=lambda x: x[1], reverse=True):
				if count >= self._save_best_train[0]:
					break

				print('Saving...\n%s' % self._models[i[0]])
				path = os.path.join('saved', '{}'.format(self._save_best_train[1]))
				if not os.path.exists(path):
					os.makedirs(path)
				path_weights = os.path.join(path, '{}.json'.format(len(os.listdir(path))))
				
				info = {'weights': [x.tolist() for x in self._models[i[0]].getWeights()]}
				info.update(self._save_best_train[2])
				try:
					info.update(self._models[i[0]].save())
				except:
					pass

				with open(path_weights, 'w') as f:
					f.write(json.dumps(info, indent=2))

				count += 1

	def _onSaveBestMP(self, train_fit, val_fit):
		if self._save_best_mp:
			train_arr = np.array(train_fit)[:,0]
			mid = (train_arr + np.array(val_fit)) / 2.0
			count = 0
			for i in sorted(enumerate(mid), key=lambda x: x[1], reverse=True):
				if count >= self._save_best_mp[0]:
					break

				print('Saving...\n%s' % self._models[i[0]])
				path = os.path.join('saved', '{}'.format(self._save_best_mp[1]))
				if not os.path.exists(path):
					os.makedirs(path)
				path_weights = os.path.join(path, '{}.json'.format(len(os.listdir(path))))
				
				info = {'weights': [x.tolist() for x in self._models[i[0]].getWeights()]}
				info.update(self._save_best_mp[2])
				try:
					info.update(self._models[i[0]].save())
				except:
					pass

				with open(path_weights, 'w') as f:
					f.write(json.dumps(info, indent=2))

				count += 1


'''
ML Layers
'''

# Dense layer
class Dense(object):
	def __init__(self, in_size, hl_size):
		self.init_weights(in_size, hl_size)

	def init_weights(self, in_size, hl_size):
		self.weights = np.random.normal(size=(hl_size, in_size))
		self.bias = np.random.normal(size=(hl_size, in_size))

	def __call__(self, data):
		return Dense.run(data)

	@jit(forceobj=True)
	def run(data):
		return np.dot(self.weights, data) + self.bias


# LSTM layer
class LSTM(object):
	def __init__(self, in_size, hl_size, out_size):
		self.in_size = in_size
		self.hl_size = hl_size
		self.out_size = out_size
		self.init_weights(in_size, hl_size, out_size)

	# Initialize Weights and Biases
	def init_weights(self, in_size, hl_size, out_size):
		weights_xi = np.random.normal(size=(hl_size, in_size))
		weights_xf = np.random.normal(size=(hl_size, in_size))
		weights_xl = np.random.normal(size=(hl_size, in_size))
		weights_xo = np.random.normal(size=(hl_size, in_size))
		self.weights_x = np.concatenate((
			weights_xi, weights_xf, weights_xl, weights_xo
		))

		bias_xi = np.random.normal(size=(hl_size))
		bias_xf = np.random.normal(size=(hl_size))
		bias_xl = np.random.normal(size=(hl_size))
		bias_xo = np.random.normal(size=(hl_size))
		self.bias_x = np.concatenate((
			bias_xi, bias_xf, bias_xl, bias_xo
		))

		weights_hi = np.random.normal(size=(hl_size, hl_size))
		weights_hf = np.random.normal(size=(hl_size, hl_size))
		weights_hl = np.random.normal(size=(hl_size, hl_size))
		weights_ho = np.random.normal(size=(hl_size, hl_size))
		self.weights_h = np.concatenate((
			weights_hi, weights_hf, weights_hl, weights_ho
		))

		bias_hi = np.random.normal(size=(hl_size))
		bias_hf = np.random.normal(size=(hl_size))
		bias_hl = np.random.normal(size=(hl_size))
		bias_ho = np.random.normal(size=(hl_size))
		self.bias_h = np.concatenate((
			bias_hi, bias_hf, bias_hl, bias_ho
		))

		self.weights_out = np.random.normal(size=(hl_size, out_size))
		self.bias_out = np.random.normal(size=(out_size))

	def get_weights(self):
		return [
			np.copy(self.weights_x), np.copy(self.bias_x),
			np.copy(self.weights_h), np.copy(self.bias_h),
			np.copy(self.weights_out), np.copy(self.bias_out)
		]

	def set_weights(self, weights):
		self.weights_x = weights[0]
		self.bias_x = weights[1]
		self.weights_h = weights[2]
		self.bias_h = weights[3]
		self.weights_out = weights[4]
		self.bias_out = weights[5]

	def __call__(self, data):
		return LSTM.run(
			data, self.hl_size, self.out_size, 
			self.weights_x, self.bias_x, 
			self.weights_h, self.bias_h, 
			self.weights_out, self.bias_out
		)

	def run(
		data, hl_size, out_size, weights_x, bias_x, 
		weights_h, bias_h, weights_out, bias_out
	):
		# Initialize cell and hidden states with zeroes
		h = np.zeros(hl_size)
		c = np.zeros(hl_size)
		off = hl_size

		f = LSTM.forget_gate(data.T, h, weights_h[off:off*2], bias_h[off:off*2], weights_x[off:off*2], bias_x[off:off*2], c)
		i = LSTM.input_gate(data.T, h, weights_h[0:off], bias_h[0:off], weights_x[0:off], bias_x[0:off],
				weights_h[off*2:off*3], bias_h[off*2:off*3], weights_x[off*2:off*3], bias_x[off*2:off*3])
		c = LSTM.cell_state(f, i)
		h = LSTM.output_gate(data.T, h, weights_h[off*3:off*4], bias_h[off*3:off*4], weights_x[off*3:off*4], bias_x[off*3:off*4], c)
		return LSTM.model_output(h, weights_out, bias_out)

	def forget_gate(x, h, weights_hf, bias_hf, weights_xf, bias_xf, prev_cell_state):
		forget_hidden = np.dot(weights_hf, h) + bias_hf
		forget_eventx = np.dot(weights_xf, x).T + bias_xf
		return np.multiply(bt.sigmoid(forget_hidden + forget_eventx), prev_cell_state)

	def input_gate(
		x, h, 
		weights_hi, bias_hi, 
		weights_xi, bias_xi, 
		weights_hl, bias_hl, 
		weights_xl, bias_xl
	):
		ignore_hidden = np.dot(weights_hi, h) + bias_hi
		ignore_eventx = np.dot(weights_xi, x).T + bias_xi
		learn_hidden = np.dot(weights_hl, h) + bias_hl
		learn_eventx = np.dot(weights_xl, x).T + bias_xl
		return np.multiply(bt.sigmoid(ignore_eventx + ignore_hidden), np.tanh(learn_eventx + learn_hidden))

	def cell_state(forget_gate_output, input_gate_output):
		return forget_gate_output + input_gate_output

	def output_gate(x, h, weights_ho, bias_ho, weights_xo, bias_xo, cell_state):
		out_hidden = np.dot(weights_ho, h) + bias_ho
		out_eventx = np.dot(weights_xo, x).T + bias_xo
		return np.multiply(bt.sigmoid(out_eventx + out_hidden), np.tanh(cell_state))

	def model_output(lstm_output, weights_out, bias_out):
	  '''Takes the LSTM output and transforms it to our desired 
	  output size using a final, fully connected layer'''
	  return np.dot(lstm_output, weights_out) + bias_out

# LSTM GPU layer
class LSTM_GPU(object):
	def __init__(self, in_size, hl_size, out_size):
		self.in_size = in_size
		self.hl_size = hl_size
		self.out_size = out_size
		self.init_weights(in_size, hl_size, out_size)

	# Initialize Weights and Biases
	def init_weights(self, in_size, hl_size, out_size):
		weights_xi = cp.random.normal(size=(hl_size, in_size))
		weights_xf = cp.random.normal(size=(hl_size, in_size))
		weights_xl = cp.random.normal(size=(hl_size, in_size))
		weights_xo = cp.random.normal(size=(hl_size, in_size))
		self.weights_x = cp.concatenate((
			weights_xi, weights_xf, weights_xl, weights_xo
		))

		bias_xi = cp.random.normal(size=(hl_size))
		bias_xf = cp.random.normal(size=(hl_size))
		bias_xl = cp.random.normal(size=(hl_size))
		bias_xo = cp.random.normal(size=(hl_size))
		self.bias_x = cp.concatenate((
			bias_xi, bias_xf, bias_xl, bias_xo
		))

		weights_hi = cp.random.normal(size=(hl_size, hl_size))
		weights_hf = cp.random.normal(size=(hl_size, hl_size))
		weights_hl = cp.random.normal(size=(hl_size, hl_size))
		weights_ho = cp.random.normal(size=(hl_size, hl_size))
		self.weights_h = cp.concatenate((
			weights_hi, weights_hf, weights_hl, weights_ho
		))

		bias_hi = cp.random.normal(size=(hl_size))
		bias_hf = cp.random.normal(size=(hl_size))
		bias_hl = cp.random.normal(size=(hl_size))
		bias_ho = cp.random.normal(size=(hl_size))
		self.bias_h = cp.concatenate((
			bias_hi, bias_hf, bias_hl, bias_ho
		))

		self.weights_out = cp.random.normal(size=(hl_size, out_size))
		self.bias_out = cp.random.normal(size=(out_size))

	def get_weights(self):
		return [
			cp.asnumpy(self.weights_x), cp.asnumpy(self.bias_x),
			cp.asnumpy(self.weights_h), cp.asnumpy(self.bias_h),
			cp.asnumpy(self.weights_out), cp.asnumpy(self.bias_out)
		]

	def set_weights(self, weights):
		self.weights_x = cp.asarray(weights[0])
		self.bias_x = cp.asarray(weights[1])
		self.weights_h = cp.asarray(weights[2])
		self.bias_h = cp.asarray(weights[3])
		self.weights_out = cp.asarray(weights[4])
		self.bias_out = cp.asarray(weights[5])

	def __call__(self, data):

		return LSTM_GPU.run(
			data, self.hl_size, self.out_size, 
			self.weights_x, self.bias_x, 
			self.weights_h, self.bias_h, 
			self.weights_out, self.bias_out
		)

	def run(
		data, hl_size, out_size, weights_x, bias_x, 
		weights_h, bias_h, weights_out, bias_out
	):
		# Initialize cell and hidden states with zeroes
		h = cp.zeros(hl_size)
		c = cp.zeros(hl_size)
		off = hl_size

		f = LSTM_GPU.forget_gate(data.T, h, weights_h[off:off*2], bias_h[off:off*2], weights_x[off:off*2], bias_x[off:off*2], c)
		i = LSTM_GPU.input_gate(data.T, h, weights_h[0:off], bias_h[0:off], weights_x[0:off], bias_x[0:off],
				weights_h[off*2:off*3], bias_h[off*2:off*3], weights_x[off*2:off*3], bias_x[off*2:off*3])
		c = LSTM_GPU.cell_state(f, i)
		h = LSTM_GPU.output_gate(data.T, h, weights_h[off*3:off*4], bias_h[off*3:off*4], weights_x[off*3:off*4], bias_x[off*3:off*4], c)
		return LSTM_GPU.model_output(h, weights_out, bias_out)

	def forget_gate(x, h, weights_hf, bias_hf, weights_xf, bias_xf, prev_cell_state):
		forget_hidden = cp.dot(weights_hf, h) + bias_hf
		forget_eventx = cp.dot(weights_xf, x).T + bias_xf
		return cp.multiply(bt.sigmoid_gpu(forget_hidden + forget_eventx), prev_cell_state)

	def input_gate(
		x, h, 
		weights_hi, bias_hi, 
		weights_xi, bias_xi, 
		weights_hl, bias_hl, 
		weights_xl, bias_xl
	):
		ignore_hidden = cp.dot(weights_hi, h) + bias_hi
		ignore_eventx = cp.dot(weights_xi, x).T + bias_xi
		learn_hidden = cp.dot(weights_hl, h) + bias_hl
		learn_eventx = cp.dot(weights_xl, x).T + bias_xl
		return cp.multiply(bt.sigmoid_gpu(ignore_eventx + ignore_hidden), cp.tanh(learn_eventx + learn_hidden))

	def cell_state(forget_gate_output, input_gate_output):
		return forget_gate_output + input_gate_output

	def output_gate(x, h, weights_ho, bias_ho, weights_xo, bias_xo, cell_state):
		out_hidden = cp.dot(weights_ho, h) + bias_ho
		out_eventx = cp.dot(weights_xo, x).T + bias_xo
		return cp.multiply(bt.sigmoid_gpu(out_eventx + out_hidden), cp.tanh(cell_state))

	def model_output(lstm_output, weights_out, bias_out):
	  '''Takes the LSTM output and transforms it to our desired 
	  output size using a final, fully connected layer'''
	  return cp.dot(lstm_output, weights_out) + bias_out


# RNN GPU layer
class RNN_1D_GPU(object):
	def __init__(self, hl_size, out_size):
		self.hl_size = hl_size
		self.out_size = out_size
		self.init_weights(hl_size, out_size)

	# Initialize Weights and Biases
	def init_weights(self, hl_size, out_size):
		self.weights_x = cp.random.normal(size=(hl_size, 1))
		self.bias_x = cp.random.normal(size=(hl_size))

		self.weights_out = cp.random.normal(size=(hl_size, out_size))
		self.bias_out = cp.random.normal(size=(out_size))

	def get_weights(self):
		return [
			cp.asnumpy(self.weights_x), cp.asnumpy(self.bias_x),
			cp.asnumpy(self.weights_out), cp.asnumpy(self.bias_out)
		]

	def set_weights(self, weights):
		self.weights_x = cp.asarray(weights[0])
		self.bias_x = cp.asarray(weights[1])
		self.weights_out = cp.asarray(weights[2])
		self.bias_out = cp.asarray(weights[3])

	def __call__(self, data):

		return RNN_1D_GPU.run(
			data, self.hl_size, 
			self.weights_x, self.bias_x, 
			self.weights_out, self.bias_out
		)

	def run(
		data, hl_size, 
		weights_x, bias_x, 
		weights_out, bias_out
	):
		c = cp.ones((data.shape[0], hl_size))
		off = hl_size

		for i in range(data.shape[1]):
			c = RNN_1D_GPU.output_gate(data[:,i,:].T, weights_x, bias_x, c)

		return RNN_1D_GPU.model_output(c, weights_out, bias_out)

	def output_gate(x, weights_x, bias_x, cell_state):
		out_eventx = cp.dot(weights_x, x).T + bias_x
		return cp.multiply(bt.relu_gpu(out_eventx), cell_state)

	def model_output(lstm_output, weights_out, bias_out):
	  return cp.dot(lstm_output, weights_out) + bias_out

# RNN (w/ Ignore Gate) layer
class RNN_TWO(object):
	def __init__(self, inpt_size, hl_size, out_size):
		self.hl_size = hl_size
		self.init_weights(inpt_size, hl_size, out_size)

	# Initialize Weights and Biases
	def init_weights(self, inpt_size, hl_size, out_size):
		weights_xo = np.random.normal(size=(hl_size, inpt_size))
		weights_xi = np.random.normal(size=(hl_size, inpt_size))

		self.weights_x = np.concatenate((
			weights_xo, weights_xi
		))

		bias_xo = np.random.normal(size=(hl_size))
		bias_xi = np.random.normal(size=(hl_size))

		self.bias_x = np.concatenate((
			bias_xo, bias_xi
		))

		self.weights_out = np.random.normal(size=(hl_size, out_size))
		self.bias_out = np.random.normal(size=(out_size))

	def get_weights(self):
		return [
			np.copy(self.weights_x), np.copy(self.bias_x),
			np.copy(self.weights_out), np.copy(self.bias_out)
		]

	def set_weights(self, weights):
		self.weights_x = weights[0]
		self.bias_x = weights[1]
		self.weights_out = weights[2]
		self.bias_out = weights[3]

	def __call__(self, data):

		return RNN_TWO.run(
			data, self.hl_size, 
			self.weights_x, self.bias_x, 
			self.weights_out, self.bias_out
		)

	def run(
		data, hl_size, 
		weights_x, bias_x, 
		weights_out, bias_out
	):
		c = np.ones((data.shape[0], hl_size))
		off = hl_size

		for i in range(data.shape[1]):
			c = RNN_TWO.output_gate(data[:,i,:].T, weights_x[:off], bias_x[:off], weights_x[off:off*2], bias_x[off:off*2], c)

		return RNN_TWO.model_output(c, weights_out, bias_out)

	def output_gate(x, weights_xo, bias_xo, weights_xi, bias_xi, cell_state):
		out_eventx = np.dot(weights_xo, x).T + bias_xo
		ignore_eventx = np.dot(weights_xi, x).T + bias_xi
		x_out = np.multiply(bt.sigmoid(ignore_eventx), bt.relu(out_eventx))
		return np.multiply(x_out, cell_state)

	def model_output(lstm_output, weights_out, bias_out):
	  return np.dot(lstm_output, weights_out) + bias_out

# RNN (w/ Ignore Gate) GPU layer
class RNN_TWO_GPU(object):
	def __init__(self, inpt_size, hl_size, out_size):
		self.hl_size = hl_size
		self.init_weights(inpt_size, hl_size, out_size)

	# Initialize Weights and Biases
	def init_weights(self, inpt_size, hl_size, out_size):
		weights_xo = cp.random.normal(size=(hl_size, inpt_size))
		weights_xi = cp.random.normal(size=(hl_size, inpt_size))

		self.weights_x = cp.concatenate((
			weights_xo, weights_xi
		))

		bias_xo = cp.random.normal(size=(hl_size))
		bias_xi = cp.random.normal(size=(hl_size))

		self.bias_x = cp.concatenate((
			bias_xo, bias_xi
		))

		self.weights_out = cp.random.normal(size=(hl_size, out_size))
		self.bias_out = cp.random.normal(size=(out_size))

	def get_weights(self):
		return [
			cp.asnumpy(self.weights_x), cp.asnumpy(self.bias_x),
			cp.asnumpy(self.weights_out), cp.asnumpy(self.bias_out)
		]

	def set_weights(self, weights):
		self.weights_x = cp.asarray(weights[0])
		self.bias_x = cp.asarray(weights[1])
		self.weights_out = cp.asarray(weights[2])
		self.bias_out = cp.asarray(weights[3])

	def __call__(self, data):

		return RNN_TWO_GPU.run(
			data, self.hl_size, 
			self.weights_x, self.bias_x, 
			self.weights_out, self.bias_out
		)

	def run(
		data, hl_size, 
		weights_x, bias_x, 
		weights_out, bias_out
	):
		c = cp.ones((data.shape[0], hl_size))
		off = hl_size

		for i in range(data.shape[1]):
			c = RNN_TWO_GPU.output_gate(data[:,i,:].T, weights_x[:off], bias_x[:off], weights_x[off:off*2], bias_x[off:off*2], c)

		return RNN_TWO_GPU.model_output(c, weights_out, bias_out)

	def output_gate(x, weights_xo, bias_xo, weights_xi, bias_xi, cell_state):
		out_eventx = cp.dot(weights_xo, x).T + bias_xo
		ignore_eventx = cp.dot(weights_xi, x).T + bias_xi
		x_out = cp.multiply(bt.sigmoid_gpu(ignore_eventx), bt.relu_gpu(out_eventx))
		return cp.multiply(x_out, cell_state)

	def model_output(lstm_output, weights_out, bias_out):
	  return cp.dot(lstm_output, weights_out) + bias_out

