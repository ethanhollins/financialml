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
train_data_one = getDonchUpDown(
	df.values[:,0],
	df.values[:,1],
	donch_period
)
timer.end()

print('Donch Data:\n%s'%train_data_one[-5:])
print(train_data_one.shape)

@jit
def getRsiData(close, period):
	X = np.zeros((close.shape[0]-period), dtype=np.float32)
	gain = np.zeros((close.shape[0]-period), dtype=np.float32)
	loss = np.zeros((close.shape[0]-period), dtype=np.float32)
	
	for i in range(period, close.shape[0]):
		gain_sum = 0.0
		loss_sum = 0.0
		if i > period:
			p_gain = gain[i-period-1]
			p_loss = loss[i-period-1]

			chng = close[i] - close[i-1]
			if chng >= 0:
				gain_sum += chng
			else:
				loss_sum += np.absolute(chng)

			gain_avg = (p_gain * (period-1) + gain_sum)/period
			loss_avg = (p_loss * (period-1) + loss_sum)/period

		else:
			for j in range(0, i):
				if j != 0:
					chng = close[j] - close[j-1]

					if chng >= 0:
						gain_sum += chng
					else:
						loss_sum += np.absolute(chng)

			gain_avg = gain_sum / period
			loss_avg = loss_sum / period

		gain[i-period] = gain_avg
		loss[i-period] = loss_avg

		if loss_avg == 0.0:
			X[i-period] = 100
		else:
			X[i-period] = 100 - (100 / (1 + gain_avg/loss_avg))
	return X

def getRsiRangeData(rsi_data, high, low, lookup, threshold=0):
	X = []
	offset = 0

	r_min = 0
	last_min = low[0]
	r_max = 0
	last_max = high[0]
	is_up = rsi_data[0] >= (50. + threshold)
	if is_up:
		c_range = [convertToPips(high[0] - low[0])]
	else:
		c_range = [abs(convertToPips(low[0] - high[0]))]

	for i in range(rsi_data.shape[0]):
		if r_max == 0 or high[i] > r_max:
			r_max = high[i]
		if r_min == 0 or low[i] < r_min:
			r_min = low[i]

		if is_up:
			if rsi_data[i] < (50. - threshold):
				if len(c_range) == 2:
					if lookup > 1:
						if len(X) > 0:
							X.append(X[-1][-lookup+1:] + [c_range])
						else:
							X.append([c_range])
					else:
						X.append(c_range)

					c_range = [abs(convertToPips(r_min - r_max))]

					if not len(X[-1]) >= lookup:
						offset += 1
					continue
				else:
					c_range.append(abs(convertToPips(r_min - r_max)))
				is_up = False
				last_max = r_max
				r_max = high[i]
			else:
				if convertToPips(r_max - r_min) > c_range[-1]:
					c_range[-1] = convertToPips(r_max - last_min)
		else:
			if rsi_data[i] > (50. + threshold):
				if len(c_range) == 2:
					if lookup > 1:
						if len(X) > 0:
							X.append(X[-1][-lookup+1:] + [c_range])
						else:
							X.append([c_range])
					else:
						X.append(c_range)

					c_range = [convertToPips(r_max - r_min)]

					if not len(X[-1]) >= lookup:
						offset += 1
					continue
				else:
					c_range.append(convertToPips(r_max - r_min))
				is_up = True
				last_min = r_min
				r_min = low[i]
			else:
				if abs(convertToPips(r_min - r_max)) > c_range[-1]:
					c_range[-1] = abs(convertToPips(r_min - last_max))

		if len(X) > 0:
			X.append(X[-1])
			if not len(X[-1]) >= lookup:
				offset += 1

	return np.array(X[offset:])

rsi_period = 10
timer = timeit()
rsi_data = getRsiData(
	df.values[:,2],
	rsi_period
)
timer.end()

print('Rsi Data:\n%s'%rsi_data[-5:])
print(rsi_data.shape)

timer = timeit()
train_data_two = getRsiRangeData(
	rsi_data, df.values[rsi_period:,0], df.values[rsi_period:,1],
	1, threshold=0
)
timer.end()

print('Rsi Range Data:\n%s'%train_data_two[-5:])
print(train_data_two.shape)

data_size = min(train_data_one.shape[0], train_data_two.shape[0])

train_size = int(data_size * 0.7)
period_off = max(donch_period+1, rsi_period)

X1_off = train_data_one.shape[0] - data_size
X1_train = train_data_one[X1_off:train_size+X1_off]

X2_off = train_data_two.shape[0] - data_size
X2_train = train_data_two[X2_off:train_size+X2_off]

y_off = df.shape[0] - data_size
y_train = df.values[y_off:train_size+y_off]

X1_val = train_data_one[train_size+X1_off:]
X2_val = train_data_two[train_size+X2_off:]
y_val = df.values[train_size+y_off:]

print('Train One Data: {} {}'.format(X1_train.shape, y_train.shape))
print('Train Two Data: {} {}'.format(X2_train.shape, y_train.shape))
print('Val One Data: {} {}'.format(X1_val.shape, y_val.shape))
print('Val Two Data: {} {}'.format(X2_val.shape, y_val.shape))

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

	def __call__(self, inpt1, inpt2):
		x = np.matmul(inpt1, self.W1[0])
		x = tf.nn.relu(x)
		for i in range(1, len(self.W1)):
			x = tf.nn.relu(x)		
			x = np.matmul(x, self.W1[i]) + self.b1[i]

		y = np.matmul(inpt2, self.W2[0])
		y = tf.nn.relu(y)
		for i in range(1, len(self.W2)):
			y = tf.nn.relu(y)		
			y = np.matmul(y, self.W2[i]) + self.b2[i]

		return [tf.nn.sigmoid(x).numpy(), np.clip(y, 25, 250)]

class GeneticPlanModel(GA.GeneticAlgorithmModel):
	def __init__(self, threshold=0.5):
		super().__init__()
		self.threshold = threshold

	def __call__(self, X, y, training=False):
		super().__call__(X, y)
		out = self._model(X[0], X[1])
		results = GeneticPlanModel.run(out[0], out[1], y, self.threshold)
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
		return BasicDenseModel(2, [16, 16, 2], 2, [16, 1])

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
	def run(out1, out2, y, threshold):
		pos_open = 0
		pos_dir = 0
		pos_sl = 0.0
		result = 0.0

		max_ret = 0.0
		dd = 0.0
		gain = 0.0
		loss = 0.0

		for i in range(out1.shape[0]):

			if pos_open != 0:
				if pos_dir == 1:
					if convertToPips(y[i][1] - pos_open) <= -pos_sl:
						loss += 1.0
						result -= 1.0
						pos_open = 0

					# elif convertToPips(y[i][0] - pos_open) >= tp:
					# 	result = result + tp
					# 	pos_open = 0
				else:
					if convertToPips(pos_open - y[i][0]) <= -pos_sl:
						loss += 1.0
						result -= 1.0
						pos_open = 0

					# elif convertToPips(pos_open - y[i][1]) >= tp:
					# 	result = result + tp
					# 	pos_open = 0

			if out1[i][1] > threshold:
				if pos_open == 0 or pos_dir != 1:
					if pos_open != 0:
						ret = convertToPips(pos_open - y[i][2])/pos_sl
						if ret >= 0:
							gain += ret
						else:
							loss += abs(ret)
						result += ret
					pos_open = y[i][2]
					pos_dir = 1
					pos_sl = out2[i][0]
			if out1[i][0] < threshold:
				if pos_open == 0 or pos_dir != 0:
					if pos_open != 0:
						ret = convertToPips(y[i][2] - pos_open)/pos_sl
						if ret >= 0:
							gain += ret
						else:
							loss += abs(ret)
						result += ret
					pos_open = y[i][2]
					pos_dir = 0
					pos_sl = out2[i][0]

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
print(gpm([X1_train, X2_train], y_train))

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
	train_data=([X1_train, X2_train], y_train),
	val_data=([X1_val, X2_val], y_val),
	generations=10
)













