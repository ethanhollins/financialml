import sys
sys.path.append('../')

from DataLoader import DataLoader
from numba import jit
import GA
import Constants
import numpy as np
import cupy as cp
import datetime as dt
import time
import pandas as pd
import bt
import json

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

start = dt.datetime(2015,1,1)
end = dt.datetime(2020,1,1)

num_months = round((end - start).days / 30.0)
val_months = 12
data_split = round((num_months-val_months) / num_months, 2)

big_period = Constants.FOUR_HOURS
small_period = Constants.TEN_MINUTES

df_big = dl.get(Constants.GBPUSD, big_period, start=start, end=end)
df_small = dl.get(Constants.GBPUSD, small_period, start=start, end=end)
df_small = df_small[['bid_open', 'bid_high', 'bid_low', 'bid_close']]

# Add Offset
df_big.index = df_big.index + dl.getPeriodOffsetSeconds(big_period)
df_small.index = df_small.index + dl.getPeriodOffsetSeconds(small_period)

'''
Feature Engineering
'''

# Convert price to pips
@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

def normalize(x, mean, std):
	return (x - mean) / std

@jit(forceobj=True)
def getTrainData(data_big, data_small, timestamps, lookup):
	X_out = []
	X_plan = []

	pad_size = int(dl.getPeriodOffsetSeconds(big_period) / dl.getPeriodOffsetSeconds(small_period)) * lookup
	c_data = np.zeros((2,pad_size,3), dtype=np.float32)
	c_plan = np.zeros((2,))

	step = 1
	pred = 3

	for t_i in range(lookup, timestamps.size-pred, step):
		data_i = data_small[(data_small.index >= timestamps[t_i-lookup]) & (data_small.index < timestamps[t_i])].values

		for d_i in range(1, data_i.shape[0]):
			size = convertToPips(data_i[d_i,3] - data_i[d_i-1,3])
			wick_up = convertToPips(data_i[d_i,1] - data_i[d_i-1,1])
			wick_down = convertToPips(data_i[d_i,2] - data_i[d_i-1,2])

			c_data = np.copy(c_data)

			c_data[:,(d_i-1),0] = size
			c_data[:,(d_i-1),1] = wick_up
			c_data[:,(d_i-1),2] = wick_down

		c_data[1] *= -1
		X_out.append(c_data)

		c_plan = np.copy(c_plan)

		c_plan[0] = convertToPips(data_big[t_i+1:t_i+pred+1,1].max() - data_big[t_i,3])
		c_plan[1] = convertToPips(data_big[t_i,3] - data_big[t_i+1:t_i+pred+1,2].min())

		X_plan.append(c_plan)

	return X_out, X_plan

timer = timeit()

lookup = 3
train_data, plan_data = getTrainData(
	df_big.values[:,4:], df_small,
	df_big.index.values,
	lookup
)
train_data = np.array(train_data, dtype=np.float32)
plan_data = np.array(plan_data, dtype=np.float32)
timer.end()

print('Train Data:\n%s\n' % train_data[:5])
print(train_data.shape)
print('Plan Data:\n%s\n' % plan_data[:5])

train_size = int(train_data.shape[0] * data_split)

period_off = df_big.shape[0] - train_data.shape[0]
train_start_ts = df_big.index[period_off]
val_start_ts = df_big.index[period_off+train_size]

X_train = train_data[:train_size]
X_train_plan = plan_data[:train_size]
X_val = train_data[train_size:]
X_val_plan = plan_data[train_size:]

mean = np.mean(X_train)
std = np.std(X_train)
X_train_norm = normalize(X_train, mean, std)
X_val_norm = normalize(X_val, mean, std)

'''
Charts processing
'''
# TODO: FIX for one minute chart only

@jit
def processChart(i, charts, timestamps, chart, chart_ts):
	for j in range(chart_ts.shape[0]):
		ts = chart_ts[j]
		idx = (np.abs(timestamps - ts)).argmin()
		charts[i][idx] = chart[j]
	return charts

# Train y data

y_train_big = df_big[period_off:period_off+train_size].values
y_train_big_ts = df_big[period_off:period_off+train_size].index.values

y_train_ts = np.sort(np.unique(np.concatenate(
	(y_train_big_ts,)
)))

unprocessed_charts = [y_train_big]
unprocessed_ts = [y_train_big_ts]
y_train = np.zeros((len(unprocessed_charts), y_train_ts.shape[0], y_train_big.shape[1]))

timer = timeit()
for i in range(len(unprocessed_charts)):
	chart = unprocessed_charts[i]
	ts = unprocessed_ts[i]
	y_train = processChart(i, y_train, y_train_ts, chart, ts)
timer.end()

print('\nTrain Chart Data:\n%s'%y_train[:,:5])

# Validation y data

y_val_big = df_big[period_off+train_size:].values
y_val_big_ts = df_big[period_off+train_size:].index.values


y_val_ts = np.sort(np.unique(np.concatenate(
	(y_val_big_ts,)
)))

unprocessed_charts = [y_val_big]
unprocessed_ts = [y_val_big_ts]
y_val = np.zeros((len(unprocessed_charts), y_val_ts.shape[0], y_val_big.shape[1]))

timer = timeit()
for i in range(len(unprocessed_charts)):
	chart = unprocessed_charts[i]
	ts = unprocessed_ts[i]
	y_val = processChart(i, y_val, y_val_ts, chart, ts)
timer.end()

# Visualize data
print('\nVal Chart Data:\n%s'%y_val[:,:5])

num_train_big = np.unique(np.nonzero(y_train[0])[0]).shape
num_val_big = np.unique(np.nonzero(y_val[0])[0]).shape

print('\nTrain Data: {} {} H4: {}'.format(X_train.shape, y_train.shape, num_train_big))
print('Train Plan Data: {}\n{}'.format(X_train_plan.shape, X_train_plan[:5]))
print('\nVal Data: {} {} H4: {}'.format(X_val.shape, y_val.shape, num_val_big))
print('Val Plan Data: {}\n{}'.format(X_val_plan.shape, X_val_plan[:5]))

'''
Create Genetic Model
'''
class GeneticPlanModel(GA.GeneticAlgorithmModel):
	def __init__(self, X_train_plan, X_val_plan, threshold=0.5):
		super().__init__()
		self.X_train_plan = X_train_plan
		self.X_val_plan = X_val_plan
		self.threshold = threshold

	def __call__(self, X, y, training=False):
		super().__call__(cp.asnumpy(X), y)
		
		out = self.getOutput(X)
		
		if training:
			_, data = bt.start(
				GeneticPlanModel.run, y.astype(np.float32), self.threshold,
				out, self.X_train_plan
			)
			self.train_results = data[:2]
		else:
			_, data = bt.start(
				GeneticPlanModel.run, y.astype(np.float32), self.threshold,
				out, self.X_val_plan
			)
			
			self.val_results = data[:2]

		return self.getPerformance(*data[:2], training=training)

	def getOutput(self, X):
		# Process model output
		
		x = self._model[0](X)
		x = self._model[1](x)
		x = self._model[2](x)
		x = self._model[3](X)
		x = self._model[4](x)

		x = self._model[5](x)
		x = cp.asnumpy(x)
		x = bt.sigmoid(x)

		return x

	def getPerformance(self, amount, result, training=False):
		return result

	def generateModel(self, model_info):
		return [
			GA.Conv2D_GPU(kernel_shape=(3,1), row_stride=1, col_stride=1),
			GA.Conv2D_GPU(kernel_shape=(3,1), row_stride=2, col_stride=1),
			GA.MaxPooling1D_GPU(kernel_size=3, stride=1),
			GA.Conv2D_GPU(kernel_shape=(3,1), row_stride=2, col_stride=1),
			GA.MaxPooling1D_GPU(kernel_size=3, stride=1),
			GA.GRU_GPU(1, 128, 1),
		]

	def newModel(self):
		return GeneticPlanModel(X_train_plan, X_val_plan, self.threshold)

	def getWeights(self):
		return (
			self._model[0].get_weights() +
			self._model[1].get_weights() +
			self._model[3].get_weights() +
			self._model[5].get_weights()
		)

	def setWeights(self, weights):
		l1 = len(self._model[0].get_weights())
		l2 = len(self._model[1].get_weights())
		l3 = len(self._model[3].get_weights())
		l4 = len(self._model[5].get_weights())
		self._model[0].set_weights(weights[:l1])
		self._model[1].set_weights(weights[l1:(l1+l2)])
		self._model[3].set_weights(weights[(l1+l2):(l1+l2+l3)])
		self._model[5].set_weights(weights[(l1+l2+l3):(l1+l2+l3+l4)])

	# def save(self):
	# 	return {'sl': float(self.sl), 'tp_increment': float(self.tp_increment)}

	def __str__(self):
		return  '  (Train) (P) {:.2f} Amt: {:.0f}\n' \
				'    (Val) (P) {:.2f} Amt: {:.0f}\n'.format(
			self.getPerformance(*self.train_results, training=True), 
			self.train_results[0],
			self.getPerformance(*self.val_results), 
			self.val_results[0], 
		)


	@jit
	def run(i, j, positions, charts, result, data, stats, threshold, out, plan):
		data[0] += 1
		if out[i,0][0] > out[i,1][0]:
			if out[i,0][0] >= threshold:
				data[1] += plan[i,0] - plan[i,1]

		else:
			if out[i,1][0] >= threshold:
				data[1] += plan[i,1] - plan[i,0]

		return positions, result, data, stats
		
'''
Create Genetic Algorithm
'''

bt.recompile_all()
bt.data_count = 20

crossover = GA.PreserveBestCrossover(preserve_rate=0.5)
mutation = GA.PreserveBestMutation(mutation_rate=0.05, preserve_rate=0.5)
ga = GA.GeneticAlgorithm(
	crossover, mutation,
	survival_rate=0.2,
	is_gpu=True
)

ga.setSeed(1)
# ga.saveBestTrain(10, 'v1.0.3_6m_5c', {'mean': float(mean), 'std': float(std)})

def generate_models(num_models):
	models = []
	for i in range(num_models):
		models.append(GeneticPlanModel(X_train_plan, X_val_plan, threshold=0.5))
	return models

num_models = 100
ga.fit(
	models=generate_models(num_models),
	train_data=(X_train_norm, y_train),
	val_data=(X_val_norm, y_val),
	generations=300
)

