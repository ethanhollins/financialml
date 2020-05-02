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

start = dt.datetime(2005,1,1)
end = dt.datetime(2020,1,1)

num_months = round((end - start).days / 30.0)
val_months = 12
data_split = round((num_months-val_months) / num_months, 2)

df_m = dl.get(Constants.GBPUSD, Constants.DAILY, start=start, end=end)

'''
Feature Engineering
'''

# Convert price to pips
def convertToPips(x):
	return np.around(x * 10000, 2)

def normalize(x, mean, std):
	return (x - mean) / std

def getTrainData(data, timestamps):
	data_points = 120
	pivot_len = 5

	X_out = []
	X_plan = []
	c_data = np.zeros((1,data_points,3), dtype=np.float32)
	c_plan = np.zeros((3,))

	last_hl = np.zeros((pivot_len,2), dtype=np.float32)

	for i in range(1,data.shape[0]):
		_open, high, low, close = data[i]
		last_hl[:(pivot_len-1)] = last_hl[-(pivot_len-1):]
		last_hl[-1,0] = high
		last_hl[-1,1] = low

		size = convertToPips(data[i,3] - data[i-1,3])
		wick_up = convertToPips(data[i,1] - data[i-1,1])
		wick_down = convertToPips(data[i,2] - data[i-1,2])

		c_data = np.copy(c_data)
		c_plan = np.copy(c_plan)

		time = dl.convertTimestampToTime(timestamps[i])
		time = dl.convertTimezone(time, 'Europe/London')

		if 6 <= time.hour < 20:
			c_plan[0] = 1
		else:
			c_plan[0] = 0
		c_plan[1] = np.amax(last_hl[:,0])
		c_plan[2] = np.amin(last_hl[:,1])

		c_data[:,:(data_points-1)] = c_data[:,-(data_points-1):]
		c_data[0,-1,0] = size
		c_data[0,-1,1] = wick_up
		c_data[0,-1,2] = wick_down

		if i >= data_points:
			X_out.append(c_data)
			X_plan.append(c_plan)

		elif len(X_out) > 0:
			X_out.append(np.zeros((data_points,3), dtype=np.float32))
			X_plan.append(c_plan)

	return X_out, X_plan

timer = timeit()
train_data, plan_data = getTrainData(
	np.round(df_m.values[:,4:], decimals=5),
	df_m.index.values
)
train_data = np.array(train_data, dtype=np.float32)
plan_data = np.array(plan_data, dtype=np.float32)
timer.end()
print('Train Data:\n%s\n' % train_data[:5])
print(train_data.shape)
print('Plan Data:\n%s\n' % plan_data[:5])

train_size = int(train_data.shape[0] * data_split)

period_off = df_m.shape[0] - train_data.shape[0]
train_start_ts = df_m.index[period_off]
val_start_ts = df_m.index[period_off+train_size]

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

y_train = df_m[period_off:period_off+train_size].values
y_train_ts = df_m[period_off:period_off+train_size].index.values

unprocessed_charts = [y_train]
unprocessed_ts = [y_train_ts]
y_train = np.zeros((len(unprocessed_charts), y_train_ts.shape[0], y_train.shape[1]))

timer = timeit()
for i in range(len(unprocessed_charts)):
	chart = unprocessed_charts[i]
	ts = unprocessed_ts[i]
	y_train = processChart(i, y_train, y_train_ts, chart, ts)
timer.end()

print('\nTrain Chart Data:\n%s'%y_train[:,:5])

# Validation y data

y_val = df_m[period_off+train_size:].values
y_val_ts = df_m[period_off+train_size:].index.values

unprocessed_charts = [y_val]
unprocessed_ts = [y_val_ts]
y_val = np.zeros((len(unprocessed_charts), y_val_ts.shape[0], y_val.shape[1]))

timer = timeit()
for i in range(len(unprocessed_charts)):
	chart = unprocessed_charts[i]
	ts = unprocessed_ts[i]
	y_val = processChart(i, y_val, y_val_ts, chart, ts)
timer.end()

# Visualize data
print('\nVal Chart Data:\n%s'%y_val[:,:5])

print('\nTrain Data: {} {}'.format(X_train.shape, y_train.shape))
print('Train Plan Data: {}\n{}'.format(X_train_plan.shape, X_train_plan[:5]))
print('\nVal Data: {} {}'.format(X_val.shape, y_val.shape))
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
			self.sl = min(max(out[0,0][2], 50.0), 200.0)
			self.tp = min(max(out[0,0][3], 30.0), 200.0)
			results, data = bt.start(
				GeneticPlanModel.run, y.astype(np.float32), self.threshold,
				out, self.X_train_plan, self.sl, self.tp
			)
			results = [
				results[0], # Return
				results[4], # Drawdown
				results[1],	# Gain
				results[2],	# Loss
				results[5], # Wins
				results[6], # losses
				data[11], # Scaled Result
				data[13], # Scaled DD
			]
			self.train_results = results
		else:
			results, data = bt.start(
				GeneticPlanModel.run, y.astype(np.float32), self.threshold,
				out, self.X_val_plan, self.sl, self.tp
			)
			results = [
				results[0], # Return
				results[4], # Drawdown
				results[1],	# Gain
				results[2],	# Loss
				results[5], # Wins
				results[6], # Losses
				data[11], # Scaled Result
				data[13], # Scaled DD
			]
			self.val_results = results

		return self.getPerformance(*results, training=training)

	def getOutput(self, X):
		# Process model output
		
		# Positive
		x = self._model[0](X)
		# x_pos = bt.relu_gpu(x_pos)
		# x_pos = self._model[1](x_pos)

		# # Negative
		# x_neg = self._model[0](-X)
		# x_neg = bt.relu_gpu(x_neg)
		# x_neg = self._model[1](x_neg)

		# x = cp.concatenate((x_pos, x_neg), axis=3)

		# Softmax
		# for i in range(x.shape[3]):
		# 	x[:,0,:,i] = cp.exp(x)[:,0,:,i] / cp.sum(cp.exp(x), axis=(2,3))

		x = self._model[2](x)
		# x = self._model[3](x.reshape(list(x.shape)+[1]))
		x = cp.asnumpy(x)
		x[:,:,:2] = bt.sigmoid(x[:,:,:2])

		t_x = np.copy(x[:,:,2])
		if t_x.max() == t_x.min():
			x[:,:,2] = (t_x - t_x.min())
		else:
			x[:,:,2] = (t_x - t_x.min()) / (t_x.max() - t_x.min())
		x[:,:,2] = np.sum(x[:,:,2])/x[:,:,2].size
		x[:,:,2] = np.round(x[:,:,2] * ((200.0-50.0) + 50.0))

		t_x = np.copy(x[:,:,3])
		if t_x.max() == t_x.min():
			x[:,:,3] = (t_x - t_x.min())
		else:
			x[:,:,3] = (t_x - t_x.min()) / (t_x.max() - t_x.min())
		x[:,:,3] = np.sum(x[:,:,3])/x[:,:,3].size
		x[:,:,3] = np.round(x[:,:,3] * ((200.0-30.0) + 30.0), decimals=2)

		return x

	def getPerformance(self, ret, dd, gain, loss, wins, losses, s_ret, s_dd, training=False):

		num_trades = wins + losses
		if training:
			min_trades = (num_months * 8) * data_split
		else:
			min_trades = (num_months * 8) * (1 - data_split)

		min_trades_mod = min_trades - num_trades if num_trades < min_trades else 0

		if training:
			ret_mod = s_ret
			dd_mod = pow(max(s_dd-8, 0), 3)
		else:
			ret_mod = ret
			dd_mod = pow(max(dd-8, 0), 3)

		return ret_mod - dd_mod# - min_trades_mod

	def generateModel(self, model_info):
		return [
			GA.Conv2D_GPU(kernel_shape=(3,3), row_stride=3, col_stride=1),
			GA.MaxPooling1D_GPU(kernel_size=2, stride=2),
			GA.GRU_GPU(1, 128, 4),
			# GA.GRU_GPU(1, 16, 4),
		]

	def newModel(self):
		return GeneticPlanModel(X_train_plan, X_val_plan, self.threshold)

	def getWeights(self):
		return (
			self._model[0].get_weights() +
			self._model[2].get_weights()
			# self._model[3].get_weights()
		)

	def setWeights(self, weights):
		l1 = len(self._model[0].get_weights())
		l2 = len(self._model[2].get_weights())
		# l3 = len(self._model[3].get_weights())
		self._model[0].set_weights(weights[:l1])
		self._model[2].set_weights(weights[l1:(l1+l2)])
		# self._model[3].set_weights(weights[(l1+l2):(l1+l2+l3)])

	# def save(self):
	# 	return {'sl': float(self.sl), 'tp_increment': float(self.tp_increment)}

	def __str__(self):
		return  '  (Train) (P) {:.2f}\tRet: {:.2f}  DD: {:.2f}  Wins: {:.0f}  Losses: {:.0f}\n' \
				'    (Val) (P) {:.2f}\tRet: {:.2f}  DD: {:.2f}  Wins: {:.0f}  Losses: {:.0f}\n' \
				' (Scaled)\t\tRet: {:.2f}  DD: {:.2f}\n'.format(
			self.getPerformance(*self.train_results, training=True), 
			self.train_results[0], self.train_results[1],
			self.train_results[4], self.train_results[5],
			self.getPerformance(*self.val_results), 
			self.val_results[0], self.val_results[1],
			self.val_results[4], self.val_results[5],
			self.train_results[6], self.train_results[7]
		)


	@jit
	def run(i, j, positions, charts, result, data, stats, threshold, out, plan, sl, tp):
		# Performance Measuring
		multi = pow(i / charts.shape[1], 0)
			
		# Result
		if stats[0] != data[10]:
			data[11] += (stats[0] - data[10]) * multi
		
		# DD
		if stats[4] != data[12]:
			data[13] += (stats[4] - data[12]) * multi


		# Last result
		data[10] = stats[0]
		# Last DD
		data[12] = stats[4]


		# Misc variables
		risk = 1.0
		max_trades = 999

		# OHLC values
		high = charts[j][i][5]
		low = charts[j][i][6]
		close = charts[j][i][7]
		
		# Current direction		
		c_dir = bt.get_direction(positions, 0)
		num_pos = bt.get_num_positions(positions)

		# TP Increment (Trailing stop)
		for x in range(num_pos):
			entry = positions[x][1]
			pos_sl = bt.get_sl(positions, x)
			direction = bt.get_direction(positions, x)

			if direction == bt.BUY:
				profit = bt.convertToPips(high - entry)
				profit_multi = profit/tp

				if profit_multi >= 2.0:
					sl_pips = (np.floor(profit_multi)-1) * tp
					if sl_pips > -pos_sl:
						positions = bt.modify_sl(positions, x, charts[j][i], -sl_pips)

			else:
				profit = bt.convertToPips(entry - low)
				profit_multi = profit/tp

				if profit_multi >= 2.0:
					sl_pips = (np.floor(profit_multi)-1) * tp
					if sl_pips > -pos_sl:
						positions = bt.modify_sl(positions, x, charts[j][i], -sl_pips)
		
		# Set current swing
		data[2] = low if low < data[2] else data[2]
		data[5] = high if high > data[5] else data[5]

		# On set pivot
		if out[i,0][0] > out[i,0][1]:
			# LONG
			if out[i,0][0] >= threshold:
				# Set pivot
				data[0] = plan[i][1]
				# Set swing
				data[1] = data[2]
				# Reset Current swing
				data[2] = low
		else:
			# SHORT
			if out[i,0][1] >= threshold:
				# Set pivot
				data[3] = plan[i][2]
				# Set swing
				data[4] = data[5]
				# Reset Current swing
				data[5] = high

		# On cancel pivot
		if data[1] != 0 and close < data[1]:
			data[0] = 0
		if data[4] != 0 and close > data[4]:
			data[3] = 0
		
		# On close AB pivot (S&R)
		if data[0] != 0 and close > data[0]:
			data[0] = 0
			if num_pos < max_trades:
				positions = bt.create_position(positions, charts[j][i], bt.BUY, sl, 0, sl/risk)

		elif data[3] != 0 and close < data[3]:
			data[3] = 0
			if num_pos < max_trades:
				positions = bt.create_position(positions, charts[j][i], bt.SELL, sl, 0, sl/risk)

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

