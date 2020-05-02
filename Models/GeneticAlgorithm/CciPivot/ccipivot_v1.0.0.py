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

start = dt.datetime(2019,8,1)
end = dt.datetime(2019,11,1)

num_months = round((end - start).days / 30.0)
print('Months: %s'%num_months)
val_months = 1.0
data_split = round((num_months-val_months) / num_months, 2)

df_m = dl.get(Constants.GBPUSD, Constants.ONE_MINUTE, start=start, end=end)

'''
Feature Engineering
'''

# Convert price to pips
def convertToPips(x):
	return np.around(x * 10000, 2)

def normalize(x, mean, std):
	return (x - mean) / std

@jit
def getCci(ohlc, period):
	X = []
	const = .015

	for i in range(period, ohlc.shape[0]):
		# Calculate Typical price
		c_typ = (ohlc[i,1] + ohlc[i,2] + ohlc[i,3])/3.0
		typ_sma = 0.0
		for j in range((i+1)-period, i+1):
			typ_sma += (ohlc[j,1] + ohlc[j,2] + ohlc[j,3])/3.0

		typ_sma /= period
		
		# Calculate Mean Deviation
		mean_dev = 0.0
		for j in range((i+1)-period, (i+1)):
			mean_dev += np.absolute(
				((ohlc[j,1] + ohlc[j,2] + ohlc[j,3])/3.0) - typ_sma
			)

		mean_dev /= period

		if mean_dev == 0:
			X.append(0)
		else:
			X.append((c_typ - typ_sma) / (const * mean_dev))

	return X

def getTrainData(ohlc, cci, timestamps):
	data_points = 20

	X_out = []
	X_plan = []
	c_data = np.zeros((1,data_points,1), dtype=np.float32)
	c_plan = np.zeros((5,))

	for i in range(ohlc.shape[0]):

		c_data = np.copy(c_data)
		c_plan = np.copy(c_plan)

		c_data[:,:(data_points-1)] = c_data[:,-(data_points-1):]
		c_data[0,-1,0] = cci[i]

		l_pivot_idx = -(data_points - c_data[0].argmax())
		l_swing_idx = -(data_points - c_data[0].argmin())
		c_plan[1] = ohlc[i+l_pivot_idx,1]
		c_plan[2] = ohlc[i+l_swing_idx,6]
		
		s_pivot_idx = -(data_points - c_data[0].argmin())
		s_swing_idx = -(data_points - c_data[0].argmax())
		c_plan[3] = ohlc[i+s_pivot_idx,2]
		c_plan[4] = ohlc[i+s_swing_idx,5]

		# Set open trading hours
		time = dl.convertTimestampToTime(timestamps[i])
		time = dl.convertTimezone(time, 'Europe/London')
		if 6 <= time.hour < 20:
			c_plan[0] = 1
		else:
			c_plan[0] = 0

		if i >= data_points:
			X_out.append(c_data)
			X_plan.append(c_plan)

	return np.array(X_out, dtype=np.float32), np.array(X_plan, dtype=np.float32)

# Get Cci Bid/Ask
timer = timeit()
# cci_ask = np.around(np.array(getCci(df_m.values[:,:4], 5), dtype=np.float32), decimals=5)
cci_bid = np.around(np.array(getCci(df_m.values[:,4:], 5), dtype=np.float32), decimals=5)
timer.end()
print(cci_bid.shape)

offset = df_m.values.shape[0] - cci_bid.shape[0]
timestamps = df_m.index.values[offset:]
ohlc = df_m.values[offset:]
print(ohlc.shape)

timer = timeit()
train_data, plan_data = getTrainData(
	np.round(ohlc, decimals=5),
	cci_bid,
	timestamps
)
timer.end()

print('Train Data:\n%s\n' % train_data[:5])
print(train_data.shape)
print('Plan Data:\n%s\n' % plan_data[:5])
print(plan_data.shape)

'''
Organise Data
'''

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
			results, data = bt.start(
				GeneticPlanModel.run, y.astype(np.float32), self.threshold,
				out, self.X_train_plan
			)
			results = [
				results[0], # Return
				results[4], # Drawdown
				results[1],	# Gain
				results[2],	# Loss
				results[5], # Wins
				results[6], # Losses
			]
			self.train_results = results
		else:
			results, data = bt.start(
				GeneticPlanModel.run, y.astype(np.float32), self.threshold,
				out, self.X_val_plan
			)
			results = [
				results[0], # Return
				results[4], # Drawdown
				results[1],	# Gain
				results[2],	# Loss
				results[5], # Wins
				results[6], # Losses
			]
			self.val_results = results

		return self.getPerformance(*results, training=training)

	def getOutput(self, X):
		# Process model output
		x = self._model[0](X)
		x = self._model[1](x.reshape((list(x.shape)+[1])))
		x = cp.asnumpy(x)
		x = bt.sigmoid(x)

		# t_x = np.copy(x[:,2])
		# if t_x.max() == t_x.min():
		# 	x[:,2] = (t_x - t_x.min())
		# else:
		# 	x[:,2] = (t_x - t_x.min()) / (t_x.max() - t_x.min())
		# x[:,2] = np.sum(x[:,2])/x[:,2].size
		# x[:,2] = np.round(x[:,2] * ((30.0-5.0) + 5.0))

		return x

	def getPerformance(self, ret, dd, gain, loss, wins, losses, training=False):
		if loss == 0:
			gpr = 0.0
		else:
			gpr = (gain / loss) + 1

		dd_mod = pow(max(dd-8, 0), 3)
		gpr_mod = pow(max(gpr-3,0), 2)

		num_trades = wins + losses
		if training:
			max_trades = (num_months * 100) * data_split
		else:
			max_trades = (num_months * 100) * (1 - data_split)
		max_trades_mod = (num_trades - max_trades)*2 if num_trades > max_trades else 0

		if training:
			min_trades = (num_months * 40) * data_split
		else:
			min_trades = (num_months * 40) * (1 - data_split)

		min_trades_mod = pow(min_trades - num_trades, 2) if num_trades < min_trades else 0

		if losses == 0:
			wl_mod = 1
		else:
			wl_mod = wins/losses + 1

		if ret > 0:
			return (ret*wl_mod) - dd_mod - min_trades_mod# - gpr_mod
		else:
			return ret - dd_mod - min_trades_mod# - gpr_mod


	def generateModel(self, model_info):
		return [
			GA.RNN_2D_GPU(1, 64, None),
			GA.RNN_2D_GPU(1, 64, 4),
			# GA.RNN_TWO_GPU(1, 32, 5),
		]

	def newModel(self):
		return GeneticPlanModel(X_train_plan, X_val_plan, self.threshold)

	def getWeights(self):
		return (
			self._model[0].get_weights() + 
			self._model[1].get_weights()# +
			# self._model[2].get_weights()
		)

	def setWeights(self, weights):
		l = len(self._model[0].get_weights())
		self._model[0].set_weights(weights[:l])
		self._model[1].set_weights(weights[l:l*2])
		# self._model[2].set_weights(weights[l*2:])

	# def save(self):
	# 	return {'sl': float(self.sl), 'tp_increment': float(self.tp_increment)}

	def __str__(self):
		return  ' (Train) Perf: {:.2f}  Ret: {:.2f}  DD: {:.2f}  Wins: {:.0f}  Losses: {:.0f}\n' \
				'   (Val) Perf: {:.2f}  Ret: {:.2f}  DD: {:.2f}  Wins: {:.0f}  Losses: {:.0f}\n'.format(
			self.getPerformance(*self.train_results, training=True), 
			self.train_results[0], self.train_results[1],
			self.train_results[4], self.train_results[5],
			self.getPerformance(*self.val_results), 
			self.val_results[0], self.val_results[1],
			self.val_results[4], self.val_results[5],
		)


	@jit
	def run(i, j, positions, charts, result, data, stats, threshold, out, plan):
		# Misc variables
		sl = 17.0
		risk = 0.5
		max_gain = 2.5
		max_loss = -2.0


		# OHLC values
		ohlc = charts[j][i]
		
		# Current direction		
		c_dir = bt.get_direction(positions, 0)
		num_pos = bt.get_num_positions(positions)

		'''
		Daily Returns Controls
		'''
		if plan[i][0] == 0:
			data[4] = result

		'''
		Set Pivot/Swing
		'''
		# LONG Pivot
		if out[i,0][0] >= threshold:
			data[0] = plan[i][1]
		
		# LONG Swing
		if out[i,0][1] >= threshold:
			data[1] = plan[i][2]

		# SHORT Pivot
		if out[i,0][2] >= threshold:
			# Set pivot
			data[2] = plan[i][3]

		# SHORT Swing
		if out[i,0][3] >= threshold:
			data[3] = plan[i][4]

		'''
		On Cancel
		'''
		# On cancel pivot
		if data[1] != 0 and ohlc[7] < data[1]:
			data[0] = 0
			if c_dir == bt.BUY:
				positions, result = bt.close_all(positions, ohlc, result, stats)

		if data[3] != 0 and ohlc[3] > data[3]:
			data[2] = 0
			if c_dir == bt.SELL:
				positions, result = bt.close_all(positions, ohlc, result, stats)
		
		'''
		On Entry
		'''
		# On close AB pivot (S&R)
		if plan[i][0] == 1:
			daily_profit = result - data[4]
			if daily_profit > max_loss and daily_profit < max_gain:

				if data[0] != 0 and ohlc[3] > data[0]:
					data[0] = 0
					if c_dir == bt.SELL:
						positions, result = bt.stop_and_reverse(positions, charts[j][i], result, stats, bt.BUY, sl, 0, sl/risk)
					elif c_dir == 0:
						positions = bt.create_position(positions, charts[j][i], bt.BUY, sl, 0, sl/risk)

				elif data[2] != 0 and ohlc[7] < data[2]:
					data[2] = 0
					if c_dir == bt.BUY:
						positions, result = bt.stop_and_reverse(positions, charts[j][i], result, stats, bt.SELL, sl, 0, sl/risk)
					elif c_dir == 0:
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
	generations=1000
)

