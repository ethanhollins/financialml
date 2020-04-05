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

start = dt.datetime(2018,1,1)
end = dt.datetime(2019,1,1)

data_split = 0.7
num_years = end.year - start.year

df_h4 = dl.get(Constants.GBPUSD, Constants.FOUR_HOURS, start=start, end=end)
df_m = dl.get(Constants.GBPUSD, Constants.FIVE_MINUTES, start=start, end=end)

# Add Offset
df_h4.index = df_h4.index + dl.getPeriodOffsetSeconds(Constants.FOUR_HOURS)
df_m.index = df_m.index + dl.getPeriodOffsetSeconds(Constants.FIVE_MINUTES)

'''
Feature Engineering
'''

# Convert price to pips
@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

def normalize(x, mean, std):
	return (x - mean) / std

@jit
def isLongSpike(data, threshold):
	s_idx = int((data.shape[0]-1)/2)
	s_high = convertToPips(data[s_idx, 1]) - threshold

	for j in range(data.shape[0]):
		if j == s_idx:
			continue
		c_high = convertToPips(data[j, 1])
		if c_high > s_high:
			return False
	return True

@jit
def isShortSpike(data, threshold):
	s_idx = int((data.shape[0]-1)/2)
	s_low = convertToPips(data[s_idx, 2]) + threshold

	for j in range(data.shape[0]):
		if j == s_idx:
			continue
		c_low = convertToPips(data[j, 2])
		if c_low < s_low:
			return False
	return True

# @jit
def getTrainData(data, timestamps):
	spike_threshold = 1.0
	lookback = 3
	swing_count = 2
	out_data = np.zeros(swing_count, dtype=np.float32) # LONG DIST, SHORT DIST
	plan_data = np.array([0,0,0], dtype=np.float32)

	# Spike LONG, Swing LONG, Spike SHORT, Swing SHORT, Current High, Current Low
	ad_data = [0,0,0,0, max(data[:lookback,1]), min(data[:lookback,2])] 
	X_out = []
	X_plan = []

	for i in range(lookback, data.shape[0]):
		out_data = np.copy(out_data)
		plan_data = np.copy(plan_data)

		# if plan_data[0]:
		# 	if data[i, 3] < ad_data[1]:
		# 		plan_data[0] = 0
		# if plan_data[1]:
		# 	if data[i, 3] > ad_data[3]:
		# 		plan_data[1] = 0

		# Get Current Spike Info
		if isLongSpike(
			data[i+1-lookback:i+1],
			spike_threshold
		):
			s_idx = int((lookback-1)/2)
			

			ad_data[0] = data[i-s_idx, 1]
			ad_data[1] = ad_data[5]

			ad_data[5] = min(data[i-s_idx, 2], data[i, 2])

			out_data[:(swing_count-1)] = out_data[-(swing_count-1):]
			out_data[-1] = convertToPips(ad_data[0] - ad_data[1])
			plan_data[0] = ad_data[0]

		if isShortSpike(
			data[i+1-lookback:i+1],
			spike_threshold
		):
			s_idx = int((lookback-1)/2)
			ad_data[2] = data[i-s_idx, 2]
			ad_data[3] = ad_data[4]

			ad_data[4] = max(data[i-s_idx, 1], data[i, 1])

			body = convertToPips(data[i-s_idx, 0] - data[i-s_idx, 3])

			if body >= 0:
				wick = convertToPips(data[i-s_idx, 3] - data[i-s_idx, 2])
			else:
				wick = convertToPips(data[i-s_idx, 0] - data[i-s_idx, 2])

			out_data[:(swing_count-1)] = out_data[-(swing_count-1):]
			out_data[-1] = convertToPips(ad_data[2] - ad_data[3])
			plan_data[1] = ad_data[2]

		ad_data[4] = data[i,1] if data[i,1] > ad_data[4] else ad_data[4]
		ad_data[5] = data[i,2] if data[i,2] < ad_data[5] else ad_data[5]

		time = dl.convertTimestampToTime(timestamps[i])
		time = dl.convertTimezone(time, 'Europe/London')
		if time.weekday() == 4 and time.hour >= 16:
			plan_data[2] = 0
		else:
			plan_data[2] = 1

		X_out.append(out_data)
		X_plan.append(plan_data)

	return X_out, X_plan

timer = timeit()
train_data, plan_data = getTrainData(
	np.round(df_h4.values[:,4:], decimals=5),
	df_h4.index.values
)
train_data = np.array(train_data, dtype=np.float32)
plan_data = np.array(plan_data, dtype=np.float32)
timer.end()
print('Train Data:\n%s\n' % train_data[:5])
print(train_data.shape)

train_size = int(train_data.shape[0] * data_split)

period_off = 3
train_start_ts = df_h4.index[period_off]
val_start_ts = df_h4.index[period_off+train_size]

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

y_train_h4 = df_h4[period_off:period_off+train_size].values
y_train_h4_ts = df_h4[period_off:period_off+train_size].index.values
y_train_m = df_m.loc[(df_m.index >= train_start_ts) & (df_m.index < val_start_ts)].values
y_train_ts_m = df_m.loc[(df_m.index >= train_start_ts) & (df_m.index < val_start_ts)].index.values

y_train_ts = np.sort(np.unique(np.concatenate(
	(y_train_h4_ts, y_train_ts_m)
)))

unprocessed_charts = [y_train_h4, y_train_m]
unprocessed_ts = [y_train_h4_ts, y_train_ts_m]
y_train = np.zeros((len(unprocessed_charts), y_train_ts.shape[0], y_train_m.shape[1]))

timer = timeit()
for i in range(len(unprocessed_charts)):
	chart = unprocessed_charts[i]
	ts = unprocessed_ts[i]
	y_train = processChart(i, y_train, y_train_ts, chart, ts)
timer.end()

print('\nTrain Chart Data:\n%s'%y_train[:,:5])

# Check 4 Hour candles are correctly ordered
# idx = np.unique(np.nonzero(y_train[0])[0])
# print(y_train[0, idx][:5,4:])

# Validation y data

y_val_h4 = df_h4[period_off+train_size:].values
y_val_h4_ts = df_h4[period_off+train_size:].index.values

y_val_m = df_m.loc[df_m.index >= val_start_ts].values
y_val_ts_m = df_m.loc[df_m.index >= val_start_ts].index.values

y_val_ts = np.sort(np.unique(np.concatenate(
	(y_val_h4_ts, y_val_ts_m)
)))

unprocessed_charts = [y_val_h4, y_val_m]
unprocessed_ts = [y_val_h4_ts, y_val_ts_m]
y_val = np.zeros((len(unprocessed_charts), y_val_ts.shape[0], y_val_m.shape[1]))

timer = timeit()
for i in range(len(unprocessed_charts)):
	chart = unprocessed_charts[i]
	ts = unprocessed_ts[i]
	y_val = processChart(i, y_val, y_val_ts, chart, ts)
timer.end()

# Visualize data
print('\nVal Chart Data:\n%s'%y_val[:,:5])

num_train_h4 = np.unique(np.nonzero(y_train[0])[0]).shape
num_val_h4 = np.unique(np.nonzero(y_val[0])[0]).shape

print('\nTrain Data: {} {} H4: {}'.format(X_train.shape, y_train.shape, num_train_h4))
print('Train Plan Data: {}\n{}'.format(X_train_plan.shape, X_train_plan[:5]))
print('\nVal Data: {} {} H4: {}'.format(X_val.shape, y_val.shape, num_val_h4))
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
			results = bt.start(
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
			results = bt.start(
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
		x = self._model[1](x)
		x = cp.asnumpy(x)

		x[:,:2] = bt.sigmoid(x[:,:2])

		t_x = np.copy(x[:,2])
		if t_x.max() == t_x.min():
			x[:,2] = (t_x - t_x.min())
		else:
			x[:,2] = (t_x - t_x.min()) / (t_x.max() - t_x.min())
		x[:,2] = np.round(x[:,2] * ((250-55) + 55))

		t_x = np.copy(x[:,3])
		if t_x.max() == t_x.min():
			x[:,3] = (t_x - t_x.min())
		else:
			x[:,3] = (t_x - t_x.min()) / (t_x.max() - t_x.min())
		x[:,3] = np.round(x[:,3] * ((200-25) + 25))

		return x

	def getPerformance(self, ret, dd, gain, loss, wins, losses, training=False):
		if loss == 0:
			gpr = 0.0
		else:
			gpr = (gain / loss) + 1

		# if dd < 2:
		# 	dd_mod = pow(dd, 2)
		# elif dd > 4:
		# 	dd_mod = pow(dd-4, 2)
		# else:
		# 	dd_mod = 0
		dd_mod = pow(max(dd-3, 0), 2)
		gpr_mod = pow(max(gpr-5,0), 2)

		if training:
			min_trades = (num_years * 48) * data_split
		else:
			min_trades = (num_years * 48) * (1 - data_split)

		num_trades = wins + losses
		trades_mod = pow(min_trades - num_trades, 2) if num_trades < min_trades else 0

		return ret - dd_mod - gpr_mod - trades_mod

	def generateModel(self, model_info):
		return [
			GA.RNN_1D_GPU(8, 4),
			GA.RNN_1D_GPU(8, 4),
		]

	def newModel(self):
		return GeneticPlanModel(X_train_plan, X_val_plan, self.threshold)

	def getWeights(self):
		return self._model[0].get_weights() + self._model[1].get_weights()

	def setWeights(self, weights):
		l = len(self._model[0].get_weights())
		self._model[0].set_weights(weights[:l])
		self._model[1].set_weights(weights[l:])

	# def save(self):
	# 	return {'sl': float(self.sl), 'tp_increment': float(self.tp_increment)}

	def __str__(self):
		return  ' (Train) Perf: {:.2f}\t% Ret: {:.2f}%\t% DD: {:.2f}%\tGPR: {:.2f}\tTrades: {}\n' \
				'   (Val) Perf: {:.2f}\t% Ret: {:.2f}%\t% DD: {:.2f}%\tGPR: {:.2f}\tTrades: {}\n'.format(
			self.getPerformance(*self.train_results),
			self.train_results[0], self.train_results[1],
			(self.train_results[2] / self.train_results[3]) if self.train_results[3] != 0 else 0,
			(self.train_results[4] + self.train_results[5]),
			self.getPerformance(*self.val_results),
			self.val_results[0], self.val_results[1],
			(self.val_results[2] / self.val_results[3]) if self.val_results[3] != 0 else 0,
			(self.val_results[4] + self.val_results[5])
		)

	@jit
	def run(i, j, positions, charts, result, data, stats, threshold, out, plan):
		risk = 1.0
		c_dir = bt.get_direction(positions, 0)

		# H4
		if j == 0:
			if plan[int(data[0])][0] == 0:
				data[1] = 0
			if plan[int(data[0])][1] == 0:
				data[2] = 0

			if int(data[0]) > 0 and plan[int(data[0])][2] == 1:
				
				# BUY Signal
				if plan[int(data[0])-1][0] != 0 and charts[j][i][7] > plan[int(data[0])-1][0]:
					if out[int(data[0])-1][0] >= threshold:
						if plan[int(data[0])-1][0] != data[1]:

							sl_c = np.array([60,80,120,150,200])
							sl = sl_c[np.abs(sl_c - out[int(data[0])-1][2]).argmin()]

							tp_c = np.array([40,55,80,100,150])
							tp = tp_c[np.abs(tp_c - out[int(data[0])-1][3]).argmin()]

							if c_dir == bt.SELL:
								positions, result = bt.stop_and_reverse(positions, charts[j][i], result, stats, bt.BUY, sl, 0, sl/risk)
								data[3] = tp
							elif c_dir == 0:
								positions = bt.create_position(positions, charts[j][i], bt.BUY, sl, 0, sl/risk)
								data[3] = tp
						data[1] = plan[int(data[0])-1][0]

				# SELL Signal
				elif plan[int(data[0])-1][1] != 0 and charts[j][i][7] < plan[int(data[0])-1][1]:
					if out[int(data[0])-1][1] >= threshold:
						if plan[int(data[0])-1][1] != data[2]:

							sl_c = np.array([60,80,120,150,200])
							sl = sl_c[np.abs(sl_c - out[int(data[0])-1][2]).argmin()]

							tp_c = np.array([40,55,80,100,150])
							tp = tp_c[np.abs(tp_c - out[int(data[0])-1][3]).argmin()]

							if c_dir == bt.BUY:
								positions, result = bt.stop_and_reverse(positions, charts[j][i], result, stats, bt.SELL, sl, 0, sl/risk)
								data[3] = tp
							elif c_dir == 0:
								positions = bt.create_position(positions, charts[j][i], bt.SELL, sl, 0, sl/risk)
								data[3] = tp
						data[2] = plan[int(data[0])-1][1]

			data[0] += 1

		# M5
		elif j == 1 and int(data[0]) > 0:
			if bt.get_num_positions(positions) > 0:
				tp_increment = data[3]
				entry = positions[0][1]
				pos_sl = bt.get_sl(positions, 0)
				direction = bt.get_direction(positions, 0)
	
				high = charts[j][i][5]
				low = charts[j][i][6]

				if direction == bt.BUY:
					profit = bt.convertToPips(high - entry)
					profit_multi = profit/tp_increment

					if profit_multi >= 2.0:
						sl_pips = (np.floor(profit_multi)-1) * tp_increment
						if sl_pips > -pos_sl:
							positions = bt.modify_sl(positions, 0, -sl_pips)

				else:
					profit = bt.convertToPips(entry - low)
					profit_multi = profit/tp_increment

					if profit_multi >= 2.0:
						sl_pips = (np.floor(profit_multi)-1) * tp_increment
						if sl_pips > -pos_sl:
							positions = bt.modify_sl(positions, 0, -sl_pips)

		return positions, result, data, stats
		
'''
Create Genetic Algorithm
'''

bt.recompile_all()

crossover = GA.PreserveBestCrossover(preserve_rate=0.5)
mutation = GA.PreserveBestMutation(mutation_rate=0.02, preserve_rate=0.5)
ga = GA.GeneticAlgorithm(
	crossover, mutation,
	survival_rate=0.2,
	is_gpu=True
)

# ga.setSeed(1)
# ga.saveBest(10, 'v2.0.0_010117', {'mean': float(mean), 'std': float(std)})

def generate_models(num_models):
	models = []
	for i in range(num_models):
		models.append(GeneticPlanModel(X_train_plan, X_val_plan, threshold=0.5))
	return models

num_models = 5000
ga.fit(
	models=generate_models(num_models),
	train_data=(X_train_norm, y_train),
	val_data=(X_val_norm, y_val),
	generations=500
)

