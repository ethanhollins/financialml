from numba.targets.registry import CPUDispatcher
from numba import jit
import numpy as np

'''
Backtester
'''

BUY = 1
SELL = -1

pos_count = 10
pos_params = 5

@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

@jit
def create_position(positions, ohlc, direction, sl, tp, risk):
	for i in range(positions.shape[0]):
		if positions[i][0] == 0:
			positions[i][0] = direction
			if direction == BUY:
				positions[i][1] = ohlc[3]
			elif direction == SELL:
				positions[i][1] = ohlc[7]
			positions[i][2] = sl
			positions[i][3] = tp
			positions[i][4] = risk
			return positions
	return positions

@jit
def del_position(positions, i):
	positions[i:-1] = positions[i+1:]
	positions[-1] = np.zeros((pos_params,))
	return positions

@jit
def close_position(positions, i, ohlc, result, stats):
	pos = positions[i]
	if pos[0] == BUY:
		profit = convertToPips(ohlc[7] - pos[1]) / pos[4]
		if profit >= 0:
			stats[5] += 1
		else:
			stats[6] += 1
		result += profit

	elif pos[0] == SELL:
		profit = convertToPips(pos[1] - ohlc[3]) / pos[4]
		if profit >= 0:
			stats[5] += 1
		else:
			stats[6] += 1
		result += profit

	positions = del_position(positions, i)
	return positions, result

@jit
def close_all(positions, ohlc, result, stats):
	i = 0
	while i < positions.shape[0]:
		if positions[i][0] == 0:
			return positions, result
		else:
			positions, result = close_position(positions, i, ohlc, result, stats)
			continue
		i += 1
	return positions, result

@jit
def stop_and_reverse(positions, ohlc, result, stats, direction, sl, tp, risk):
	positions, result = close_all(positions, ohlc, result, stats)
	positions = create_position(positions, ohlc, direction, sl, tp, risk)
	return positions, result

@jit
def reset_positions():
	return np.zeros((pos_count,pos_params), dtype=np.float32)

@jit
def get_direction(positions, i):
	return positions[i][0]

@jit
def get_num_positions(positions):
	for i in range(positions.shape[0]):
		pos = positions[i]
		if pos[0] == 0:
			return i
	return i

@jit
def get_sl(positions, i):
	return positions[i][2]

@jit
def modify_sl(positions, i, sl):
	positions[i][2] = sl
	return positions

@jit
def get_tp(positions, i):
	return positions[i][3]

@jit
def modify_tp(positions, i, tp):
	positions[i][3] = tp
	return positions

@jit
def get_risk(positions, i):
	return positions[i][4]

@jit
def get_profit(positions, i, ohlc):
	pos = positions[i]
	if pos[0] == BUY:
		return convertToPips(ohlc[7] - pos[1])
	elif pos[0] == SELL:
		return convertToPips(pos[1] - ohlc[3])

@jit
def get_total_profit(positions, ohlc):
	profit = 0.0
	for i in range(positions.shape[0]):
		if positions[i][0] == 0:
			return profit
		profit += get_profit(positions, i, ohlc)

	return profit

@jit
def check_sl(positions, ohlc, result, stats):
	i = 0
	while i < positions.shape[0]:
		pos = positions[i]
		if pos[0] == 0:
			return positions, result
		elif pos[2] == 0:
			i += 1
			continue
		elif pos[0] == BUY:
			if convertToPips(ohlc[6] - pos[1]) <= -pos[2]:
				profit = -(pos[2]/pos[4])
				if profit >= 0:
					stats[5] += 1
				else:
					stats[6] += 1

				result += profit
				positions = del_position(positions, i)
				continue

		elif pos[0] == SELL:
			if convertToPips(pos[1] - ohlc[1]) <= -pos[2]:
				profit = -(pos[2]/pos[4])
				if profit >= 0:
					stats[5] += 1
				else:
					stats[6] += 1

				result += profit
				positions = del_position(positions, i)
				continue
		i+=1
	return positions, result


@jit
def check_tp(positions, ohlc, result, stats):
	i = 0
	while i < positions.shape[0]:
		pos = positions[i]
		if pos[0] == 0:
			return positions, result
		elif pos[3] == 0:
			i += 1
			continue
		elif pos[0] == BUY:
			if convertToPips(ohlc[5] - pos[1]) >= pos[3]:
				profit = pos[3] / pos[4]
				if profit >= 0:
					stats[5] += 1
				else:
					stats[6] += 1

				result += profit
				positions = del_position(positions, i)
				continue
		elif pos[0] == SELL:
			if convertToPips(pos[1] - ohlc[2]) >= pos[3]:
				profit = pos[3] / pos[4]
				if profit >= 0:
					stats[5] += 1
				else:
					stats[6] += 1

				result += profit
				positions = del_position(positions, i)
				continue
		i+=1
	return positions, result

@jit
def get_stats(stats, result, prev_result):
	stats[0] = result
	# Calculate Gain/Loss
	ret = result - prev_result
	if ret >= 0:
		stats[1] += ret # Gain
	else:
		stats[2] += abs(ret) # Loss

	# Calculate Drawdown
	if result > stats[3]:
		stats[3] = result # Max Return
	elif (stats[3] - result) > stats[4]:
		stats[4] = (stats[3] - result) # Drawdown
	# 5 - wins
	# 6 - losses

	return stats

@jit
def start(runloop, charts, *args):
	positions = np.zeros((pos_count,pos_params), dtype=np.float32)
	data = np.zeros((10,), dtype=np.float32)
	stats = np.zeros((7,), dtype=np.float32)
	result = 0.0
	prev_result = 0.0

	for i in range(charts.shape[1]):
		prev_result = result

		positions, result = check_sl(positions, charts[-1][i], result, stats)
		positions, result = check_tp(positions, charts[-1][i], result, stats)

		for j in range(charts.shape[0]):
			if np.any(charts[j][i] != 0):
				positions, result, data, stats = runloop(i, j, positions, charts, result, data, stats, *args)

		stats = get_stats(stats, result, prev_result)

	return stats

def step(runloop, charts, *args):
	positions = np.zeros((pos_count,pos_params), dtype=np.float32)
	data = np.zeros((10,), dtype=np.float32)
	stats = np.zeros((7,), dtype=np.float32)
	result = 0.0
	prev_result = 0.0

	for i in range(charts.shape[1]):
		prev_result = result

		positions, result = check_sl(positions, charts[-1][i], result, stats)
		positions, result = check_tp(positions, charts[-1][i], result, stats)

		for j in range(charts.shape[0]):
			if np.any(charts[j][i] != 0):
				positions, result, data, stats = runloop(i, positions, charts[j], result, data, stats, *args)

		stats = get_stats(stats, result, prev_result)

		input('--|\n')

	return stats

'''
Activation Functions and NN functions
'''

@jit
def matmul(x, y):
	rx, cx = x.shape
	ry, cy = y.shape
	out = np.zeros((rx, cy))
	for i in range(rx):
		for j in range(cy):
			out[i,j] = np.sum(x[i] * y[:,j])
	return out

@jit
def relu(x):
	return np.maximum(0, x)

@jit
def elu(x):
	x[x<0] = np.exp(x[x<0]) - 1
	return x

@jit
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

@jit
def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

'''
Misc functions
'''

def recompile_all():
	for k in globals():
		if type(globals()[k]) == CPUDispatcher:
			globals()[k].recompile()			


