from numba.targets.registry import CPUDispatcher
from numba import jit
import numpy as np

BUY = 1
SELL = -1

pos_count = 10
pos_params = 4

@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

@jit
def create_position(positions, ohlc, direction, sl, tp):
	for i in range(positions.shape[0]):
		if positions[i][0] == 0:
			positions[i][0] = direction
			positions[i][1] = ohlc[3]
			positions[i][2] = sl
			positions[i][3] = tp
			return positions
	return positions

@jit
def del_position(positions, i):
	positions[i:-1] = positions[i+1:]
	positions[-1] = np.zeros((4,))
	return positions

@jit
def close_position(positions, i, ohlc, result):
	pos = positions[i]
	if pos[0] == BUY:
		result += convertToPips(ohlc[3] - pos[1]) / pos[2]
	elif pos[0] == SELL:
		result += convertToPips(pos[1] - ohlc[3]) / pos[2]
	positions = del_position(positions, i)
	return positions, result

@jit
def close_all(positions, ohlc, result):
	i = 0
	while i < positions.shape[0]:
		if positions[i][0] == 0:
			return positions, result
		else:
			positions, result = close_position(positions, i, ohlc, result)
			continue
		i += 1
	return positions, result

@jit
def stop_and_reverse(positions, ohlc, result, direction, sl, tp):
	positions, result = close_all(positions, ohlc, result)
	positions = create_position(positions, ohlc, direction, sl, tp)
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
def get_profit(positions, i, ohlc):
	pos = positions[i]
	if pos[0] == BUY:
		return convertToPips(ohlc[3] - pos[1])
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
def check_sl(positions, ohlc, result):
	i = 0
	while i < positions.shape[0]:
		pos = positions[i]
		if pos[0] == 0:
			return positions, result
		elif pos[2] == 0:
			i += 1
			continue
		elif pos[0] == BUY:
			if convertToPips(ohlc[2] - pos[1]) <= -pos[2]:
				result += -1.0
				positions = del_position(positions, i)
				continue
		elif pos[0] == SELL:
			if convertToPips(pos[1] - ohlc[1]) <= -pos[2]:
				result += -1.0
				positions = del_position(positions, i)
				continue
		i+=1
	return positions, result


@jit
def check_tp(positions, ohlc, result):
	i = 0
	while i < positions.shape[0]:
		pos = positions[i]
		if pos[0] == 0:
			return positions, result
		elif pos[3] == 0:
			i += 1
			continue
		elif pos[0] == BUY:
			if convertToPips(ohlc[1] - pos[1]) >= pos[3]:
				result += pos[3] / pos[2]
				positions = del_position(positions, i)
				continue
		elif pos[0] == SELL:
			if convertToPips(pos[1] - ohlc[2]) >= pos[3]:
				result += pos[3] / pos[2]
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

	return stats

@jit
def start(runloop, ohlc, *args):
	positions = np.zeros((pos_count,pos_params), dtype=np.float32)
	data = np.zeros((10,), dtype=np.float32)
	stats = np.zeros((5,), dtype=np.float32)
	result = 0.0
	prev_result = 0.0

	for i in range(ohlc.shape[0]):
		prev_result = result

		positions, result = check_sl(positions, ohlc[i], result)
		positions, result = check_tp(positions, ohlc[i], result)

		positions, result, data = runloop(i, positions, ohlc, result, data, *args)

		stats = get_stats(stats, result, prev_result)

	return stats

def recompile_all():
	for k in globals():
		if type(globals()[k]) == CPUDispatcher:
			globals()[k].recompile()			


