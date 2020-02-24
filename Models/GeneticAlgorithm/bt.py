from numba import jit
import numpy as np

BUY = 1
SELL = -1

@jit
def convertToPips(x):
	return np.around(x * 10000, 2)

@jit
def create_position(positions, _open, direction, sl, tp)
	for i in range(positions):
		if positions[i][0] == 0:
			positions[i][0] = _open
			positions[i][1] = direction
			positions[i][2] = sl
			positions[i][3] = tp
	return positions

@jit
def del_position(positions, i)
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
	for i in range(positions.shape[0]):
		if positions[i][0] == 0:
			return positions, result
		else:
			positions, result = close_position(positions, i, ohlc, result)

	return positions, result

@jit
def stop_and_reverse(positions, ohlc, result, _open, direction, sl, tp):
	positions, result = close_all(positions, ohlc, result)
	positions = create_position(positions, _open, direction, sl, tp)
	return positions, result

@jit
def reset_positions(positions):
	return np.zeros((10,4), dtype=np.float32)

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
def check_sl(positions, ohlc, result):
	i = 0
	while i < positions.shape[0]:
		pos = positions[i]
		if pos[0] == 0:
			return positions, result
		elif pos[0] == BUY:
			if convertToPips(ohlc[2] - pos[1]) <= pos[2]:
				result += -1.0
				positions = del_position(positions, i)
				continue
		elif pos[0] == SELL:
			if convertToPips(pos[1] - ohlc[1]) <= pos[2]:
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
def get_profit(positions, i):
	pos = positions[i]
	if pos[0] == BUY:
		return convertToPips(ohlc[3] - pos[1])
	elif pos[0] == SELL:
		return convertToPips(pos[1] - ohlc[3])

@jit
def get_total_profit(positions):
	profit = 0.0
	for i in range(positions.shape[0]):
		if positions[i][0] == 0:
			return profit
		profit += get_profit(positions, i)

	return profit

@jit
def get_direction(positions, i):
	return positions[i][0]

@jit
def start(runloop, ohlc, *args):
	positions = np.zeros((10,4), dtype=np.float32)
	data = np.zeros((10,), dtype=np.float32)
	result = np.zeros((1,), dtype=np.float32)

	for i in range(ohlc.shape[0]):
		positions, result = check_sl(positions, ohlc, result)
		positions, result = check_tp(positions, ohlc, result)

		positions, result, data = runloop(i, positions, result, data, *args)

	return result, data

















