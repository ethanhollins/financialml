from DataLoader import DataLoader
from numba import jit
import GA
import Constants
import datetime as dt
import time
import numpy as np

class timeit(object):
	def __init__(self):
		print('\nTimer module started.')
		self.start = time.time()
	def end(self):
		print('Timer module finished: {:.2f}s'.format(time.time() - self.start))

# Retrieve data

dl = DataLoader()
df = dl.get(Constants.GBPUSD, Constants.FOUR_HOURS, start=dt.datetime(2015,1,1))
df = df[['bid_close']]	

# Feature Engineering
@jit
def convertToPips(x):
	return np.around(x * 1000, 2)

# Produce SMA train data
@jit
def getSmaDiff(data, periods, lookup):
	X = []
	for i in range(periods.max()+lookup, data.shape[0]):
		c_lookup = []
		for j in range(lookup):
			p_diff = []
			for p_i in range(len(periods)-1):
				p_x = periods[p_i]
				for p_y in periods[p_i+1:]:
					x = np.sum(data[i-j+1-p_x:i-j+1])/p_x
					y = np.sum(data[i-j+1-p_y:i-j+1])/p_y
					p_diff.append(convertToPips(x) - convertToPips(y))
			c_lookup.append(p_diff)
		X.append(c_lookup)
	return np.array(X)

# Generate training data
def generateTrainData(data):
	return getSmaDiff(data, np.array([20,50,100]), 5), data[:,0]

def normalize(x):
	return (x - np.mean(x)) / np.std(x)

timer = timeit()
X, y = generateTrainData(df.values)
X = normalize(X)

# Visualize train data
print('X: {}'.format(X[-5:]))
print('y: {}'.format(y[-5:]))
timer.end()

train_size = int(0.7 * X.shape[0])

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:]
y_val = y[train_size:]

# Create Genetic Algorithm

evaluator = GA.SimplePipReturnEvaluator(threshold=0.4)
crossover = GA.PreserveBestCrossover(preserve_rate=1.0)
mutation = GA.PreserveBestMutation(preserve_rate=1.0)
ga = GA.GeneticAlgorithm(
	evaluator, crossover, mutation,
	num_models=100, 
	survival_rate=0.2
)

ga.fit(
	train_data=(X_train, y_train),
	val_data=(X_val, y_val)
)














