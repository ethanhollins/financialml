import tensorflow as tf
import numpy as np
import time
from numba import jit

class timeit(object):
	def __init__(self):
		self.start = time.time()
		print('\nTimer module started.')
	def end(self):
		print('Timer module finished %.2f'%(time.time() - self.start))

def py_add(x, y):
	return x + y

@tf.function
def add(x, y):
	return x + y

timer = timeit()
print(py_add(1, 2))
timer.end()

timer = timeit()
print(add(1, 2))
timer.end()

timer = timeit()
print(add(tf.constant(1, tf.int32), tf.constant(2, tf.int32)))
timer.end()

@tf.function(input_signature=(
	tf.TensorSpec(shape=[2], dtype=tf.int32),
	tf.TensorSpec(shape=[], dtype=tf.int32)
))
def add_w_spec(x, y):
	return x + y

timer = timeit()
print(add_w_spec(tf.constant([1, 3], tf.int32), tf.constant(2, tf.int32)))
timer.end()


class cust_class(object):
	def __init__(self):
		self._result = None
		self._c_dir = None
		self._last_dir = None
		self._last_entry = None
		self._c_out = None

	def reset(self):
		self._result = None
		self._c_dir = None
		self._last_dir = None
		self._last_entry = None
		self._c_out = None

	@tf.function
	def __call__(self, y, out):
		if self._result is None:
			self._result = tf.Variable(0.0, tf.float32)
		if self._c_dir is None:
			self._c_dir = tf.Variable(-1, tf.int32)
		if self._last_dir is None:
			self._last_dir = tf.Variable(-1, tf.int32)
		if self._last_entry is None:
			self._last_entry = tf.Variable(0.0, tf.float32)
		if self._c_out is None:
			self._c_out = tf.Variable(0.0, tf.float32)

		for i in tf.range(out.shape[0]):
			self._c_out.assign(out[i])

			if self._c_out > (1.0 - threshold):
				self._c_dir.assign(1)
			elif self._c_out < threshold:
				self._c_dir.assign(0)
			else:
				self._c_dir.assign(-1)

			if self._c_dir != -1:
				if self._last_dir != -1:
					if self._last_dir != self._c_dir:
						if self._last_dir == 1:
							self._result.assign_add(y[i] - self._last_entry)
						else:
							self._result.assign_add(self._last_entry - y[i])
						self._last_dir.assign(self._c_dir)
						self._last_entry.assign(y[i])
				else:
					self._last_dir.assign(self._c_dir)
					self._last_entry.assign(y[i])

		return self._result

def py_cust_func(y, out):
	result = 0.0
	c_dir = -1
	last_dir = -1
	last_entry = 0.0

	for i in range(out.shape[0]):
		c_out = out[i]

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

	return result

threshold = 0.5
y = np.random.uniform(-100, 100, size=(100,))
y = np.array(y, dtype=np.float32)
out = np.random.normal(size=(100,))
out = np.array(out, dtype=np.float32)

print()
print(y.shape)
print(out.shape)

timer = timeit()
result = cust_class()(tf.constant(y, tf.float32), tf.constant(out, tf.float32))
timer.end()
print(result.numpy())

print('Running tensorflow...')
timer = timeit()
t = cust_class()
for i in range(100):
	t(tf.constant(y, tf.float32), tf.constant(out, tf.float32))
	t.reset()
timer.end()

print('Running python...')
timer = timeit()
for i in range(100):
	py_cust_func(y, out)
timer.end()

@jit
def jit_cust_func(y, out):
	result = 0.0
	c_dir = -1
	last_dir = -1
	last_entry = 0.0

	for i in range(out.shape[0]):
		c_out = out[i]

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

	return result

jit_cust_func(y, out)
print('Running jit...')
timer = timeit()
for i in range(100000):
	jit_cust_func(y, out)
timer.end()


@jit('int32(int32, int32)')
def jit_add(x, y):
	return x + y

print(jit_add(3,7))

