import numpy as np
import cupy as cp

import time

x_gpu_1 = cp.array([1,2,3])
x_gpu_2 = cp.array([1,2,3])
x_cpu = np.array([1,2,3])

print(x_gpu_1)
print(x_cpu)

print(cp.multiply(x_gpu_1, x_gpu_2))

start = time.time()
gpu_test = cp.ones((1000,1000))*2
print('{:.2f}s'.format(time.time() - start))

start = time.time()
cpu_test = np.ones((1000,1000))*2
print('{:.2f}s'.format(time.time() - start))

start = time.time()
for i in range(5000):
	cp.asarray(cpu_test)
print('{:.2f}s'.format(time.time() - start))
# start = time.time()
# for i in range(100000):
# 	cp.multiply(gpu_test, gpu_test)
# print('{:.2f}s'.format(time.time() - start))

# start = time.time()
# for i in range(100000):
# 	np.multiply(cpu_test, cpu_test)
# print('{:.2f}s'.format(time.time() - start))


print(type(cp.asnumpy(x_gpu_1)))
test = cp.asnumpy(x_gpu_1)
test[0] = 5
print(x_gpu_1)
print(test)