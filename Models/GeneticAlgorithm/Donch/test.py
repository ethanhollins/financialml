from multiprocessing import Pool, cpu_count
import time

def f(x):
	return x*x

def x(x):
	return x*x*2

if __name__ == '__main__':
	pool = Pool(cpu_count())
	start = time.time()
	print(pool.map(f, range(10000000))[:5])
	print(time.time() - start)
	print(pool.map(f, range(10000000))[:5])
	print(time.time() - start)


