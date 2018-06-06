# -*- coding: utf-8 -*-
from numba import cuda
import numpy as np

@cuda.jit('void(float32[:,:], float32[:,:], float32[:,:], float32[:,:], int32)', nopython = True)
def vecSolve(er, out, q, dcp, n):
	x, y = cuda.grid(2)
	if x == 0 and y < n:
		out[y, x] = 0
	elif y == 0 and x < n:
		out[y, x] = 0
	elif x == n - 1 and y < n:
		out[y, x] = 0
	elif y == n - 1 and x < n:
		out[y, x] = 0
	elif x < n and y < n:
		tmp = q[y, x - 1] + q[y, x + 1] + q[y - 1, x] + q[y + 1, x]
		tmp = tmp / 4 + dcp[y, x]
		out[y, x] = tmp
	if x < n and y < n:
		er[y, x] = out[y, x] - q[y, x]

def solve(q0, p):
	q = np.array(q0, dtype = np.float32)
	n = q.shape[0]
	d = 1 / n
	dc = d * d / 4
	dcp = dc * p
	l2 = n ** 2
	threadN = (8, 8)
	blockN = ((n + threadN[0] - 1) // threadN[0], (n + threadN[1] - 1) // threadN[1])
	while (True):
		# solve poisson equation by CUDA
		qtmp = np.zeros((n, n), dtype=np.float32)
		er = np.zeros((n, n), dtype=np.float32)
		vecSolve[blockN, threadN](er, qtmp, q, dcp, n)

		# error function
		er = np.sqrt(np.dot(er, er).sum())/n
		if er < 1e-3:
			return qtmp
		q = qtmp

if __name__ == "__main__":
	p = np.zeros((8, 8), dtype=np.float32)
	p[0] = np.ones(8, dtype=np.float32)
	p[len(p) - 1] = np.ones(8, dtype=np.float32)
	for i in range(len(p)):
		p[i][0] = float(1)
		p[i][len(p[0]) - 1] = float(1)
	q0 = np.zeros((8, 8), dtype=np.float32)
	q = solve(q0, p)
	print(q)
