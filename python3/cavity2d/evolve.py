# -*- coding: utf-8 -*-
import numpy as np
from numba import cuda

@cuda.jit
def vecEvolve(out, ps, om, n, d, d2, dt, nu):
	x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
	if x == 0 and y < n:
		out[y][x] = -2 * ps[y][1] / d2
	elif x == n - 1 and y < n:
		out[y][x] = -2 * ps[y][n - 2] / d2
	elif y == 0 and x < n:
		out[y][x] = -2 * ps[1][x] / d2
	elif y == n - 1 and x < n:
		out[y][x] = -2 * (ps[n - 2][x] + d) / d2
	elif x < n and y < n:
		# 1st term
		psx = ps[y][x + 1] - ps[y][x - 1]
		psy = ps[y + 1][x] - ps[y - 1][x]
		omx = om[y][x + 1] - om[y][x - 1]
		omy = om[y + 1][x] - om[y - 1][x]
		tmp1 = dt / (4 * d2) * (psx * omy - psy * omx)

		# 2nd term
		sx = om[y][x + 1] + om[y][x - 1] - 2 * om[y][x]
		sy = om[y + 1][x] + om[y - 1][x] - 2 * om[y][x]
		tmp2 = (dt * nu / d2) * (sx + sy)

		# output value
		out[y][x] = om[y][x] + tmp1 + tmp2
		
def evolve(ps, om, dt, nu):
	n = len(ps)
	d = 1 / n
	d2 = d ** 2
	threadN = (8, 8)
	blockN = ((n + threadN[0] - 1) // threadN[0], (n + threadN[1] - 1) // threadN[1])
	r = np.zeros((n, n), dtype=np.float32)
	vecEvolve[blockN, threadN](r, ps, om, n, d, d2, dt, nu)
	return r

if __name__ == "__main__":
	om = np.zeros((8, 8), dtype=np.float32)
	ps = np.zeros((8, 8), dtype=np.float32)
	dt = 0.01
	nu = 0.025
	om = evolve(ps, om, dt, nu)
	om = evolve(ps, om, dt, nu)
	om = evolve(ps, om, dt, nu)
	print(om)

