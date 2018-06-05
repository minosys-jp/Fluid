# -*- coding: utf-8 -*-
import numpy as np
import evolve
import poisson
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import ArtistAnimation

mul = 1
grid = 20
n = grid * mul
psi = np.zeros((n, n), dtype=np.float32)
ome = np.zeros((n, n), dtype=np.float32)
dt = float(0.01 / mul)
nu = float(0.061)

ps = psi
om = ome
ps_list = []
#om_list = []
for t in range(int(1 / dt)):
	# omega を時間発展する
	omout = evolve.evolve(ps, om, dt, nu)

	# psi を poisson 方程式から求める
	psout = poisson.solve(ps, omout)
	ps_list.append(psout)
#	om_list.append(omout)
	ps = psout
	om = omout

kx = np.array([ [-1, 1] ])
ky = np.array([ [-1], [1] ])
xx, yy = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))
xs = xx[0::mul, 0::mul]
ys = yy[0::mul, 0::mul]
fig = plt.figure(figsize=(10,10))
ims = []
for t in range(4, len(ps_list), 5):
	u = signal.correlate(ps_list[t], ky, 'same') * n
	v = signal.correlate(ps_list[t], -kx, 'same') * n

	# 境界条件
	ut = u.T
	vt = v.T
	ut[0] = np.zeros(n, dtype=np.float32)
	ut[n - 1] = np.zeros(n, dtype=np.float32)
	vt[0] = np.zeros(n, dtype=np.float32)
	vt[n - 1] = np.zeros(n, dtype=np.float32)
	u = ut.T
	v = vt.T
	u[0] = np.zeros(n, dtype=np.float32)
	u[n - 1] = np.ones(n, dtype=np.float32)
	v[0] = np.zeros(n, dtype=np.float32)
	v[n - 1] = np.zeros(n, dtype=np.float32)
	art = []
	sca = plt.scatter(xs.flatten(), ys.flatten(), c = 'blue', marker = 'o')
	art.append(sca)
	uu = u[0::mul, 0::mul]
	vv = v[0::mul, 0::mul]
	xe = xs + uu / 10
	ye = ys + vv / 10
	xxx = [[xs[i][j], xe[i][j]] for j in range(n) for i in range(n)]
	yyy = [[ys[i][j], ye[i][j]] for j in range(n) for i in range(n)]
	for i in range(n ** 2):
		seg, = plt.plot(xxx[i], yyy[i], 'r-')
		art.append(seg)
	tt = plt.title('Fluid dynamics in the cavity (nu=' + str(nu) + ')')
	art.append(tt)
	ims.append(art)
#	plt.cla()
	print('t=' + str(t) + ' Done.')
ani = ArtistAnimation(fig, ims, interval = 50, blit = False)
plt.show()
#print("Now saving file...")
#ani.save('cavity2d.mp4', writer='ffmpeg')
