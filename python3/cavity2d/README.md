# 2d Cavity analysis by psi-omega method

## requirements
- numpy
- scipy
- numba
- matplotlib
- NVidia CUDA

## parameters
- mul: mesh mugnifier
- grid: number of grids for each axises
- dt: interval for evolving simulation time

Note: dt * (gird * mul) ** 2 must be less than 1
to converge the results.
