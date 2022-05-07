#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from math import pi, tanh
import math


#xp = 0.0
xp = 0.25
#xp = 0.75
#xp = 1.0
#xp = 1.5
#xp = 2.0


def dtanhdx(x):
	#d/dx tanh(x) = (sech(x))^2 = (cosh(x))^-2
	cosh = np.cosh(x)
	return np.power(cosh, -2.0)


x = np.arange(-1000, 1000) / 100.0
y = np.tanh(x)

yp = tanh(xp)

# y = mx + b
# b = y - mx

m = dtanhdx(xp)
b = tanh(xp) - m*xp

y_lin = m*x + b

err = y - y_lin
abs_err = np.abs(err)

plt.figure()

plt.plot(x, y, label='true')
plt.plot(xp, yp, '.', label='tangent point')
plt.plot(x, y_lin, label='linear')
plt.plot(x, abs_err, label='abs error')

plt.ylim([-1, 1])
plt.xlim([-4, 4])

print('Error at +0.1:  %f' % err[np.where(x == (xp + 0.1))])
print('Error at -0.1:  %f' % err[np.where(x == (xp - 0.1))])

print('Error at +0.05: %f' % err[np.where(x == (xp + 0.05))])
print('Error at -0.05: %f' % err[np.where(x == (xp - 0.05))])

print('Error at +0.01: %f' % err[np.where(x == (xp + 0.01))])
print('Error at -0.01: %f' % err[np.where(x == (xp - 0.01))])

plt.grid()
plt.legend()

plt.show()



