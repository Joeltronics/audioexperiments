#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.special
import struct

# http://math.stackexchange.com/questions/107292/rapid-approximation-of-tanhx

log2e = math.log2(math.e)


def pun_float_to_int(x: float) -> int:
	fx = float(x)
	return struct.unpack("@i", struct.pack("@f", fx))[0]


def pun_int_to_float(x: int) -> float:
	ix = int(x)
	return struct.unpack("@f", struct.pack("@i", ix))[0]


def exp2_approx(x):
	c1 = float(0x00800000)
	c2 = float(0x3f800000)
	#c2 = float(0x3f78acfb)
	y_int = int(x*c1 + c2)
	return pun_int_to_float(y_int)


def exp_approx(x):
	c1 = float(0x00800000) * log2e
	#c2 = float(0x3f800000)
	c2 = float(0x3f78acfb)
	y_int = int(x*c1 + c2)
	return pun_int_to_float(y_int)


def tanh_approx_type_pun(x):

	y = np.zeros_like(x)
	for n, xx in enumerate(x):
		exp_x = exp_approx(xx)
		exp_nx = exp_approx(-xx)
		yy = (exp_x - exp_nx) / (exp_x + exp_nx)
		y[n] = yy

	return y


def nCr(n, r):
	return scipy.special.comb(n, r)


def tanh_approx_continued_fraction_5(x):

	x2 = np.square(x)

	y = 9.0 + x2 / 11.0
	y = 7.0 + x2 / y
	y = 5.0 + x2 / y
	y = 3.0 + x2 / y
	y = 1.0 + x2 / y
	y = x / y
	
	return y


def tanh_approx_continued_fraction_3(x):
	
	x2 = np.square(x)

	y = 5.0 + x2 / 7.0
	y = 3.0 + x2 / y
	y = 1.0 + x2 / y
	y = x / y

	return y


def tanh_approx_continued_fraction(x, order):

	x2 = np.square(x)

	#coeffs = [1, 3, 5, 7, 9, 11, ...]
	coeffs = range(1, 2*order, 2)
	y = coeffs[-1]

	for n in range(order, 0, -1):
		y = coeffs[n-1] + x2 / y

	y = x / y

	return y


def tanh_approx_pade_3(x):

	x2 = np.square(x)
	x4 = np.square(x2)
	x6 = x2 * x4

	num = x*(600.0 + 70.0*x2 + x4)
	den = 600.0 + 270.0*x2 + 11.0*x4 + x6/24.0

	return num / den
	

def pade_n(x, n):

	y = 0.0
	for j in range(n):

		num = float(nCr(n, j)) * np.power(x,j)
		den = float(nCr(2*n, j) * math.factorial(j))

		y += num / den

	return y


def tanh_approx_pade(x, order: int):
	pn = pade_n(x, order)
	pn_neg = pade_n(-x, order)

	pn2 = np.square(pn)
	pn_neg2 = np.square(pn_neg)

	num = pn2 - pn_neg2
	den = pn2 + pn_neg2

	return num/den


def tanh_approx_taylor(x, order: int):

	y = 1.0*x

	if order >= 3:
		x2 = x*x
		x3 = x2*x

		y -= (1.0/3.0)*x3

	if order >= 5:
		x5 = x3*x2
		y += (2.0/15.0)*x5

	if order >= 7:
		x7 = x5*x2
		y -= (17.0/315.0)*x7

	return y


def get_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('-o', '--order', type=int, default=7)
	parser.add_argument('-r', '--range', type=int, default=20)
	return parser


def plot(args, verbose=False):

	order = args.order

	x = np.linspace(-args.range, args.range, 1024)

	yactual = np.tanh(x)

	ycf = tanh_approx_continued_fraction(x, order)

	if order == 3:
		yp = tanh_approx_pade_3(x)
	else:
		yp = tanh_approx_pade(x, order)

	#yt = tanh_approx_taylor(x, order)
	yt = tanh_approx_type_pun(x)

	if (order % 2 == 1) and verbose:
		# if odd only
		n_min = np.argmin(ycf)
		n_max = np.argmax(ycf)

		print('Continued fraction:')
		print('Min: %.6f, %.6f' % (x[n_min], ycf[n_min]))
		print('Max: %.6f, %.6f' % (x[n_max], ycf[n_max]))

		n_min = np.argmin(yp)
		n_max = np.argmax(yp)

		print('Pade:')
		print('Min: %.12f, %.12f' % (x[n_min], yp[n_min]))
		print('Max: %.12f, %.12f' % (x[n_max], yp[n_max]))

	ep = yp - yactual
	ecf = ycf - yactual
	et = yt - yactual

	#ep = 1.0 - yp/yactual
	#ecf = 1.0 - ycf/yactual
	#et = 1.0 - yt/yactual

	#ep = np.abs(ep)
	#ecf = np.abs(ecf)
	#et = np.abs(et)

	plt.figure()

	plt.subplot(311)
	plt.plot(x, yp, label='Pade')
	plt.plot(x, yp, label='CF')
	plt.plot(x, yp, label='Type pun')
	plt.plot(x, yp, label='Actual')
	plt.grid()
	plt.legend(loc=4)
	plt.xlim([-2., 2.])
	plt.ylim([-1.1, 1.1])

	plt.subplot(312)
	plt.plot(x, yp, label='Pade')
	plt.plot(x, ycf, label='CF')
	plt.plot(x, yt, label='Type pun')
	plt.plot(x, yactual, label='Actual')
	plt.grid()
	plt.legend(loc=4)
	plt.ylim([-1.1, 1.1])

	plt.subplot(313)
	plt.plot(x, ep, label='Pade')
	plt.plot(x, ecf, label='CF')
	plt.plot(x, et, label='Type pun')
	plt.xlim([-4.7, 4.7])
	#plt.ylim([0, 0.001])
	plt.ylim([-0.01, 0.01])
	plt.grid()
	plt.title('Error')
	plt.legend(loc=4)

	plt.show()


def main(args):
	plot(args, verbose=True)
