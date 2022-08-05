#!/usr/bin/env python3

import argparse
import math
import struct
from typing import Callable, Tuple

import numpy as np
from matplotlib import pyplot as plt

from utils.utils import to_dB, from_dB


exp = np.exp
ln = np.log
log2 = np.log2
sqrt = np.sqrt
tanh = np.tanh

def exp2(x):
	return np.power(2.0, x)

def inv_sqrt(x):
	return 1.0 / np.sqrt(x)


LOG2_E = log2(math.e)
LOG2_10 = log2(10.0)


def pun_float_to_int(x: float) -> int:
	return struct.unpack("@i", struct.pack("@f", float(x)))[0]


def pun_int_to_float(x: int) -> float:
	return struct.unpack("@f", struct.pack("@i", int(x)))[0]


@np.vectorize
def inv_sqrt_approx(x, order=1):
	xhalf = 0.5*x

	i = pun_float_to_int(x)
	i = 0x5F375A86 - ( i >> 1 )
	x = pun_int_to_float(i)

	for _ in range(order):
		# Newton-Raphson step - repeat for more accuracy
		x = x*(1.5 - xhalf*x*x )

	return x


@np.vectorize
def sqrt_approx(x, order=1):
	i = pun_float_to_int(x)
	i -= (1 << 23)
	i >>= 1
	i += (1 << 29)
	y = pun_int_to_float(i)

	for _ in range(order):
		# Newton-Raphson step - repeat for more accuracy
		y = ( y*y + x ) / ( 2*y )

	return y


@np.vectorize
def log2_approx(x, c1=0x3f800000, c2=0x00800000):
	# 0x00800000 = 0x40000000 - 0x3f800000
	return float(pun_float_to_int(x) - c1) / c2


@np.vectorize
def ln_approx(x, c1=0x3f800000, c2=0x00800000):
	#c1 = 0x3f78acfb
	y = float(pun_float_to_int(x) - c1) / (c2 * LOG2_E)
	#y *= 1.03 # optional, improves accuracy slightly
	return y


@np.vectorize
def exp2_approx(x, c1=0x3f800000, c2=0x00800000):
	#c1 = 0x3f78acfb
	# TODO: round to int instead?
	y_int = int(x*c2 + c1)
	return pun_int_to_float(y_int)


@np.vectorize
def exp_approx(x, c1=0x3f800000, c2=0x00800000):
	#c1 = float(0x3f800000)
	#c1 = float(0x3f78acfb)
	#c2 = float(0x00800000)*LOG2_E
	# TODO: round to int instead?
	y_int = int(x*c2*LOG2_E + c1)
	y = pun_int_to_float(y_int)
	#y *= 0.96 # optional, improves accuracy slightly

	# Newton-Raphson - these don't help because ln_approx is just as imprecise
	# Might work better once I find better "magic numbers" though
	#y = y - y*ln_approx(y) + x*y

	return y


@np.vectorize
def to_dB_approx(x, c1=0x3f800000, c2=0x00800000):
	def log10_approx(x):
		y = float(pun_float_to_int(x) - c1) / (c2 * LOG2_10)
		return y

	return 20.0 * log10_approx(x)


@np.vectorize
def from_dB_approx(x, c1=0x3f800000, c2=0x00800000):
	def exp10_approx(x):
		y_int = int(x*c2*LOG2_10 + c1)
		y = pun_int_to_float(y_int)
		return y

	return exp10_approx(x / 20.0)


@np.vectorize
def exp_approx_multi(x):

	c1 = 0.5*ln(2.0)
	c2 = 0.25*ln(2.0)
	expc1 = exp(c1)
	expc2 = exp(c2)

	exp_x = [
		exp_approx(x),
		exp_approx(x + c1) / expc1,
		exp_approx(x - c1) * expc1,
		exp_approx(x + c2) / expc2,
		exp_approx(x - c2) * expc2,
	]

	y = sum(exp_x) / 5.0

	# 0.96 is just from "eyeballing" it
	#y *= 0.96

	return y


@np.vectorize
def tanh_approx(x, c1=0x3f800000, c2=0x00800000):
	if True:
		exp_x = exp_approx(x, c1=c1, c2=c2)
		exp_nx = exp_approx(-x, c1=c1, c2=c2)
		#exp_nx = 1.0 / exp_x # actually less precise
	elif False:
		exp_x    = 0.5*(exp_approx(x, c1=c1, c2=c2) + 1.0/exp_approx(-x, c1=c1, c2=c2))
		exp_nx = 0.5*(exp_approx(-x, c1=c1, c2=c2) + 1.0/exp_approx(x, c1=c1, c2=c2))
	else:
		exp_x = exp_approx_multi(x, c1=c1, c2=c2)
		exp_nx = exp_approx_multi(-x, c1=c1, c2=c2)

	y = (exp_x - exp_nx) / (exp_x + exp_nx)

	# Newton Raphson - doesn't seem to help
	#y = y - (1.0 - y*y) *  (0.5*ln_approx((1+y)/(1-y)) - x)

	return y


def sanity_check():
	x = math.pi

	x_int = pun_float_to_int(x)

	print(f'{x=:f}')
	print(f'{x_int=}')
	print('')
	print('log2(x)=%f' % (math.log(x)/math.log(2)))
	print(f'{log2_approx(x)=:f}')
	print('exp2(x)=%f' % (2.0**x))
	print(f'{exp2_approx(x)=:f}')
	print('')
	print(f'{log2_approx(0)=:f}')
	print(f'{ln_approx(0)=:f}')
	print(f'{ln_approx(math.e)=:f}')
	print(f'{exp2_approx(0)=:f}')
	print(f'{exp2_approx(-1)=:f}')
	print('')
	print(f'{to_dB_approx(1)=:f}')
	print(f'{to_dB_approx(sqrt(2))=:f}')
	print(f'{to_dB_approx(2)=:f}')
	print(f'{to_dB_approx(10)=:f}')
	print(f'{to_dB_approx(0.5)=:f}')
	print(f'{to_dB_approx(0.1)=:f}')
	print(f'{to_dB_approx(from_dB(-72))=:f}')
	print(f'{from_dB(0)=:f}')
	print(f'{from_dB(6.02)=:f}')
	print(f'{from_dB(12.04)=:f}')
	print(f'{from_dB(20)=:f}')
	print(f'{from_dB(-6.02)=:f}')
	print(f'{from_dB(-12.04)=:f}')
	print(f'{from_dB(-20)=:f}')
	print(f'{from_dB(-120)=:f}')


def plot_func(
		f_exact: Callable,
		f_approx: Callable,
		x_range: Tuple[float, float],
		title='',
		ax_plot=None,
		ax_deriv=None,
		ax_err=None,
		rel_err=False,
		num_points=5000):

	if not title:
		title = f'{f_exact.__name__}(x)'

	if ax_plot is None:
		ax_plot = plt.gca()

	x = np.linspace(x_range[0], x_range[1], num_points)
	y_exact = f_exact(x)
	y_approx = f_approx(x)

	ax_plot.plot(x, y_exact, label='Exact')
	ax_plot.plot(x, y_approx, label='Approximation')
	ax_plot.grid()
	ax_plot.set_title(title)

	if ax_deriv is not None:
		dy_exact = np.gradient(y_exact, x)
		dy_approx = np.gradient(y_approx, x)

		ax_deriv.plot(x, dy_exact, label='Exact')
		ax_deriv.plot(x, dy_approx, label='Approximation')
		ax_deriv.grid()
		ax_deriv.set_title('d/dx %s' % title)

	if ax_err is not None:
		if rel_err:
			err = np.abs((y_approx - y_exact) / y_exact)
		else:
			err = np.abs(y_approx - y_exact)

		ax_err.plot(x, err, 'r')
		ax_err.set_ylabel('rel error' if rel_err else 'abs error')
		ax_err.grid()


def sweep_1d(x, f_exact, f_approx, f_err) -> Tuple[np.ndarray, int, int, float]:

	c1_init = 0x3f800000
	c2 = 0x00800000

	y_exact = f_exact(x)

	c1_min_err = [c1_init]
	min_err = [None]
	def update_err(c1, c2, err):
		if not np.isnan(err):
			if (not min_err[0]) or (err < min_err[0]):
				min_err[0] = err
				c1_min_err[0] = c1
				print('c1 = 0x%08x, c2 = 0x%08x, err^2 = %.6e <-- new minimum' % (c1, c2, err))
			else:
				#if (err < 0.9):
				print('c1 = 0x%08x, c2 = 0x%08x, err^2 = %.6e' % (c1, c2, err))

	# A bisection search would be way faster, but whatever

	min  = 0x3D000000
	max  = 0x3FF00000
	step = 0x00100000

	while (step >= 1):

		print('0x%08x to 0x%08x, step: 0x%08x' % (min, max, step))

		prev_err = None
		for c1 in np.arange(min, max+1, step):
			y_approx = f_approx(x, c1=c1, c2=c2)
			err = f_err(y_approx, y_exact)
			update_err(c1, c2, err)

			# Error should have a single minimum - so if it ever increases, can break loop early
			if prev_err and (err > prev_err):
				break
			prev_err = err

		min = c1_min_err[0] - step
		max = c1_min_err[0] + step
		step //= 16

	y_approx = f_approx(x, c1_min_err[0], c2)

	return y_approx, c1_min_err[0], c2, min_err[0]


def sweep_2d_plot(x, f_exact, f_approx, f_err, title=None):

	y_exact = f_exact(x)

	#c1_init = 0x3f800000
	#c2_init = 0x00800000

	#c1_sweep = np.arange(0x3D000000, 0x3FF00000+1, 0x00100000, dtype=np.uint32)
	c1_sweep = np.arange(0x3F000000, 0x3FF00000+1, 0x00100000, dtype=np.uint32)

	#c2_sweep = np.arange(0x00000000, 0x00F00000+1, 0x00100000, dtype=np.uint32)
	c2_sweep = np.arange(0x00700000, 0x00900000+1, 0x00010000, dtype=np.uint32)
	#c2_sweep = np.arange(0x00800000, 0x00800000+1, 0x00010000, dtype=np.uint32)
	#c2_sweep = 0x40000000 - c1_sweep

	num_vals = len(c1_sweep) * len(c2_sweep)

	plt_c1  = np.zeros(num_vals, dtype=c1_sweep.dtype)
	plt_c2  = np.zeros(num_vals, dtype=c2_sweep.dtype)
	plt_err = np.zeros(num_vals)

	n = 0

	print('Sweeping %u c1 values x %u c2 values = %u total' % (len(c1_sweep), len(c2_sweep), num_vals))
	for c1 in c1_sweep:
		print('c1 = 0x%08x' % c1)
		for c2 in c2_sweep:

			y_approx = f_approx(x, c1, c2)
			err = f_err(y_approx, y_exact)

			plt_c1[n] = c1
			plt_c2[n] = c2
			plt_err[n] = err

			n += 1

	#plt_c1 = np.array(plt_c1)
	#plt_c2 = np.array(plt_c2)
	#plt_err = np.array(plt_err)

	plt_err = np.log10(np.abs(plt_err))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	#ax.plot_surface(plt_c1, plt_c2, plt_err)
	#ax.plot_wireframe(plt_c1, plt_c2, plt_err)
	ax.plot_trisurf(plt_c1, plt_c2, plt_err)
	#ax.scatter(plt_c1, plt_c2, plt_err)

	plt.xlabel('C1')
	plt.ylabel('C2')
	ax.set_zlabel('log10(err)')

	if title:
		plt.title(title)


def squ_rel_err(y_approx, y_exact):
	err = 1.0 - y_approx[y_exact != 0] / y_exact[y_exact != 0]
	return np.average(np.square(err))


def squ_abs_err(y_approx, y_exact):
	err = y_approx - y_exact
	return np.average(np.square(err))


def max_abs_err(y_approx, y_exact):
	err = y_approx - y_exact
	err = np.abs(err)
	return np.amax(err)


def max_deriv_err(y_approx, y_exact):
	return np.amax(np.diff(y_approx) - np.diff(y_exact))


def optimize():
	x = np.linspace(-5, 5, 5000)

	# TODO: use scipy.optimize for this

	if True:
		sweep_2d_plot(x, exp, exp_approx, squ_rel_err, title='exp')

	if False:
		sweep_2d_plot(x, tanh, tanh_approx, squ_abs_err, title='tanh')
		#sweep_2d_plot(x, tanh, tanh_approx, max_abs_err, title='tanh')
		#sweep_2d_plot(x, tanh, tanh_approx, max_deriv_err, title='tanh')

	if True:
		print('')
		print('exp(x)')
		print('')

		y_exp_approx, c1_min_exp_err, c2_min_exp_err, min_exp_err = sweep_1d(x, exp, exp_approx, squ_rel_err)

		print('')
		print('tanh(x)')
		print('')

		y_tanh_approx, c1_min_tanh_err, c2_min_tanh_err, min_tanh_err = sweep_1d(x, tanh, tanh_approx, max_abs_err)

		print('Minimum exp error: c1 = 0x%08x, c2 = 0x%08x, err^2 = %.12f' % (c1_min_exp_err, c2_min_exp_err, min_exp_err))
		print('Minimum tanh error: c1 = 0x%08x, c2 = 0x%08x, err^2 = %.12f' % (c1_min_tanh_err, c2_min_tanh_err, min_tanh_err))

		fig, (ax1, ax2) = plt.subplots(2, 1)
		fig.suptitle('exp')

		y_exact = exp(x)
		y_approx = exp_approx(x, c1=c1_min_exp_err, c2=c2_min_exp_err)
		err = y_approx[y_exact != 0] / y_exact[y_exact != 0] - 1.0

		ax1.plot(x, y_exact, x, y_approx)
		ax1.grid()

		ax2.plot(x, err)
		ax2.grid()

		fig, (ax1, ax2) = plt.subplots(2, 1)
		fig.suptitle('tanh')

		y_exact = tanh(x)
		y_approx = tanh_approx(x, c1_min_tanh_err, c2_min_tanh_err)
		err = y_approx - y_exact

		ax1.plot(x, y_exact, x, y_approx)
		ax1.grid()

		ax2.plot(x, err)
		ax2.grid()

	plt.show()


def get_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--optimize', action='store_true')
	return parser


def plot(args):

	fig, ((ax_plot1, ax_plot2), (ax_deriv1, ax_deriv2)) = plt.subplots(2, 2, sharex='col')
	plot_func(ln, ln_approx, (0.01, 20), 'ln(x)', ax_plot=ax_plot1, ax_deriv=ax_deriv1)
	plot_func(exp, exp_approx, (-5, 5), 'e^x', ax_plot=ax_plot2, ax_deriv=ax_deriv2)

	fig, ((ax_plot1, ax_plot2), (ax_deriv1, ax_deriv2)) = plt.subplots(2, 2, sharex='col')
	plot_func(inv_sqrt, inv_sqrt_approx, (0.1, 20), '1/sqrt(x)', ax_plot=ax_plot1, ax_deriv=ax_deriv1)
	plot_func(sqrt, sqrt_approx, (0, 20), ax_plot=ax_plot2, ax_deriv=ax_deriv2)

	fig, (ax_plot, ax_deriv) = plt.subplots(2, 1, sharex='col')
	plot_func(tanh, tanh_approx, (-5, 5), ax_plot=ax_plot, ax_deriv=ax_deriv)

	fig, (ax_ln_exp, ax_exp_ln) = plt.subplots(2, 1)
	plot_func(np.zeros_like, lambda x: ln_approx(exp_approx(x)) - x, (-5, 5), 'ln(exp(x))', ax_plot=ax_ln_exp)
	plot_func(np.zeros_like, lambda x: exp_approx(ln_approx(x)) - x, (0.01, 20), 'exp(ln(x))', ax_plot=ax_exp_ln)

	fig, ((ax_plot1, ax_plot2), (ax_err1, ax_err2)) = plt.subplots(2, 2, sharex='col')
	plot_func(lambda x: x, lambda x: to_dB_approx(from_dB(x)), (-144, 48), 'to_dB', ax_plot=ax_plot1, ax_err=ax_err1)
	plot_func(lambda x: x, lambda x: from_dB_approx(to_dB(x)), (from_dB(-144), from_dB(48)), 'from_dB', ax_plot=ax_plot2, ax_err=ax_err2, rel_err=True)

	plt.show()


def main(args):

	if args.optimize:
		optimize()
	else:
		sanity_check()
		plot(args)
