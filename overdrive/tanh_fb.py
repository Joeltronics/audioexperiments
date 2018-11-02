#!/usr/bin/env python3


import numpy as np
from typing import Union
from utils import utils
from scipy import integrate


def _tanh_x_dx(x):
	"""d/dx tanh(x)"""
	return 1 - np.square(np.tanh(x))


def _fb_gain_formula(open_loop, neg_feedback):
	"""
	:param open_loop: open-loop gain 
	:param neg_feedback: positive value, < 1
	:return: 
	"""
	return open_loop / (1.0 + open_loop * neg_feedback)


def _derivative(y, x):
	return np.gradient(y, x)


def tanh_fb(x: Union[float, np.ndarray], open_loop_gain, neg_feedback):
	"""
	:param open_loop_gain: open-loop gain
	:param neg_feedback: positive value, < 1
	:return:
	"""
	pass


def plot(args):
	pass


def main(args):
	from matplotlib import pyplot as plt

	def sanity_check(x):
		y = np.tanh(x)
		dydx = _tanh_x_dx(x)
		plt.figure()
		plt.plot(x, y, label='tanh(x)')
		plt.plot(x, dydx, label='d/dx tanh(x)')
		plt.grid()
		plt.title('Sanity check')
		plt.legend()

	sanity_check(np.linspace(-10, 10, 1001))

	olg = 100.0  # Open loop gain - seems to only work in a certain range. 100 and 1000 work
	nfb = 1.0 / 10.0  # Negative feedback

	gain = _fb_gain_formula(olg, nfb)

	n_samp = 20001
	#x_range = 2.0
	x_range = 0.5

	x = np.linspace(-x_range, x_range, n_samp)
	x_half = x[:-1] + np.diff(x) / 2  # Halfway between samples

	def _opamp_x_dx(x):
		return olg * _tanh_x_dx(olg * x)

	def _opamp(pos, neg):
		return np.tanh(olg * (pos - neg))

	def _ideal_opamp(x):
		y_ideal = x * olg
		for n, samp in enumerate(y_ideal):
			if samp > 1.0:
				y_ideal[n] = 1.0
			elif samp < -1.0:
				y_ideal[n] = -1.0
		return y_ideal

	def _fb(x):
		return x * nfb

	def _fb_x_dx(x):
		return nfb

	def _iterate_1_samp(x, n_iter_max=100, eps=1.0e-6):

		assert n_iter_max >= 1

		y = _opamp(x, 0.0)
		neg_in = _fb(y)

		limiting = 1.0 / (olg * nfb)

		y_prev = y
		for n in range(n_iter_max):
			y_theoretical = _opamp(x, neg_in)
			y = y_theoretical * limiting + y_prev * (1.0 - limiting)
			neg_in = _fb(y)

			if abs(y - y_prev) < eps:
				break
			y_prev = y

		return y, n + 1

	def _iterative(x):
		y = np.zeros_like(x)
		n_iter = np.zeros_like(x)
		for n, samp in enumerate(x):
			y[n], n_iter[n] = _iterate_1_samp(samp)
		return y, n_iter

	y_iter, n_iter = _iterative(x)

	y_tanh = np.tanh(x * gain)

	y_ideal = _ideal_opamp(x)

	# Calculate gain for these

	gain_iter = _derivative(y_iter, x)
	gain_tanh = _derivative(y_tanh, x)
	gain_ideal = _derivative(y_ideal, x)

	# Here's the real meat of all this: try to do this non-iteratively

	y_ol = np.tanh(olg * x)
	gain_ol = _tanh_x_dx(olg * x) * olg
	gain_ol_sanity = _derivative(y_ol, x)
	sanity = np.max(np.abs(gain_ol - gain_ol_sanity))
	if sanity > 0.01:
		print('gain_ol sanity check failed! ' + str(sanity))
		return 1

	gain_fb = np.ones_like(x) / nfb
	gain_non_iter = _fb_gain_formula(gain_ol, 1.0 / gain_fb)

	max_gain_err = np.max(np.abs(gain_iter - gain_non_iter))
	if max_gain_err > 0.01:
		print('WARNING! max error between iterative and non-iterative %f' % max_gain_err)

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.plot(x, y_iter, label='Iterative')
	plt.legend()
	plt.ylabel('Transfer Function')
	plt.xlim([-x_range, x_range])
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.plot(x, gain_iter, label='Iterative')
	plt.plot(x, gain_tanh, label='tanh')
	plt.plot(x, gain_ideal, label='Ideal opamp')
	plt.plot(x, gain_non_iter, label='Non-iterative')
	plt.ylabel('Gain')
	plt.legend()
	plt.xlim([-x_range, x_range])
	plt.grid()

	plt.draw()

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.plot(
		x, y_iter,
		x, y_tanh,
		x, y_ideal)
	plt.legend(['Actual', 'tanh', 'Ideal'], loc=2)
	plt.ylabel('Transfer Function')
	plt.xlim([-x_range, x_range])
	plt.title('Iterative solution; OL gain = ' + str(olg) + ", FB gain = " + str(1.0 / nfb))
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.plot(x, n_iter)
	plt.ylabel('# iterations')
	plt.xlim([-x_range, x_range])
	plt.grid()

	plt.draw()

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.plot(x, y_ol)
	plt.legend(['Open loop'], loc=2)
	plt.ylabel('Transfer Function')
	plt.xlim([-x_range, x_range])
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.plot(x, gain_ol, label='Open loop')
	plt.plot(x, gain_fb, label='Feedback')
	plt.plot(x, gain_non_iter, label='Total')
	plt.legend()
	plt.ylabel('Gain')
	plt.xlim([-x_range, x_range])
	plt.grid()

	if True:
		y_open_loop = np.tanh(olg * x)
		y_fb_only = x / nfb

		def ol_gain(x):
			return olg * _tanh_x_dx(olg * x)

		def gain(x):
			return ol_gain(x) / (1 + nfb * ol_gain(x))

		y_ol_gain = ol_gain(x)
		y_gain = gain(x)

		#y_actual = x * gain(x)
		#y_actual = np.cumsum(y_gain)
		#y_actual = np.cumsum(y_gain) / 50.0 - 1.0

		#y_actual = np.trapz(y_gain, x=x)
		#y_actual = np.trapz(y_gain, dx=(x[1]-x[0]))
		y_actual = integrate.cumtrapz(y_gain, x=x)
		y_actual = y_actual * 5.0 - 1.0

		plt.figure()

		plt.subplot(2, 1, 1)
		plt.title('Transfer func')
		plt.plot(x, y_open_loop, label='Open loop')
		plt.plot(x, y_fb_only, label='FB only')
		plt.plot(x_half, y_actual, label='Both')
		plt.legend()
		plt.grid()

		plt.xlim([-0.1, 0.1])
		plt.ylim([-1, 1])

		plt.subplot(2, 1, 2)
		plt.plot(
			x, y_ol_gain,
			[-2, 2], [1 / nfb, 1 / nfb],
			x, y_gain)
		plt.legend(['open loop', 'FB only', 'Both'])
		plt.title('Gain')
		plt.grid()

	plt.show()
