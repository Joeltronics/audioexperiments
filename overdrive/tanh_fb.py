#!/usr/bin/env python3


import numpy as np
from scipy import integrate

from utils import utils

from typing import Union, Tuple


def _clip1(x):
	"""Clip to range [-1, 1]"""
	return np.clip(x, -1., 1.)


def _tanh_x_dx(x):
	"""d/dx tanh(x)"""
	return 1. - np.square(np.tanh(x))


def _atanh_x_dx(x):
	"""d/dx atanh(x)"""
	return 1. / (1. - np.square(x))


def _fb_gain_formula(open_loop, neg_feedback):
	"""
	Calculate gain based on open-loop gain and negative feedback

	:param open_loop: open-loop gain 
	:param neg_feedback: positive value, < 1
	:return: resulting gain
	"""
	return open_loop / (1.0 + open_loop * neg_feedback)


def _interp_method_inner(olg, nfb, tanh_xg, clip_xg):
	interp_scale = 1.0 - 1.0 / (nfb * olg)  # assumes olg * nfb >> 1
	if not 0 < interp_scale < 1:
		raise ValueError('Interp method only works if nfb * olg > 1')
	return utils.lerp((tanh_xg, clip_xg), interp_scale)


def _interp_method(x, olg, nfb, gain=None):
	"""
	Very basic estimation by interpolation between tanh and hard clipped
	Not very good on its own (has a 2nd-order discontinuity), but great as an initial estimate for iterative methods

	:param x: input value
	:param olg: open-loop gain
	:param nfb: negative feedback (positive value < 1)
	:param gain: optional precalculated gain from _fb_gain_formula
	"""

	if gain is None:
		gain = _fb_gain_formula(olg, nfb)

	return _interp_method_inner(olg, nfb, np.tanh(gain * x), _clip1(gain * x))


def _nr_iterate_1_samp(x, olg, nfb, gain=None, n_iter_max=100, eps=1.0e-6) -> Tuple[float, int]:
	"""
	Iterate Newton-Raphson method (for single input sample)

	:param x: input value
	:param olg: open-loop gain
	:param nfb: negative feedback (positive value < 1)
	:param gain: ptional precalculated gain from _fb_gain_formula
	:param n_iter_max: Max number of iterations
	:param eps: maximum error
	"""

	assert n_iter_max >= 1

	if x == 0.:
		return 0., 0

	if gain is None:
		gain = _fb_gain_formula(olg, nfb)

	if x < -1/gain:
		y = -1 + eps
	elif x > 1/gain:
		y = 1 - eps
	else:
		y = _interp_method(x, olg, nfb, gain)

	for n in range(1, n_iter_max+1):
		y_prev = y

		# f(y) = olg*nfb*y + atanh(y) - olg*x
		fy = olg*nfb*y + np.arctanh(y) - olg*x

		# f'(y) = olg*nfb + _atanh_x_dx(y)
		fpy = olg*nfb + _atanh_x_dx(y)

		r = fy / fpy  # residue
		y -= r

		# At values where y is close to 1, we have a bit of a problem:
		# f(y) and f'(y) are only valid in range (-1, 1)
		# So N-R can result in a next estimate value that's out of bounds

		# If residue was smaller than eps, then just clip to (-1, 1), don't worry about next NR estimate
		if abs(r) < eps:
			y = _clip1(y)
			break

		# Otherwise, residue was still too large but we're out of bounds
		# In this case, go almost all the way to +/- 1
		if abs(y) >= 1.:
			y = utils.lerp((y_prev, 1. if y > 0 else -1.), 1. - eps)

	return y, n


def tanh_fb(x: Union[float, np.ndarray], open_loop_gain: float, neg_feedback: float) -> Union[float, np.ndarray]:
	"""
	Solves: y = tanh(olg * (x - nfb*y))
	i.e. olg*nfb*y + atanh(y) - olg*x = 0

	:param open_loop_gain: open-loop gain
	:param neg_feedback: positive value, < 1
	:return:
	"""

	if neg_feedback < 0:
		raise ValueError('Negative feedback must be a positive value (yeah, you read that right)')
	elif neg_feedback == 0:
		return np.tanh(open_loop_gain * x)

	gain = _fb_gain_formula(open_loop_gain, neg_feedback)

	if np.isscalar(x):
		y, _ = _nr_iterate_1_samp(x, open_loop_gain, neg_feedback, gain)
	else:
		y = np.zeros_like(x)
		for n, xx in enumerate(y):
			y[n], _ = _nr_iterate_1_samp(x, open_loop_gain, neg_feedback, gain)

	return y


def plot(args):
	from matplotlib import pyplot as plt
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--olg', type=float, help='Open-loop gain, default 100')
	parser.add_argument('--fbg', type=float, help='Gain from negative feedback (i.e. reciprocal of actual feedback), default 10')
	args = parser.parse_args(args)

	# Open loop gain
	# Use a relatively low value (100) because:
	# * Graphs are easier to see
	# * It's a more difficult case for most methods - except:
	# * Original method takes > 100 iter to converge if olg is too high
	#
	# Note that several pieces of code here make the assumption olg * nfb >> 1, so can't go *too* low
	if args.olg is not None:
		olg = args.olg
	else:
		olg = 100.0

	# Negative feedback
	if args.fbg is not None:
		recip_nfb = args.fbg
	else:
		recip_nfb = 10.

	nfb = 1.0 / recip_nfb

	if olg * nfb <= 1.:
		raise ValueError('Feedback must be < open-loop gain')
	elif olg * nfb < 9:
		print('WARNING: feedback should be much smaller than open-loop')

	gain = _fb_gain_formula(olg, nfb)

	n_samp = 20001
	x_range = 2.0
	# x_range = 0.5
	# x_range = 0.25

	n_iter_max = 100

	x = np.linspace(-x_range, x_range, n_samp)

	y = np.zeros_like(x)
	n_iter = np.zeros_like(x)
	for n, samp in enumerate(x):
		y[n], n_iter[n] = _nr_iterate_1_samp(samp, olg, nfb, gain, n_iter_max=n_iter_max)

	if any(n_iter >= n_iter_max):
		print('WARNING: Failed to converge in %i iterations' % np.amax(n_iter_max))

	#print('Calculating combined method')
	#y_iter_combined, n_iter_combined = _iterate_over_inputs(_combined, x)

	print('Calculating other methods')
	y_tanh = np.tanh(x * gain)
	y_open_loop = np.tanh(x * olg)
	y_interp_method = _interp_method(x, olg, nfb, gain)
	y_ideal = _clip1(x * gain)  # Ideal but with finite open-loop gain
	y_infinite_olg = _clip1(x / nfb)

	gain_tanh = np.gradient(y_tanh, x)
	gain_ideal = np.gradient(y_ideal, x)
	gain_interp = np.gradient(y_interp_method, x)
	gain_open_loop = np.gradient(y_open_loop, x)
	gain_actual = np.gradient(y, x)

	gain_ylims = [-0.1/nfb, 1.1/nfb]

	print('Plotting')

	plt.figure()

	plt.subplot(3, 1, 1)

	plt.title('Iterative solution; OL gain %g, FB gain %g' % (olg, 1.0 / nfb))
	plt.plot(x, y_open_loop, 'r', label='open-loop')
	plt.plot(x, y_tanh, label='tanh')
	plt.plot(x, y_ideal, label='Ideal opamp')
	plt.plot(x, y_interp_method, label='Interp tanh and ideal')
	plt.plot(x, y, label='Actual')
	plt.legend()
	plt.ylabel('Transfer Function')
	plt.xlim([-x_range, x_range])
	plt.grid()

	plt.subplot(3, 1, 2)
	plt.plot(x, gain_open_loop, 'r', label='open-loop')
	plt.plot(x, gain_tanh, label='tanh')
	plt.plot(x, gain_ideal, label='Ideal opamp')
	plt.plot(x, gain_interp, label='Interp tanh and ideal')
	plt.plot(x, gain_actual, label='Actual')
	plt.axhline(1.0 / nfb, label='Feedback gain')
	plt.legend()
	plt.ylabel('Slope (gain)')
	plt.xlim([-x_range, x_range])
	plt.ylim(gain_ylims)
	plt.grid()

	plt.subplot(3, 1, 3)
	plt.plot(x, n_iter, label='# iter')
	plt.ylabel('# iterations')
	plt.xlim([-x_range, x_range])
	plt.grid()

	plt.show()


def main(args):
	from matplotlib import pyplot as plt
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--olg', type=float, help='Open-loop gain, default 100')
	parser.add_argument('--fbg', type=float, help='Gain from negative feedback (i.e. reciprocal of actual feedback), default 10')
	args = parser.parse_args(args)

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

	# Open loop gain
	# Use a relatively low value (100) because:
	# * Graphs are easier to see
	# * It's a more difficult case for most methods - except:
	# * Original method takes > 100 iter to converge if olg is too high
	#
	# Note that several pieces of code here make the assumption olg * nfb >> 1, so can't go *too* low
	if args.olg is not None:
		olg = args.olg
	else:
		olg = 100.0

	# Negative feedback
	if args.fbg is not None:
		recip_nfb = args.fbg
	else:
		recip_nfb = 10.

	nfb = 1.0 / recip_nfb

	if olg * nfb <= 1.:
		raise ValueError('Feedback must be < open-loop gain')
	elif olg * nfb < 9:
		print('WARNING: feedback should be much smaller than open-loop')

	gain = _fb_gain_formula(olg, nfb)

	print('Open-loop gain: %g' % olg)
	print('Negative feedback: 1/%g' % recip_nfb)
	print('Resulting gain: %g' % gain)
	print('')

	n_samp = 20001
	x_range = 2.0
	#x_range = 0.5
	#x_range = 0.25

	x = np.linspace(-x_range, x_range, n_samp)
	x_half = x[:-1] + np.diff(x) / 2  # Halfway between samples

	def _opamp_x_dx(x):
		return olg * _tanh_x_dx(olg * x)

	def _opamp(pos, neg):
		return np.tanh(olg * (pos - neg))

	def _ideal_opamp(x):
		return _clip1(x * olg)

	def _fb(x):
		return \
			x * nfb

	def _fb_x_dx(x):
		return nfb

	def _iterate_over_inputs(f, x, n_iter_max=100, eps=1.0e-6, **kwargs):
		y = np.zeros_like(x)
		n_iter = np.zeros_like(x)
		for n, samp in enumerate(x):
			y[n], n_iter[n] = f(samp, n_iter_max=n_iter_max, eps=eps, **kwargs)

		if any(n_iter >= n_iter_max):
			print('WARNING: Failed to converge in %i iterations' % np.amax(n_iter_max))
		else:
			print('Iterations range: %u-%u, average %.2f' % (np.amin(n_iter), np.amax(n_iter), np.average(n_iter)))

		return y, n_iter

	def _original_iterate_1_samp(x, n_iter_max, eps):

		assert n_iter_max >= 1

		# Initial estimate
		y = _interp_method(x, olg, nfb, gain)

		# Limit the rate we can converge (one-pole lowpass)
		# This is part of the reason this method is so slow - even if we get the right value, we have to wait for this
		# lowpass filter to converge
		b0 = 1.0 / (olg * nfb)
		assert 0. < b0 < 1.
		na1 = 1.0 - b0

		for n in range(1, n_iter_max+1):
			y_prev = y
			neg_in = _fb(y)

			y_raw = _opamp(x, neg_in)  # theoretical y, pre-limiting
			y = y_raw*b0 + y_prev*na1

			if abs(y - y_prev) < eps:
				break

		return y, n

	def _bisect_1_samp(x, n_iter_max=100, eps=1.0e-6, interp=True):

		assert n_iter_max >= 1

		# y = tanh(olg * (x - nfb*y))
		#
		# So solve for:
		# olg*nfb*y + atanh(y) - olg*x = 0

		def f(y):
			return olg*nfb*y + np.arctanh(y) - olg*x

		if x == 0.:
			return 0., 0

		# Value is guaranteed between tanh and clip, so narrow down the search space to much smaller than (-1, 1)
		y_tanh = np.tanh(gain * x)
		y_clip = _clip1(gain * x)

		fy_tanh = f(y_tanh)
		fy_clip = f(utils.clip(y_clip, (eps - 1, 1 - eps)))  # arctanh is bounded in (-1, 1), so clip its input

		# Because we clip y for fy_clip, it's possible both are on the same side of zero - this should only happen if y
		# is very very close to +/- 1 anyway, so return that.
		#
		# Neither can be exactly 0 unless x was, and that case was handled above. So no need to handle < vs <=
		if (fy_tanh > 0) == (fy_clip > 0):
			return y_clip, 0

		if fy_tanh < 0:
			assert fy_clip > 0
			y_range = [y_tanh, y_clip]
			fy_range = [fy_tanh, fy_clip]

		else:
			assert fy_clip < 0
			y_range = [y_clip, y_tanh]
			fy_range = [fy_clip, fy_tanh]

		# First iteration, use interp method as initial estimate
		y = _interp_method_inner(olg, nfb, y_tanh, y_clip)

		fy = f(y)

		if fy < 0:
			y_range[0] = y
			fy_range[0] = fy
		elif fy > 0:
			y_range[1] = y
			fy_range[1] = fy
		else:
			return y, 1

		for n in range(2, n_iter_max+1):

			fy_prev = fy
			y_prev = y

			if interp:
				y = utils.scale(0, fy_range, y_range)
			else:
				y = 0.5 * sum(y_range)

			if abs(fy) > abs(fy_prev):
				raise Exception('Diverged: x=%g, n %u, f(%.10f)=%.10f -> f(%.10f)=%.10f' % (x, n, y_prev, fy_prev, y, fy))

			if abs(y - y_prev) < eps:
				break

			fy = f(y)

			if fy < 0:
				y_range[0] = y
				fy_range[0] = fy
			elif fy > 0:
				y_range[1] = y
				fy_range[1] = fy
			else:
				break

		return y, n

	def _combined(x, n_iter_max=100, eps=.10e-6):
		if abs(x) > 1.0 / gain:
			return _bisect_1_samp(x, n_iter_max, eps, interp=False)
		else:
			return _nr_iterate_1_samp(x, olg, nfb, gain, n_iter_max, eps, new_init_guess_method=True)

	print('Calculating original iterative method')
	y_iter_orig, n_iter_orig = _iterate_over_inputs(_original_iterate_1_samp, x)

	print('Calculating bisection method with interpolation')
	y_iter_bisect, n_iter_bisect_interp = _iterate_over_inputs(_bisect_1_samp, x, interp=True)

	print('Calculating bisection method with averaging')
	y_iter_bisect2, n_iter_bisect_avg = _iterate_over_inputs(_bisect_1_samp, x, interp=False)

	print('Calculating Newton-Raphson method')
	y_iter_nr, n_iter_nr = _iterate_over_inputs(_nr_iterate_1_samp, x, olg=olg, nfb=nfb, gain=gain)

	#print('Calculating combined method')
	#y_iter_combined, n_iter_combined = _iterate_over_inputs(_combined, x)

	print('Calculating other methods')
	y_tanh = np.tanh(x * gain)
	y_open_loop = _opamp(x, 0.0)
	y_interp_method = _interp_method(x, olg, nfb, gain)
	y_ideal = _clip1(x * gain)  # Ideal but with finite open-loop gain
	y_infinite_olg = _clip1(x / nfb)

	print('Plotting')

	plt.figure()

	plt.subplot(2, 1, 1)

	plt.title('Iterative solution; OL gain %g, FB gain %g' % (olg, 1.0 / nfb))
	plt.plot(x, y_iter_nr, label='Iterative (N-R)')
	plt.plot(x, y_tanh, label='tanh')
	plt.plot(x, y_ideal, label='Ideal opamp')
	plt.plot(x, y_interp_method, label='interp tanh and ideal')
	plt.plot(x, y_open_loop, label='open-loop')
	plt.legend()
	plt.ylabel('Transfer Function')
	plt.xlim([-x_range, x_range])
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.axvline(-1.0/gain)
	plt.axvline(1.0/gain)
	plt.plot(x, n_iter_orig, label='# iter orig')
	plt.plot(x, n_iter_bisect_interp, label='# iter bisect, interp')
	plt.plot(x, n_iter_bisect_avg, label='# iter bisect, average')
	plt.plot(x, n_iter_nr, label='# iter N-R')
	#plt.plot(x, n_iter_combined, label='# iter, combined method')
	plt.legend()
	plt.ylabel('# iterations')
	plt.xlim([-x_range, x_range])
	plt.grid()

	if False:

		# Calculate gain for these

		print('Calculating gains')

		gain_iter = np.gradient(y_iter_orig, x)
		gain_tanh = np.gradient(y_tanh, x)
		gain_ideal = np.gradient(y_ideal, x)

		# Here's the real meat of all this: try to do this non-iteratively

		y_ol = np.tanh(olg * x)
		gain_ol = _tanh_x_dx(olg * x) * olg
		gain_ol_sanity = np.gradient(y_ol, x)
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
		plt.plot(x, y_iter_orig, label='Iterative')
		plt.legend()
		plt.ylabel('Transfer Function')
		plt.xlim([-x_range, x_range])
		plt.grid()

		plt.subplot(2, 1, 2)
		plt.plot(x, gain_iter, label='Iterative')
		plt.plot(x, gain_tanh, label='tanh')
		plt.plot(x, gain_ideal, label='Ideal opamp but finite open-loop gain')
		plt.plot(x, gain_non_iter, label='Non-iterative')
		plt.ylabel('Gain')
		plt.legend()
		plt.xlim([-x_range, x_range])
		plt.grid()

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
			plt.plot(x, y_ol_gain, label='Open loop')
			plt.plot([-2, 2], [1 / nfb, 1 / nfb], label='FB only')
			plt.plot(x, y_gain, label='Both')
			plt.legend()
			plt.title('Gain')
			plt.grid()

	plt.show()
