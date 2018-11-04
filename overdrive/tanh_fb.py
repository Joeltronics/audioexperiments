#!/usr/bin/env python3


import numpy as np
from scipy import integrate
from utils import utils
from typing import Union, Tuple


"""
A long-tailed transistor pair is used as an input stage for nearly all opamps and OTAs, and is typically one of the most
significant nonlinearities. The large-signal behavior of an ideal long-tailed pair follows the formula:

    y = tanh(g_ol * (x_pos - x_neg)) 

Opamps are commonly used in a negative feedback configuration, where the output is connected to the negative input -
either directly, in order to act as a unity-gain buffer, or through some sort of gain reduction stage (typically a
voltage divider), in which case it acts as an amplifier where the small-signal gain is given by the formula:

    gain = g_ol / (1 + g_ol * g_fb)

For sufficiently high open loop gain, the resulting system gain can be approximated as:
 
    gain = 1 + 1/g_fb

In this case, the large-signal formula for the opamp (simulating only the tanh nonlinearity) is:

    y = tanh(g_ol * (x - g_fb * y))

y = f(x) cannot be solved analytically; the inverse of this equation, x = f(y), can:

    x = g_fb * y + atanh(y) / g_ol

The goal of this module is to provide a numerical solution to y = f(x)
"""


def _clip1(x):
	"""Clip to range [-1, 1]"""
	return np.clip(x, -1., 1.)


def _tanh_x_dx(x):
	"""
	d/dx tanh(x) = 1 - tanh(x)^2
	"""
	return 1. - np.square(np.tanh(x))


def _atanh_x_dx(x):
	"""
	d/dx atanh(x) = 1 / (1 + x^2)
	"""
	return 1. / (1. - np.square(x))


def _fb_gain_formula(open_loop_gain: float, neg_feedback: float) -> float:
	"""
	Calculate gain based on open-loop gain and negative feedback

	:param open_loop_gain: open-loop gain
	:param neg_feedback: positive value, < 1
	:return: resulting gain
	"""
	return open_loop_gain / (1.0 + open_loop_gain * neg_feedback)


def _get_x_calculation_range(open_loop_gain: float, neg_feedback: float, eps=utils.from_dB(160.)):
	"""
	Calculate the maximum value of abs(x) where tanh_fb(x) will return a value that is not within epsilon of +/- 1
	i.e. the x range within which a calculation is actually necessary

	:param open_loop_gain: open loop gain
	:param neg_feedback: positive value, < 1
	:param eps: epsilon value
	:return:
	"""
	return inverse_tanh_fb(1.0 - eps, open_loop_gain, neg_feedback)


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


def _nr_iterate_1_samp(x, olg, nfb, gain=None, n_iter_max=100, eps=1.0e-6, x_range=None) -> Tuple[float, int]:
	"""
	Iterate Newton-Raphson method (for single input sample)

	:param x: input value
	:param olg: open-loop gain
	:param nfb: negative feedback (positive value < 1)
	:param gain: optional precalculated gain from _fb_gain_formula
	:param n_iter_max: Max number of iterations
	:param eps: maximum error
	:param x_range: optional precalculated x range from _get_x_calculation_range()
	"""

	assert n_iter_max >= 1

	if x == 0.:
		return 0., 0

	if x_range is None:
		x_range = _get_x_calculation_range(olg, nfb, eps=eps)

	if x > x_range:
		return 1., 0
	elif x < -x_range:
		return -1., 0

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

		# TODO: is this even possible anymore now that input outside of x_range is clipped?

		# If residue was smaller than eps, then just clip to (-1, 1), don't worry about next NR estimate
		if abs(r) < eps:
			y = _clip1(y)
			break

		# Otherwise, residue was still too large but we're out of bounds
		# In this case, go almost all the way to +/- 1
		if abs(y) >= 1.:
			y = utils.lerp((y_prev, 1. if y > 0 else -1.), 1. - eps)

	return y, n


def tanh_fb(x: Union[float, np.ndarray], open_loop_gain: float, neg_feedback: float, eps=1.e-6) -> Union[float, np.ndarray]:
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

	x_range = _get_x_calculation_range(open_loop_gain, neg_feedback, eps=eps)

	if np.isscalar(x):
		y, _ = _nr_iterate_1_samp(x, open_loop_gain, neg_feedback, gain, eps=eps, x_range=x_range)
	else:
		y = np.zeros_like(x)
		for n, xx in enumerate(y):
			y[n], _ = _nr_iterate_1_samp(x, open_loop_gain, neg_feedback, gain, eps=eps, x_range=x_range)

	return y


def inverse_tanh_fb(y: Union[float, np.ndarray], open_loop_gain: float, neg_feedback: float) -> Union[float, np.ndarray]:
	"""
	Solves inverse of tanh feedback
	i.e. solves x = f(y), for y = tanh(olg * (x - nfb*y))

	:param open_loop_gain: open-loop gain
	:param neg_feedback: positive value, < 1
	:return:
	"""

	if neg_feedback < 0:
		raise ValueError('Negative feedback must be a positive value (yeah, you read that right)')
	elif open_loop_gain == 0:
		raise ValueError('Open-loop gain cannot be negative')

	return neg_feedback * y + np.arctanh(y) / open_loop_gain


def _parse_args(args):
	import argparse

	# Open loop gain
	# Use a relatively low value (100) because:
	# * Graphs are easier to see
	# * It's a more difficult case for most methods - except:
	# * Original method takes > 100 iter to converge if olg is too high
	#
	# Note that several pieces of code here make the assumption olg * nfb >> 1, so can't go *too* low
	default_olg = 100
	default_fbg = 10

	default_eps_dB = -120

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--olg', type=float, default=float(default_olg),
		help='Open-loop gain, default %i' % default_olg)
	parser.add_argument(
		'--fbg', type=float, default=float(default_fbg),
		help='Gain from negative feedback (i.e. reciprocal of actual feedback), default %i' % default_fbg)
	parser.add_argument(
		'--precision', type=float, default=float(default_eps_dB),
		help='Precision, in dB, default %i' % default_eps_dB)
	parser.add_argument('--xrange', type=float, default=0.5, help='Range to plot')
	parser.add_argument('--nsamp', type=int, default=20001, help='# samples to plot')
	args = parser.parse_args(args)

	if args.fgb >= 0.:
		raise ValueError('Feedback amount must be positive')

	if args.olg / args.fbg <= 1.:
		raise ValueError('Feedback must be < open-loop gain')
	elif args.olg / args.fbg < 9:
		print('WARNING: feedback should be much smaller than open-loop')

	if args.precision >= 0.:
		raise ValueError('precision must be negative dB value')

	return args


def plot(args):
	from matplotlib import pyplot as plt

	args = _parse_args(args)

	olg = args.olg
	nfb = 1.0 / args.fbg
	eps_dB = args.precision

	gain = _fb_gain_formula(olg, nfb)
	eps = utils.from_dB(eps_dB)

	n_samp = args.nsamp
	x_range = args.xrange

	n_iter_max = 100

	x = np.linspace(-x_range, x_range, n_samp)

	y = np.zeros_like(x)
	n_iter = np.zeros_like(x)
	for n, samp in enumerate(x):
		y[n], n_iter[n] = _nr_iterate_1_samp(samp, olg, nfb, gain, n_iter_max=n_iter_max, eps=eps)

	if any(n_iter >= n_iter_max):
		print('WARNING: Failed to converge in %i iterations' % np.amax(n_iter_max))

	print('Calculating other methods')
	y_tanh = np.tanh(x * gain)
	y_open_loop = np.tanh(x * olg)
	y_interp_method = _interp_method(x, olg, nfb, gain)
	y_ideal_finite_olg = _clip1(x * gain)
	y_ideal_infinite_olg = _clip1(x / nfb)

	gain_tanh = np.gradient(y_tanh, x)
	gain_ideal_finite_olg = np.gradient(y_ideal_finite_olg, x)
	gain_ideal_infinite_olg = np.gradient(y_ideal_infinite_olg, x)
	gain_interp = np.gradient(y_interp_method, x)
	gain_open_loop = np.gradient(y_open_loop, x)
	gain_actual = np.gradient(y, x)

	gain_ylims = [-0.1/nfb, 1.1/nfb]

	print('Plotting')

	plt.figure()

	plt.subplot(3, 1, 1)

	plt.title('Iterative solution; OL gain %g, FB gain %g, precision %i dB' % (olg, 1.0 / nfb, eps_dB))
	plt.plot(x, y_open_loop, 'r', label='open-loop')
	plt.plot(x, y_tanh, label='tanh')
	plt.plot(x, y_ideal_finite_olg, label='Ideal opamp (finite OLG)')
	plt.plot(x, y_ideal_infinite_olg, label='Ideal opamp (infinite OLG)')
	plt.plot(x, y_interp_method, label='Interp tanh and ideal')
	plt.plot(x, y, label='Actual')
	plt.legend()
	plt.ylabel('Transfer Function')
	plt.xlim([-x_range, x_range])
	plt.grid()

	plt.subplot(3, 1, 2)
	plt.plot(x, gain_open_loop, 'r', label='open-loop')
	plt.plot(x, gain_tanh, label='tanh')
	plt.plot(x, gain_ideal_finite_olg, label='Ideal opamp (finite OLG)')
	plt.plot(x, gain_ideal_infinite_olg, label='Ideal opamp (infinite OLG)')
	plt.plot(x, gain_interp, label='Interp tanh and ideal')
	plt.plot(x, gain_actual, label='Actual')
	plt.axhline(1.0 / nfb, label='Feedback gain')
	plt.legend()
	plt.ylabel('Slope (small-signal gain)')
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

	# Open loop gain
	# Use a relatively low value (100) because:
	# * Graphs are easier to see
	# * It's a more difficult case for most methods - except:
	# * Original method takes > 100 iter to converge if olg is too high
	#
	# Note that several pieces of code here make the assumption olg * nfb >> 1, so can't go *too* low
	default_olg = 100
	default_fbg = 10

	default_eps_dB = -120

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--olg', type=float, default=float(default_olg),
		help='Open-loop gain, default %i' % default_olg)
	parser.add_argument(
		'--fbg', type=float, default=float(default_fbg),
		help='Gain from negative feedback (i.e. reciprocal of actual feedback), default %i' % default_fbg)
	parser.add_argument(
		'--precision', type=float, default=float(default_eps_dB),
		help='Precision, in dB, default %i' % default_eps_dB)
	parser.add_argument('--xrange', type=float, default=0.5, help='Range to plot')
	parser.add_argument('--nsamp', type=int, default=20001, help='# samples to plot')
	args = parser.parse_args(args)

	"""
	def sanity_check(x):
		y = np.tanh(x)
		dydx = _tanh_x_dx(x)
		plt.figure()
		plt.plot(x, y, label='tanh(x)')
		plt.plot(x, dydx, label='d/dx tanh(x)')
		plt.grid()
		plt.title('Sanity check d/dx tanh(x)')
		plt.legend()

	sanity_check(np.linspace(-10, 10, 1001))
	"""

	olg = args.olg
	recip_nfb = args.fbg
	nfb = 1.0 / recip_nfb

	eps_dB = args.precision

	n_samp = args.nsamp
	x_range = args.xrange

	if olg * nfb <= 1.:
		raise ValueError('Feedback must be < open-loop gain')
	elif olg * nfb < 9:
		print('WARNING: feedback should be much smaller than open-loop')

	if eps_dB >= 0.:
		raise ValueError('precision must be negative dB value')

	gain = _fb_gain_formula(olg, nfb)
	eps = utils.from_dB(eps_dB)

	print('Open-loop gain: %g' % olg)
	print('Negative feedback: 1/%g' % recip_nfb)
	print('Resulting gain: %g' % gain)
	print('')

	x_calc_range = _get_x_calculation_range(olg, nfb, eps=eps)
	y_calc_range = 1.0 - eps

	y_inv = np.linspace(-y_calc_range, y_calc_range, n_samp)
	x_inv = inverse_tanh_fb(y_inv, olg, nfb)

	assert utils.approx_equal(x_calc_range, np.amax(np.abs(x_inv)))

	print('Precision: %g dB = %g' % (eps_dB, eps))
	print('x range where y != +/- 1: +/- %f' % x_calc_range)
	print('')

	x = np.linspace(-x_range, x_range, n_samp)
	x_half = x[:-1] + np.diff(x) / 2  # Halfway between samples

	def _opamp_x_dx(x):
		return olg * _tanh_x_dx(olg * x)

	def _opamp(pos, neg):
		return np.tanh(olg * (pos - neg))

	def _ideal_opamp(x):
		return _clip1(x * olg)

	def _fb(x):
		return x * nfb

	def _fb_x_dx(x):
		return nfb

	def _iterate_over_inputs(f, x, n_iter_max=100, eps=eps, **kwargs):
		y = np.zeros_like(x)
		n_iter = np.zeros_like(x)
		for n, samp in enumerate(x):
			y[n], n_iter[n] = f(samp, n_iter_max=n_iter_max, eps=eps, **kwargs)

		if any(n_iter >= n_iter_max):
			print('WARNING: Failed to converge in %i iterations' % np.amax(n_iter_max))
		else:
			print('Iterations range: %u-%u, average %.2f' % (np.amin(n_iter), np.amax(n_iter), np.average(n_iter)))

		return y, n_iter

	def _naive_iterate_1_samp(x, n_iter_max, eps=eps):

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

	def _bisect_1_samp(x, n_iter_max=100, eps=eps, interp=True):

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

	print('Calculating naive iteration')
	y_iter_naive, n_iter_naive = _iterate_over_inputs(_naive_iterate_1_samp, x)

	print('Calculating bisection method with interpolation')
	y_iter_bisect, n_iter_bisect_interp = _iterate_over_inputs(_bisect_1_samp, x, interp=True)

	print('Calculating bisection method with averaging')
	y_iter_bisect2, n_iter_bisect_avg = _iterate_over_inputs(_bisect_1_samp, x, interp=False)

	print('Calculating Newton-Raphson method')
	y_iter_nr, n_iter_nr = _iterate_over_inputs(_nr_iterate_1_samp, x, olg=olg, nfb=nfb, gain=gain)

	print('Calculating other methods')
	y_tanh = np.tanh(x * gain)
	y_open_loop = _opamp(x, 0.0)
	y_interp_method = _interp_method(x, olg, nfb, gain)
	y_ideal_finite_olg = _clip1(x * gain)  # Ideal but with finite open-loop gain
	y_ideal_infinite_olg = _clip1(x * recip_nfb)

	print('Plotting')

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.title('Actual (based on inverse)')
	plt.plot(x_inv, y_inv)
	plt.grid()
	plt.ylabel('Transfer function')

	plt.subplot(2, 1, 2)
	plt.plot(x_inv, np.gradient(y_inv, x_inv))
	plt.grid()
	plt.ylabel('Slope (small-signal gain)')

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.title('OL gain %g, FB gain %g' % (olg, 1.0 / nfb))
	plt.plot(x, y_open_loop, label='open-loop')
	plt.plot(x_inv, y_inv, label='Actual')
	plt.plot(x, y_tanh, label='tanh')
	plt.plot(x, y_ideal_finite_olg, label='Ideal opamp (finite OLG)')
	plt.plot(x, y_ideal_infinite_olg, label='Ideal opamp (infinite OLG)')
	plt.plot(x, y_interp_method, label='interp tanh and ideal')
	plt.legend()
	plt.xlim([-x_range, x_range])
	plt.grid()
	plt.ylabel('Transfer Function')

	plt.subplot(2, 1, 2)
	plt.plot(x, np.gradient(y_open_loop, x), label='open-loop')
	plt.plot(x_inv, np.gradient(y_inv, x_inv), label='Actual')
	plt.plot(x, np.gradient(y_tanh, x), label='tanh')
	plt.plot(x, np.gradient(y_ideal_finite_olg, x), label='Ideal opamp (finite OLG)')
	plt.plot(x, np.gradient(y_ideal_infinite_olg, x), label='Ideal opamp (infinite OLG)')
	plt.plot(x, np.gradient(y_interp_method, x), label='interp tanh and ideal')
	plt.xlim([-x_range, x_range])
	plt.grid()
	plt.legend()
	plt.ylabel('Slope (small-signal gain)')
	plt.ylim([-0.1 / nfb, 1.1 / nfb])

	plt.figure()

	#plt.subplot(2, 1, 1)
	plt.title('Iterative solution; OL gain %g, FB gain %g, precision %g dB' % (olg, 1.0 / nfb, eps_dB))
	# TODO: plot error here
	#plt.plot(x_inv, y_inv, label='Actual')
	#plt.legend()
	#plt.ylabel('Transfer Function')
	#plt.xlim([-x_range, x_range])
	#plt.grid()

	#plt.subplot(2, 1, 2)
	plt.plot(x, n_iter_naive, label='# iter naive')
	plt.plot(x, n_iter_bisect_interp, label='# iter bisect, interp')
	plt.plot(x, n_iter_bisect_avg, label='# iter bisect, average')
	plt.plot(x, n_iter_nr, label='# iter N-R')
	plt.legend()
	plt.ylabel('# iterations')
	plt.xlim([-x_range, x_range])
	plt.grid()

	if False:

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
		plt.plot(x, y_iter_naive, label='Iterative')
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
