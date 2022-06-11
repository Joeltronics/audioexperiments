#!/usr/bin/env python3

from typing import Tuple, Iterable, Optional, Callable, Union

import scipy.optimize

from solvers.iter_stats import IterStats
from utils import utils


DEFAULT_EPS = 1.0e-6


def solve_fb_iterative(
		f: Callable[[float], float],
		estimate: float,
		rate_limit: Optional[float]=None,
		eps: Optional[float]=DEFAULT_EPS,
		max_num_iter=100,
		throw_if_failed_converge=True,
		return_vector=False,
		iter_stats: Optional[IterStats]=None,
		verbose=False,
) -> Union[float, Iterable[float]]:
	"""
	Solves f(y) = y

	:param f: function of y to solve
	:param estimate: initial estimate
	:param rate_limit: value 0-1, typically << 1; if given, apply lowpass filter to estimates
	:param eps: max absolute error; if None, will continue calculating until max_num_iter is reached
	:param max_num_iter: Max number of iterations
	:param throw_if_failed_converge: if True, will throw if fails to converge in max_num_iter (unless eps is None)
	:param return_vector: if true, returns vector of all iterated values
	:return: x, number of iterations; or vector of all iterations if return_vector is True
	"""

	if max_num_iter < 1:
		raise ValueError('max_num_iter must be at least 1')

	y = estimate
	yv = [] if return_vector else None

	if rate_limit:
		b0 = rate_limit
		na1 = 1.0 - b0
	else:
		b0 = 1.
		na1 = 0.

	errs = [] if iter_stats is not None else None

	n = 0
	success = False
	for n in range(1, max_num_iter + 1):

		if yv is not None:
			yv.append(y)

		y_prev = y
		y = f(y)

		err = abs(y - y_prev)
		if errs is not None:
			errs.append(err)

		if eps and err < eps:
			success = True
			break

		if verbose:
			print('iter %i: f(%g) = %g, err %g' % (n, y_prev, y, err))

		if rate_limit:
			y = y*b0 + y_prev*na1

	else:
		if throw_if_failed_converge and eps:
			raise RuntimeError('Failed to converge in %i iterations' % n)

	if iter_stats is not None:
		iter_stats.add(
			success=success,
			est=estimate,
			n_iter= n + 1,
			final=y,
			err=errs
		)

	if yv is not None:
		return yv
	else:
		return y


def solve_iterative(
		f: Callable[[float], float],
		estimate: float,
		rate_limit: Optional[float]=None,
		eps: Optional[float]=DEFAULT_EPS,
		max_num_iter=100,
		throw_if_failed_converge=True,
		return_vector=False,
		iter_stats: Optional[IterStats]=None,
		verbose=False,
) -> Union[float, Iterable[float]]:
	"""
	Solves f(x) = 0

	:param f: function of x to solve
	:param estimate: initial estimate
	:param rate_limit: value 0-1, typically << 1; if given, apply lowpass filter to estimates
	:param eps: max absolute error; if None, will continue calculating until max_num_iter is reached
	:param max_num_iter: Max number of iterations
	:param throw_if_failed_converge: if True, will throw if fails to converge in max_num_iter (unless eps is None)
	:param return_vector: if true, returns vector of all iterated values
	:return: x, number of iterations; or vector of all iterations if return_vector is True
	"""

	"""
	solve_fb_iterative needs to be of form f(x) = x

	f(x) = 0
	f(x) + x = x
	
	define: g(x) = f(x) + x

	Now solve for:
	g(x) = x
	"""

	def g(x):
		return f(x) + x

	return solve_fb_iterative(
		f=g,
		estimate=estimate,
		rate_limit=rate_limit,
		eps=eps,
		max_num_iter=max_num_iter,
		throw_if_failed_converge=throw_if_failed_converge,
		return_vector=return_vector,
		iter_stats=iter_stats,
		verbose=verbose)


def solve_bisection(
		f: Callable[[float], float],
		init_range: Tuple[float, float],
		estimate: Optional[float]=None,
		interp=True,
		eps: Optional[float]=DEFAULT_EPS,
		max_num_iter=100,
		throw_if_failed_converge=True,
		return_vector=False) -> Union[Tuple[float, int], Iterable[float]]:
	"""
	Solve f(x) = 0 by bisection

	:param f:
	:param init_range: initial range
	:param estimate: initial estimate
	:param interp: if true, interpolates (regula falsi); if false, performs naive bisection (average of 2 values)
	:param eps: max absolute error; if None, will continue calculating until max_num_iter is reached
	:param max_num_iter: Max number of iterations
	:param throw_if_failed_converge: if True, will throw if fails to converge in max_num_iter (unless eps is None)
	:param return_vector: if true, returns vector of all iterated values
	:return: x, number of iterations; or vector of all iterations if return_vector is True
	"""

	if max_num_iter < 1:
		raise ValueError('max_num_iter must be at least 1')

	xv = [] if return_vector else None

	xmin, xmax = init_range
	ymin = f(xmin)
	ymax = f(xmax)

	if (ymax > 0) == (ymin > 0):
		raise Exception('Invalid input range')

	if ymin > ymax:
		# Swap
		ymin, ymax = ymax, ymin
		xmin, xmax = xmax, xmin

	n = 0
	for n in range(1, max_num_iter + 1):

		if (n == 1) and (estimate is not None):
			x = estimate
		elif interp:
			x = utils.scale(0, (ymin, ymax), (xmin, xmax))
		else:
			x = 0.5 * (xmin + xmax)

		if xv is not None:
			xv.append(x)

		y = f(x)

		if eps and (abs(y) < eps):
			break
		
		if y < 0:
			xmin, ymin = x, y
		else:
			xmax, ymax = x, y

	else:
		if throw_if_failed_converge and eps:
			raise RuntimeError('Failed to converge in %i iterations' % n)

	if xv is not None:
		return xv
	else:
		return x, n


def solve_nr(
		f: Callable[[float], float],
		df: Callable[[float], float],
		estimate: float,
		eps: Optional[float]=DEFAULT_EPS,
		max_num_iter=100,
		throw_if_failed_converge=True,
		return_vector=False,
		iter_stats: Optional[IterStats]=None,
		limit_range: Optional[Tuple[float, float]]=None,
		clip_to_limit_range=False,
) -> Union[float, Iterable[float]]:
	"""
	Solves f(x) = 0 using Newton-Raphson method

	:param f: function of x to solve
	:param df: derivative of f(x)
	:param estimate: initial estimate
	:param eps: max absolute error; if None, will continue calculating until max_num_iter is reached
	:param max_num_iter: Max number of iterations
	:param throw_if_failed_converge: if True, will throw if fails to converge in max_num_iter (unless eps is None)
	:param return_vector: if true, returns vector of all iterated values
	:return: x, number of iterations; or vector of all iterations if return_vector is True
	"""

	if max_num_iter < 1:
		raise ValueError('max_num_iter must be at least 1')

	x = estimate
	xv = [] if return_vector else None

	if (limit_range is not None) and not (limit_range[0] <= estimate <= limit_range[1]):
		raise ValueError('Estimate out of limit_range')

	errs = [] if iter_stats is not None else None

	n = 0
	success = False
	iter_num = -1
	for iter_num in range(1, max_num_iter + 1):

		if xv is not None:
			xv.append(x)

		fx = f(x)
		dfx = df(x)

		residue = fx / dfx
		x_prev = x
		x -= residue

		if (limit_range is not None) and not (limit_range[0] <= x <= limit_range[1]):
			if not clip_to_limit_range:
				raise RuntimeError('Failed to converge - value out of limit_range, and clip_to_limit_range not set')

			x = utils.clip(x, limit_range)
			if x == x_prev:
				raise RuntimeError('Failed to converge - clip_to_limit_range failed, returned out of range twice in a row')

		if errs is not None:
			errs.append(abs(residue))

		if eps and (abs(residue) < eps):
			success = True
			break
	else:
		if throw_if_failed_converge and eps:
			raise RuntimeError('Failed to converge in %i iterations' % iter_num)

	if iter_stats is not None:
		iter_stats.add(
			success=success,
			est=estimate,
			n_iter=iter_num + 1,
			final=x,
			err=errs
		)

	if xv is not None:
		return xv
	else:
		return x


def solve_bisection_then_nr(
		f: Callable[[float], float],
		df: Callable[[float], float],
		init_range: Tuple[float, float],
		eps: Optional[float]=DEFAULT_EPS,
		max_num_iter=100,
		throw_if_failed_converge=True,
		require_in_range=False,
		return_vector=False) -> Union[Tuple[float, int], Iterable[float]]:

	if max_num_iter < 1:
		raise ValueError('max_num_iter must be at least 1')

	xmin, xmax = init_range
	ymin = f(xmin)
	ymax = f(xmax)

	if require_in_range and (ymax > 0) == (ymin > 0):
		raise ValueError('Invalid input range')

	# TODO: since we have a derivative function, try using spline interpolation

	# If not require_in_range, then clip=True will cause using the one closer to zero as the estimate
	x = utils.scale(0, (ymin, ymax), (xmin, xmax), clip=True)

	return solve_nr(
		f=f,
		df=df,
		estimate=x,
		eps=eps,
		max_num_iter=(max_num_iter - 1),
		throw_if_failed_converge=throw_if_failed_converge,
		return_vector=return_vector,
		limit_range=(init_range if require_in_range else None),

		# If require_in_range, then we've already guaranteed result is bracketed by range, so fail if estimate outside
		# If not, then this value is unused anyway
		clip_to_limit_range=False,
	)


# TODO: try Brent's method


def plot(args):
	from matplotlib import pyplot as plt
	import numpy as np

	"""
	y = (x + 1)(x - 1)(x - 4) = (x^2 - 1)(x - 4) = x^3 - 4x^2 - x + 4
	dy/dx = 3x^2 - 8x - 1
	"""

	def plot_nr(f, df, range, estimates, title: str):

		fig = plt.figure()
		plt.suptitle('Newton-Raphson')

		plt.title(title)

		x = np.linspace(range[0], range[1], 1024)
		y = f(x)
		plt.plot(x, y, label='Actual', zorder=99)

		for estimate in estimates:
			xv = solve_nr(f, df, estimate=estimate, return_vector=True)

			x_plot = []
			y_plot = []
			for x_val in xv:

				y_val = f(x_val)

				x_plot.append(x_val)
				x_plot.append(x_val)

				y_plot.append(0.0)
				y_plot.append(y_val)

			plt.plot(x_plot, y_plot, label=f'est={estimate:g} ({len(xv)} iter)')

		plt.legend()
		plt.grid()

	def plot_bi_nr(f, df, range, title: str):
		fig = plt.figure()
		plt.suptitle('Regula Falsi then Newton-Raphson')

		plt.title(title)

		x = np.linspace(range[0], range[1], 1024)
		y = f(x)
		plt.plot(x, y, label='Actual', zorder=99)

		y_min = f(range[0])
		y_max = f(range[1])
		x0 = utils.scale(0, (y_min, y_max), range)
		plt.plot([range[0], x0, range[1]], [y_min, 0, y_max], '.-', label='Regula falsi')

		xv = solve_bisection_then_nr(f, df, init_range=range, return_vector=True)

		x_plot = []
		y_plot = []
		for x_val in xv:

			y_val = f(x_val)

			x_plot.append(x_val)
			x_plot.append(x_val)

			y_plot.append(0.0)
			y_plot.append(y_val)

		plt.plot(x_plot, y_plot, label=f'range=[{range[0]:g}, {range[1]:g}] ({len(xv)} iter)')

		plt.legend()
		plt.grid()

	def f(x):
		return (x ** 3.) - 4.*(x ** 2.) - x + 4.

	def df(x):
		return 3.*(x ** 2.) - 8.*x - 1.
	
	estimates = [-3, -1.1, 1.1, 2., 5]
	plot_nr(f=f, df=df, range=[-3, 5], estimates=estimates, title='f(x) = (x + 1)(x - 1)(x - 4)')

	def f(x):
		return np.tanh(x) - 0.9
	
	def df(x):
		return (np.cosh(x)) ** -2.0

	estimates = [1.472, 1.5, 2., 1., 0., -0.9]
	plot_nr(f=f, df=df, range=[-1, 2], estimates=estimates, title='f(x) = tanh(x) - 0.9')
	plot_bi_nr(f=f, df=df, range=[-2, 2], title='f(x) = tanh(x) - 0.9')

	def f(x):
		return np.arctanh(x) - 0.9
	
	def df(x):
		return 1.0 / (1.0 - (x ** 2.0))

	estimates = [0.75, 0, 0.9999, -0.9999]
	plot_nr(f=f, df=df, range=[-0.9999, 0.9999], estimates=estimates, title='f(x) = atanh(x) - 0.9')
	plot_bi_nr(f=f, df=df, range=[-0.9999, 0.9999], title='f(x) = atanh(x) - 0.9')

	plt.show()


def main(args):
	import math
	import time

	"""
	y = (x + 1)(x - 1)(x - 4) = (x^2 - 1)(x - 4) = x^3 - 4x^2 - x + 4
	dy/dx = 3x^2 - 8x - 1
	"""

	def f(x):
		return (x ** 3.) - 4.*(x ** 2.) - x + 4.

	def df(x):
		return 3.*(x ** 2.) - 8.*x - 1.

	print('')
	print('f(x) = (x + 1)(x - 1)(x - 4) = x^3 - 4x^2 - x + 4')
	print('d/dx f(x) = 3x^2 - 8x - 1')
	print('')
	print('Roots: -1, 1, 4')

	print('')
	est = 0.75
	rate_limit = 0.1
	print('Solving iterative, est %g, rate limit %g' % (est, rate_limit))
	xv = solve_iterative(f, estimate=est, rate_limit=rate_limit, return_vector=True)
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	rate_limit = 0.25
	print('Solving iterative, est %g, rate limit %g' % (est, rate_limit))
	xv = solve_iterative(f, estimate=est, rate_limit=rate_limit, return_vector=True)
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	rate_limit = 0.05
	print('Solving iterative, est %g, rate limit %g' % (est, rate_limit))
	xv = solve_iterative(f, estimate=est, rate_limit=rate_limit, return_vector=True)
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	print('')
	range = (-0.9, 3.75)
	est = 0.75
	print('True bisection, init range: x = (%g, %g), y = (%g, %g)' % (range[0], range[1], f(range[0]), f(range[1])))
	xv = solve_bisection(f, init_range=range, interp=False, return_vector=True)
	print('%i iterations:' % len(xv))
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))
	
	print()
	print('True bisection, init range: x = (%g, %g), y = (%g, %g), est = %g' % (range[0], range[1], f(range[0]), f(range[1]), est))
	xv = solve_bisection(f, estimate=est, init_range=range, interp=False, return_vector=True)
	print('%i iterations:' % len(xv))
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	print()
	print('Interpolating bisection, init range: x = (%g, %g), y = (%g, %g)' % (range[0], range[1], f(range[0]), f(range[1])))
	xv = solve_bisection(f, init_range=range, interp=True, return_vector=True)
	print('%i iterations:' % len(xv))
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	print()
	print('Interpolating bisection, init range: x = (%g, %g), y = (%g, %g), est = %g' % (range[0], range[1], f(range[0]), f(range[1]), est))
	xv = solve_bisection(f, estimate=est, init_range=range, interp=True, return_vector=True)
	print('%i iterations:' % len(xv))
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	print('')
	range = (1.5, 7.)
	est = 0.5 * (range[0] + range[1])
	print('Interpolating bisection, init range: x = (%g, %g), y = (%g, %g)' % (range[0], range[1], f(range[0]), f(range[1])))
	xv = solve_bisection(f, init_range=range, interp=True, return_vector=True)
	print('%i iterations:' % len(xv))
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	range = (-0.9, 3.75)
	print()
	print('scipy.optimize.brentq, init range: x = (%g, %g), y = (%g, %g)' % (range[0], range[1], f(range[0]), f(range[1])))
	x, r = scipy.optimize.brentq(f, *range, full_output=True, xtol=DEFAULT_EPS, rtol=DEFAULT_EPS)
	print('Result %.12f, %i iterations, %i function calls' % (
		x,
		r.iterations,
		r.function_calls,
	))

	range = (1.5, 7.)
	print()
	print('scipy.optimize.brentq, init range: x = (%g, %g), y = (%g, %g)' % (range[0], range[1], f(range[0]), f(range[1])))
	x, r = scipy.optimize.brentq(f, *range, full_output=True, xtol=DEFAULT_EPS, rtol=DEFAULT_EPS)
	print('Result %.12f, %i iterations, %i function calls' % (
		x,
		r.iterations,
		r.function_calls,
	))

	print('')
	print('Newton-Raphson:')
	for est in [-10, -1.1, 1.1, 2., 10]:
		print('')
		print('Estimate %g' % est)

		start = time.time()
		xv = solve_nr(f, df, estimate=est, return_vector=True)
		duration = time.time() - start
		print('Duration: %s' % duration)
		print('%i iterations:' % len(xv))
		for n, x in enumerate(xv):
			print('x[%i] = %.12f' % (n, x))

		start = time.time()
		x, r = scipy.optimize.newton(func=f, x0=est, fprime=df, rtol=DEFAULT_EPS, full_output=True)
		duration = time.time() - start
		print('With scipy.optimize.newton:')
		print('Duration: %s' % duration)
		print('%i iterations, %i function calls' % (r.iterations, r.function_calls))

	print()
	print('y = tanh(x)')
	print('dy/dx = sech(x)^2 = cosh(x)^-2')
	print()
	print('Solving for y=0.9')
	print('Solution is approx 1.472')

	def f(x):
		return math.tanh(x) - 0.9

	def df(x):
		return (math.cosh(x)) ** -2.0

	print()
	print('Newton-Raphson:')
	# Fails to converge if estimate is too far off (e.g. -1 or 3)
	for estimate in [1.472, 1.5, 2., 1., 0., -0.9, 2.5]:
		print()
		print(f'Estimate {estimate}')
		start = time.time()
		xv = solve_nr(f, df, estimate=estimate, return_vector=True)
		duration = time.time() - start
		print('Duration: %s' % duration)
		print(f'{len(xv)} iterations:')
		for n, x in enumerate(xv):
			print(f'x[{n}] = {x:.12f}')
		
		start = time.time()
		x, r = scipy.optimize.newton(func=f, x0=estimate, fprime=df, rtol=DEFAULT_EPS, full_output=True)
		duration = time.time() - start
		print('With scipy.optimize.newton:')
		print('Duration: %s' % duration)
		print('%i iterations, %i function calls' % (r.iterations, r.function_calls))

	print()
	print('Regula falsi then NR, range +/- 2:')
	xv = solve_bisection_then_nr(f, df, (-2, 2), return_vector=True)
	for n, x in enumerate(xv):
		print(f'x[{n}] = {x:.12f}')

	print()
	print('y = atanh(x)')
	print('dy/dx = 1 / (1 - x^2')
	print()
	print('Solving for y=0.9')
	print('Solution is approx 0.716')

	def f(x):
		return math.atanh(x) - 0.9

	def df(x):
		return 1.0 / (1.0 - (x ** 2.0))

	print()
	print('Newton-Raphson:')
	for estimate in [0.75, 0, 0.99999, -0.99999]:
		print()
		print(f'Estimate {estimate}')
		start = time.time()
		xv = solve_nr(f, df, estimate=estimate, return_vector=True)
		duration = time.time() - start
		print('Duration: %s' % duration)
		print(f'{len(xv)} iterations:')
		for n, x in enumerate(xv):
			print(f'x[{n}] = {x:.12f}')
		
		start = time.time()
		x, r = scipy.optimize.newton(func=f, x0=estimate, fprime=df, rtol=DEFAULT_EPS, full_output=True)
		duration = time.time() - start
		print('With scipy.optimize.newton:')
		print('Duration: %s' % duration)
		print('%i iterations, %i function calls' % (r.iterations, r.function_calls))

	print()
	print('Regula falsi then NR, range +/- 0.99999:')
	xv = solve_bisection_then_nr(f, df, (-0.99999, 0.99999), return_vector=True)
	for n, x in enumerate(xv):
		print(f'x[{n}] = {x:.12f}')
