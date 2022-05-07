#!/usr/bin/env python3

from solvers.iter_stats import IterStats

from utils import utils

from typing import Tuple, Iterable, Optional, Callable, Union


# FIXME: this is bad, these are globals that get modified from outside the module
# (these are used as a default argument, and modified after the module is imported - does this even work properly?)
legacy_max_num_iter = 20
legacy_eps = 1e-6


def solve_fb_iterative(
		f: Callable[[float], float],
		estimate: float,
		rate_limit: Optional[float]=None,
		eps: Optional[float]=1.0e-6,
		max_num_iter=100,
		throw_if_failed_converge=True,
		return_vector=False,
		iter_stats: Optional[IterStats]=None,
) -> Union[Tuple[float, int], Iterable[float]]:
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
		return y, n


def solve_iterative(
		f: Callable[[float], float],
		estimate: float,
		rate_limit: Optional[float]=None,
		eps: Optional[float]=1.0e-6,
		max_num_iter=100,
		throw_if_failed_converge=True,
		return_vector=False,
		iter_stats: Optional[IterStats]=None,
) -> Union[Tuple[float, int], Iterable[float]]:
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

	g(x) = x
	"""

	return solve_fb_iterative(
		lambda x: f(x) + x,
		estimate,
		rate_limit,
		eps,
		max_num_iter,
		throw_if_failed_converge,
		return_vector,
		iter_stats=iter_stats)


# TODO: consolidate this with solve_iterative()
def solve_iterative_legacy(
		f_zero,
		estimate=None,
		max_num_iter=legacy_max_num_iter,
		eps=legacy_eps,
		iter_stats: Optional[IterStats]=None,):
	if estimate is None:
		estimate = 0.0

	y = estimate

	errs = []

	success = False
	prev_abs_err = None
	for iter_num in range(max_num_iter):

		err = f_zero(y)

		abs_err = abs(err)
		errs += [abs_err]

		if abs_err <= eps:
			success = True
			break

		if (prev_abs_err is not None) and (abs_err >= prev_abs_err):
			print('Warning: failed to converge! Falling back to initial estimate')
			# return estimate
			y = estimate
			break

		y = y - err

		prev_abs_err = abs_err

	if iter_stats is not None:
		iter_stats.add(
			success=success,
			est=estimate,
			n_iter=iter_num + 1,
			final=y,
			err=errs)

	return y


def solve_bisection(
		f: Callable[[float], float],
		estimate: float,
		init_range: Tuple[float, float],
		interp=True,
		eps: Optional[float]=1.0e-6,
		max_num_iter=100,
		throw_if_failed_converge=True,
		return_vector=False) -> Union[Tuple[float, int], Iterable[float]]:
	"""
	Solve f(x) = 0 by bisection

	:param f:
	:param estimate: initial estimate
	:param init_range: initial range
	:param interp: if true, interpolates; if false, performs naive bisection (average of 2 values)
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
		x_prev = x

		if xv is not None:
			xv.append(x)

		y = f(x)

		if y < 0:
			xmin, ymin = x, y
		else:
			xmax, ymax = x, y

		if interp:
			x = utils.scale(0, (ymin, ymax), (xmin, xmax))
		else:
			x = 0.5 * (xmin + xmax)

		if eps and (abs(x - x_prev) < eps):
			break

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
		eps: Optional[float]=1.0e-6,
		max_num_iter=100,
		throw_if_failed_converge=True,
		return_vector=False,
		iter_stats: Optional[IterStats]=None,
) -> Union[Tuple[float, int], Iterable[float]]:
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
		x -= residue

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
		return x, n


# TODO: consolidate this with solve_nr()
def solve_nr_legacy(
		f,
		df,
		estimate,
		max_num_iter=legacy_max_num_iter,
		eps=legacy_eps,
		iter_stats: Optional[IterStats]=None):

	y = estimate

	errs = []

	success = False
	prev_err = None
	for iter_num in range(max_num_iter):

		fy = f(y)

		err = abs(fy)

		errs += [err]

		if err <= eps:
			success = True
			break

		if (prev_err is not None) and (err >= prev_err):
			print('Warning: failed to converge! Falling back to initial estimate')
			y = estimate
			break

		dfy = df(y)

		# Prevent divide-by-zero, or very shallow slopes
		if dfy < eps:
			# this shouldn't be possible with the functions we're actually using (derivative of tanh)
			print("Warning: d/dy f(y=%f) = 0.0, can't solve Newton-Raphson" % y)
			break

		y = y - fy / dfy

		prev_err = err

	if iter_stats is not None:
		iter_stats.add(
			success=success,
			est=estimate,
			n_iter=iter_num + 1,
			final=y,
			err=errs)

	return y


def main(args):
	"""
	y = (x + 1)(x - 1)(x - 4) = (x^2 - 1)(x - 4) = x^3 - 4x^2 - x + 4
	dy/dx = 3x^2 - 8x - 1
	"""

	def f(x):
		return (x ** 3.) - 4.*(x ** 2.) - x + 4

	def df(x):
		return 3.*(x ** 2.) - 8.*x - 1

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
	range = (-0.75, 3.75)
	est = 0.5 * (range[0] + range[1])
	print('True bisection, init range: x = (%g, %g), y = (%g, %g), est = %g' % (range[0], range[1], f(range[0]), f(range[1]), est))
	xv = solve_bisection(f, estimate=est, init_range=range, interp=False, return_vector=True)
	print('%i iterations:' % len(xv))
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	print('')
	print('Interpolating bisection, init range: x = (%g, %g), y = (%g, %g), est = %g' % (range[0], range[1], f(range[0]), f(range[1]), est))
	xv = solve_bisection(f, estimate=est, init_range=range, interp=True, return_vector=True)
	print('%i iterations:' % len(xv))
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	print('')
	range = (1.5, 7.)
	est = 0.5 * (range[0] + range[1])
	print('Interpolating bisection, init range: x = (%g, %g), y = (%g, %g), est = %g' % (range[0], range[1], f(range[0]), f(range[1]), est))
	xv = solve_bisection(f, estimate=est, init_range=range, interp=True, return_vector=True)
	print('%i iterations:' % len(xv))
	for n, x in enumerate(xv):
		print('x[%i] = %.12f' % (n, x))

	print('')
	print('Newton-Raphson:')
	for est in [-10, -1.1, 1.1, 2., 10]:
		print('')
		print('Estimate %g' % est)

		xv = solve_nr(f, df, estimate=est, return_vector=True)
		print('%i iterations:' % len(xv))
		for n, x in enumerate(xv):
			print('x[%i] = %.12f' % (n, x))
