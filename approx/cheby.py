#!/usr/bin/env python3

import argparse
import math
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, Union, Optional, Tuple, Iterable


_eps = 1e-12


def _is_zero(val: float):
	return np.abs(val) < _eps


def _calc_cheby_coeffs(order, second_kind=False):

	if order < 0:
		raise ValueError

	if second_kind:
		cheb = [
			[1],
			[0, 2]
		]
	else:
		cheb = [
			[1],
			[0, 1]
		]

	if order < 2:
		return cheb[:(order + 1)]

	for _ in range(order - 1):

		c_prev_2 = cheb[-2] + [0, 0]
		c_prev_1_x = [0] + cheb[-1]

		this_c = [
			2*c1 - c2
			for c2, c1
			in zip(c_prev_2, c_prev_1_x)
		]

		cheb.append(this_c)

	return cheb


MAX_CHEBY_ORDER = 10
CHEBY_COEFFS = _calc_cheby_coeffs(MAX_CHEBY_ORDER)
CHEBY_COEFFS_SECOND_KIND = _calc_cheby_coeffs(MAX_CHEBY_ORDER, second_kind=True)


def cheby_poly(n: int, x: Union[float, np.ndarray], second_kind=False) -> Union[float, np.ndarray]:
	"""
	Chebyshev polynomial
	"""

	# Yes, I know there are functions to do this in numpy
	# Yes, I also know this could be done iteratively

	if n < 0:
		raise ValueError('n must be >= 0')

	coeff_list = CHEBY_COEFFS_SECOND_KIND if second_kind else CHEBY_COEFFS

	if n > MAX_CHEBY_ORDER:
		# TODO: could just update CHEBY_COEFFS on the fly
		raise NotImplementedError('Chebyshev polynomials > C%i not implemented' % MAX_CHEBY_ORDER)

	if n == 0:
		return np.ones_like(x)

	vals = np.zeros_like(x)

	for m in range(n + 1):
		c = coeff_list[n][m]
		if c:
			vals += c * (x ** m)

	return vals


def _cheby_nodes(num):
	#theta = np.linspace(-np.pi, np.pi, num, endpoint=False)
	#theta += 0.5*(theta[1] - theta[0])
	#return np.cos(theta)

	t = (np.array(range(0, num)) + 0.5) / num
	return -np.cos(t*math.pi)


def cheby_fit(f: Callable, value_range: Tuple[float, float], degree=MAX_CHEBY_ORDER, verbose=False) -> np.ndarray:
	"""
	:param f: Function to fit
	:param value_range: Range of values to fit over
	:param degree: Specify a degree. For most precise results, leave at default (max) and truncate to desired degree
	:param verbose:
	:return: numpy array of Chebyshev polynomial coefficients
	"""

	a = value_range[0]
	b = value_range[1]

	nodes = _cheby_nodes(degree + 1)

	x = (nodes * (b - a) + (b + a)) / 2.0
	y = f(x)

	if verbose:
		print('Nodes: %s' % nodes)
		print('x: %s' % x)
		print('y: %s' % y)

	coeffs = np.zeros(degree + 1)
	for n in range(degree + 1):

		cheb = cheby_poly(n, nodes)
		sum = np.dot(cheb, y)

		coeff = sum / len(y)

		if n > 0:
			coeff *= 2

		coeffs[n] = coeff

	return coeffs


def cheby_coeffs_to_polynomial(cheby_coeffs: Iterable[float]) -> np.ndarray:

	poly_coeffs = np.zeros(len(cheby_coeffs))

	for n, cheby_coeff in enumerate(cheby_coeffs):

		cheby_coeffs_this_order = CHEBY_COEFFS[n]

		for m, c in enumerate(cheby_coeffs_this_order):
			if c:
				poly_coeffs[m] += (c * cheby_coeff)

	return poly_coeffs


def _print_poly_coeffs(poly_coeffs):

	order = len(poly_coeffs) - 1

	print('As polynomial, order %u:' % order)
	for n, coeff in enumerate(poly_coeffs):
		if _is_zero(coeff):
			continue
		print('  x^%u * %0.24f' % (n, coeff))


def _const_to_str(val: float) -> str:

	for eq_val, val_as_str in [
		(math.pi, 'math.pi'),
		(2 * math.pi, '(2 * math.pi)'),
		(0.5 * math.pi, '(0.5 * math.pi)'),
		(0.25 * math.pi, '(0.25 * math.pi)'),
		(0.125 * math.pi, '(0.125 * math.pi)'),
		(math.sqrt(2), 'math.sqrt(2)'),
		(1.0 / math.sqrt(2), '(1 / math.sqrt(2))'),
		(math.sqrt(3), 'math.sqrt(3)'),
		(0.5 * math.sqrt(3), '(0.5 * math.sqrt(3))'),
		(math.e, 'math.e'),
	]:
		if math.isclose(val, eq_val, rel_tol=_eps):
			return val_as_str

	return '%.24f' % val


def _divide_by_const_to_str(denom: float) -> str:

	if denom == 0:
		raise ZeroDivisionError

	for eq_denom, val_as_str in [
		(1, ''),
		(math.pi, ' / math.pi'),
		(2 * math.pi, ' * 0.5 / math.pi'),
		(0.5 * math.pi, ' * 2 / math.pi'),
		(0.25 * math.pi, ' * 4 / math.pi'),
		(0.125 * math.pi, ' * 8 / math.pi'),
		(math.sqrt(2), ' / math.sqrt(2)'),
		(1.0 / math.sqrt(2), ' * math.sqrt(2)'),
		(math.sqrt(3), ' / math.sqrt(3)'),
		(0.5 * math.sqrt(3), ' / (0.5 * math.sqrt(3))'),
		(math.e, ' / math.e'),
	]:
		if math.isclose(denom, eq_denom, rel_tol=_eps):
			return val_as_str

	return ' * %.24f' % (1.0 / denom)


def _get_py_range_scaling(value_range: Tuple[float, float]) -> str:

	a, b = value_range

	if math.isclose(a, -1, rel_tol=_eps) and math.isclose(b, 1, rel_tol=_eps):
		return ''

	"""
	v = (x - a) / (b - a)
	u = 2*v - 1

	u = 2 * (x - a) / (b - a) - 1
	u = (x - a) / (0.5 * (b - a)) - 1

	We'll use this, but say we wanted to go further:

	u = 2 * (x - a) / (b - a) - 1
	u = 2 * (x - a) / (b - a) - (b - a) / (b - a)
	u = [ 2 * (x - a) - (b - a) ] / (b - a)
	u = (2*x - 2*a - b + a) / (b - a)
	u = (2*x - a - b) / (b - a)
	"""

	denom = 0.5 * (b - a)

	if math.isclose(denom, 1, rel_tol=_eps):
		add = -a + 1
		assert add != 0  # Would have already returned '' above

		if add < 0:
			return 'x = x - %s' % _const_to_str(-add)
		else:
			return 'x = x + %s' % _const_to_str(add)

	if denom == a:
		"""
		u = (x - a) / a - 1
		u = x/a - a/a - 1
		u = x/a - 1 - 1
		u = x/a - 2
		"""
		return 'x = x%s - 2' % _divide_by_const_to_str(denom)

	elif denom == -a:
		"""
		u = (x + a) / a - 1
		u = x/a + a/a - 1
		u = x/a + 1 - 1
		u = x/a
		"""
		return 'x = x%s' % _divide_by_const_to_str(denom)

	s = 'x = '

	if a < 0:
		s += '(x + %s)' % _const_to_str(-a)
	elif a:
		s += '(x - %s)' % _const_to_str(a)
	else:
		s += 'x'

	s += _divide_by_const_to_str(denom)
	s += ' - 1'

	return s


def _print_py_func(
		polynomial_coeffs: Iterable[float],
		value_range: Tuple[float, float],
		func_name_template: str,
		tabs=True,
):

	order = len(polynomial_coeffs) - 1

	tab = '\t' if tabs else '    '

	func_name = func_name_template % order

	print('def %s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:' % func_name)

	range_scaling_str = _get_py_range_scaling(value_range)
	if range_scaling_str:
		print('%s%s' % (tab, range_scaling_str))

	print('%sreturn \\' % tab)

	for n, coeff in enumerate(polynomial_coeffs):

		if _is_zero(coeff):
			continue

		end_token = '' if n >= order else ' + \\'

		if n == 0:
			print('%s%s%0.24f%s' % (tab, tab, coeff, end_token))
		elif n == 1:
			print('%s%s%0.24f * x%s' % (tab, tab, coeff, end_token))
		else:
			print('%s%s%0.24f * (x ** %u)%s' % (tab, tab, coeff, n, end_token))

	print('')


def _calc_cheby_fit(py_format=False, tabs=False):

	PI = np.pi
	HALF_PI = 0.5 * PI
	TWO_PI = 2.0 * PI

	print('Chebyshev coeffs:')
	for n, coeff_list in enumerate(CHEBY_COEFFS):
		print('  C%i: %s' % (n, coeff_list))

	min_poly_order = 2
	max_poly_order = 8

	for name, func_name_template, func, value_range in [
		('cos [0, PI/2]',     'cos_cheb%u_q1',     np.cos,  (0, HALF_PI)),
		('cos [-PI/2, PI/2]', 'cos_cheb%u_q14',    np.cos,  (-HALF_PI, HALF_PI)),
		('cos [-PI, PI]',     'cos_cheb%u_npi_pi', np.cos,  (-PI, PI)),
		('cos [0, 2PI]',      'cos_cheb%u_0_2pi',  np.cos,  (0, TWO_PI)),

		('sin [0, PI/2]',     'sin_cheb%u_q1',     np.sin,  (0, HALF_PI)),
		('sin [-PI/2, PI/2]', 'sin_cheb%u_q14',    np.sin,  (-HALF_PI, HALF_PI)),
		('sin [-PI, PI]',     'sin_cheb%u_npi_pi', np.sin,  (-PI, PI)),
		('sin [0, 2PI]',      'sin_cheb%u_0_2pi',  np.sin,  (0, TWO_PI)),

		('norm sinc [-1, 1]', 'sinc_lobe_cheb%u',  np.sinc, (-1, 1)),
	]:

		print('')
		print('%s:' % name)

		cheby_coeffs = cheby_fit(func, value_range)

		even = any([not _is_zero(c) for c in cheby_coeffs[0::2]])
		odd = any([not _is_zero(c) for c in cheby_coeffs[1::2]])
		assert even or odd

		for n, coeff in enumerate(cheby_coeffs):
			if _is_zero(coeff):
				continue
			print('  C%u(x) * %0.24f' % (n, coeff))

		for order in range(min_poly_order, max_poly_order + 1):

			if order % 2 == 0:
				if not even:
					continue
			elif not odd:
				continue

			poly_coeffs = cheby_coeffs_to_polynomial(cheby_coeffs[:order + 1])

			if py_format:
				_print_py_func(poly_coeffs, value_range, func_name_template, tabs=tabs)
			else:
				_print_poly_coeffs(poly_coeffs)


def _plot_example():
	PI = np.pi
	HALF_PI = 0.5 * PI

	range = HALF_PI

	x = np.linspace(-range, range, 1024)

	u = x / range
	assert math.isclose(u[0], -1)
	assert math.isclose(u[-1], 1)

	y_actual = np.cos(x)

	cheby_coeffs_trunc4 = cheby_fit(np.cos, (-range, range), MAX_CHEBY_ORDER)[:5]
	cheby_coeffs_order4 = cheby_fit(np.cos, (-range, range), 4)

	poly_coeffs = cheby_coeffs_to_polynomial(cheby_coeffs_trunc4)

	assert len(cheby_coeffs_trunc4) == 5

	assert abs(cheby_coeffs_trunc4[1]) < 1e-12
	assert abs(cheby_coeffs_trunc4[3]) < 1e-12

	assert abs(poly_coeffs[1]) < 1e-12
	assert abs(poly_coeffs[3]) < 1e-12

	y_cheby = \
		cheby_coeffs_trunc4[4] * cheby_poly(4, u) + \
		cheby_coeffs_trunc4[2] * cheby_poly(2, u) + \
		cheby_coeffs_trunc4[0] * cheby_poly(0, u)

	y_cheby_4 = \
		cheby_coeffs_order4[4] * cheby_poly(4, u) + \
		cheby_coeffs_order4[2] * cheby_poly(2, u) + \
		cheby_coeffs_order4[0] * cheby_poly(0, u)

	y_poly = \
		poly_coeffs[4] * (u ** 4) + \
		poly_coeffs[2] * (u ** 2) + \
		poly_coeffs[0]

	err_cheby = np.abs(y_actual - y_cheby)
	err_cheby_4 = np.abs(y_actual - y_cheby_4)
	err_poly = np.abs(y_actual - y_poly)

	label_cheby = '%.6f*C0(x) + %.6f*C2(x) + %.6f*C4(x)' % (
		cheby_coeffs_trunc4[0],
		cheby_coeffs_trunc4[2],
		cheby_coeffs_trunc4[4],
	)

	label_poly = '%.6f + %.6f*x^2 + %.6f*x^4' % (
		poly_coeffs[0],
		poly_coeffs[2],
		poly_coeffs[4],
	)

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.plot(x / PI, y_cheby, label='Chebyshev calculated at order %u truncated to 4' % MAX_CHEBY_ORDER)
	plt.plot(x / PI, y_poly, label='Chebyshev calculated at order 4')
	plt.plot(x / PI, y_actual, label='cos(x)')
	plt.grid()
	plt.legend()

	plt.subplot(2, 1, 2)
	plt.plot(x / PI, err_cheby, label='Calc order %u & truncate error' % MAX_CHEBY_ORDER)
	plt.plot(x / PI, err_cheby_4, label='Calc order 4 error')
	plt.grid()
	plt.legend()
	plt.ylabel('Error')
	plt.xlabel('x / pi')

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.plot(x / PI, y_cheby, label=label_cheby)
	plt.plot(x / PI, y_poly, label=label_poly)
	plt.plot(x / PI, y_actual, label='cos(x)')
	plt.grid()
	plt.legend()

	plt.subplot(2, 1, 2)
	plt.plot(x / PI, err_cheby, label='Cheby error')
	plt.plot(x / PI, err_poly, label='Poly error')
	plt.grid()
	plt.legend()
	plt.ylabel('Error')
	plt.xlabel('x / pi')


def _plot_cheby_functions():

	x = np.linspace(-1, 1, 1024)

	fig = plt.figure()
	fig.suptitle('Chebyshev functions')

	plt.subplot(2, 1, 1)
	for n in range(6):
		y = cheby_poly(n, x, second_kind=False)
		assert y.shape == x.shape
		plt.plot(x, y, label='U%u(x)' % n)
	plt.ylabel('1st kind')
	plt.legend()
	plt.grid()

	plt.subplot(2, 1, 2)
	for n in range(6):
		y = cheby_poly(n, x, second_kind=True)
		plt.plot(x, y, label='U%u(x)' % n)
	plt.ylabel('2nd kind')
	plt.legend()
	plt.grid()


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--py', dest='py_format', action='store_true', help='Print polynomials as python code')
	parser.add_argument('--spaces', dest='tabs', action='store_false', help='Use spaces for python code')
	# TODO
	#parser.add_argument('-c', '--cpp', dest='c_format', action='store_true', help='Print polynomials as C/C++ code')

	return parser


def plot(args):
	_plot_cheby_functions()
	_plot_example()
	plt.show()


def main(args):
	_calc_cheby_fit(py_format=args.py_format, tabs=args.tabs)
