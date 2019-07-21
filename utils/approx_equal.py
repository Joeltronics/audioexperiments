#!/usr/bin/env python3

import numpy as np
import math
from typing import Union, Iterable


default_eps = 0.0001


def approx_equal_scalar(
		*args: Union[int, float],
		eps: Union[int, float]=default_eps,
		rel: bool=False) -> bool:
	"""Compare 2 or more values for equality within threshold

	:param args: values to be compared
	:param eps: comparison threshold
	:param rel: if true, operation is performed in log domain, and eps is relative to log2
	:return: True if values are within eps of each other
	"""

	if rel and any([val == 0.0 for val in args]):
		raise ZeroDivisionError("Cannot call approx_equal(rel=True) for value 0")

	if len(args) == 1:
		raise ValueError('Must give more than 1 argument!')
	elif len(args) == 2:
		val1, val2 = args
	else:
		val1, val2 = min(args), max(args)

	if rel:
		if (val1 > 0) != (val2 > 0):
			return False

		val1 = math.log2(abs(val1))
		val2 = math.log2(abs(val2))

	return abs(val1 - val2) < eps


def approx_equal_vector(
		first_vector: Union[np.ndarray, Iterable],
		*args: Union[np.ndarray, Iterable],
		eps: Union[int, float]=default_eps,
		rel: bool=False) -> bool:
	"""Compare 2 or more vectors for equality within threshold

	:param first_vector: vector to be compared against
	:param args: vectors to be compared against first
	:param eps: comparison threshold
	:param rel: if true, operation is performed in log domain, and eps is relative to log2
	:return: True if values are within eps of each other
	"""

	if not all([len(vec) == len(first_vector) for vec in args]):
		raise ValueError('All vectors must have the same length!')

	if not rel:
		return all([np.amax(np.abs(arg - first_vector)) < eps for arg in args])

	# TODO: can numpy do this more easily?
	if any([val == 0 for val in first_vector]) or any([any([val == 0 for val in arg]) for arg in args]):
		raise ZeroDivisionError("Cannot call approx_equal(rel=True) for vector with any value 0")

	# Check that log2(abs(val)) is within eps
	if any([np.amax(np.abs(np.log2(np.abs(arg)) - np.log2(np.abs(first_vector)))) >= eps for arg in args]):
		return False

	# Because we took abs, also check that sign matches
	# TODO: can numpy do this more easily?
	if any([any([(v1 > 0) != (v2 > 0) for v1, v2 in zip(arg, first_vector)]) for arg in args]):
		return False

	return True


def approx_equal(
		*args: Union[int, float, np.ndarray, Iterable],
		eps: Union[int, float]=default_eps,
		rel: bool=False) -> bool:
	"""Compare 2 or more values for equality within threshold

	:param args: values to be compared. If no scalars and more than 2 vectors are given, all vectors are compared to the
	first (e.g. if 3 vectors, 1 & 2 and 1 & 3 will be compared; 2 & 3 will not be compared)
	:param eps: comparison threshold
	:param rel: if true, operation is performed in log domain, and eps is relative to log2
	:return: True if values are within eps of each other
	"""

	if len(args) == 1:
		if np.isscalar(args[0]):
			raise ValueError('Must give array_like, or more than 1 argument!')
		else:
			return approx_equal_scalar(np.amin(args[0]), np.amax(args[0]))

	is_scalar = [np.isscalar(arg) for arg in args]

	if any(is_scalar):
		# If at least 1 scalar, then can just compare absolute max & min
		min_val = min([np.amin(arg) for arg in args])
		max_val = max([np.amax(arg) for arg in args])
		return approx_equal_scalar(min_val, max_val, eps=eps, rel=rel)
	else:
		# All vectors
		return approx_equal_vector(*args, eps=eps, rel=rel)


_unit_tests = []


def _test_scalar():
	from unit_test import unit_test

	def _test(*args, rel=False, eps=default_eps):
		eq1 = approx_equal_scalar(*args, rel=rel, eps=eps)
		eq2 = approx_equal(*args, rel=rel, eps=eps)
		if eq1 != eq2:
			raise AssertionError('approx_equal and approx_equal_scalar returned different')
		return eq1

	assert _test(1, 1.0000001)
	assert not _test(1, 1.001)
	assert _test(1, 1.1, eps=0.11)
	assert _test(1, 1.0000001, rel=True)
	assert _test(0.33333333333, 1.0 / 3.0)
	assert _test(0.33333333333, 1.0 / 3.0, rel=True)

	assert _test(1e-8, 1e-9, rel=False)
	assert not _test(1e-8, 1e-9, rel=True)

	assert _test(-1e-8, -1e-9, rel=False)
	assert not _test(-1e-8, -1e-9, rel=True)

	assert _test(-0.0000001, 0.0000001, rel=False)
	assert not _test(-0.0000001, 0.0000001, rel=True)

	assert _test(1, 0.9999999, 1.0000001)
	assert not _test(0.9, 1.0, 1.1, 1.2, eps=0.2)

	assert _test(-1e-9, 1e-9, 0.0)

	unit_test.test_threw(_test, -1e-9, 1e-9, 0.0, rel=True)


_unit_tests.append(_test_scalar)


def _test_vector():
	from unit_test import unit_test
	from generation import signal_generation

	def test_vector(*args, rel=False, eps=default_eps):
		eq1 = approx_equal_vector(*args, rel=rel, eps=eps)
		eq2 = approx_equal(*args, rel=rel, eps=eps)
		if eq1 != eq2:
			raise AssertionError('approx_equal and approx_equal_vector returned different')
		return eq1

	# abs error tests

	f1 = 440. / 44100.
	f2 = 7000. / 44100.

	n_samp = 2 * int(math.ceil(1.0 / f1))

	err_mag = 0.0001

	sine1 = signal_generation.gen_sine(f1, n_samp)
	sine2 = signal_generation.gen_sine(f2, n_samp, start_phase=0.83)

	assert test_vector(sine1, sine1 + err_mag*sine2, eps=2.0*err_mag)
	assert not test_vector(sine1, sine1 + err_mag*sine2, eps=0.5*err_mag)

	# rel error tests

	# log2 error of these all is in ballpark of 0.002
	vec1 = np.array([10.00, 1001., -5.00, -1.000e-11, 1.000e-5])
	vec2 = np.array([10.02, 1000., -5.01, -1.001e-11, 1.000e-5])

	assert test_vector(vec1, vec2, eps=0.01, rel=True)
	assert not test_vector(vec1, vec2, eps=0.0001, rel=True)

	vec1 = np.array([10.00, 1001., -5.00, -1.000e-11, 1.000e-5])
	vec2 = np.array([10.02, 1000., -5.01,  1.001e-11, 1.000e-5])

	assert not test_vector(vec1, vec2, eps=9999.9, rel=True)

	vec2[2] = 0.0
	unit_test.test_threw(test_vector, vec1, vec2, rel=True)


_unit_tests.append(_test_vector)


def _test_misc():
	# Single list tests

	assert approx_equal([1, 0.9999999, 1.0000001])
	assert not approx_equal([0.9, 1.0, 1.1, 1.2], eps=0.2)
	assert approx_equal(np.array([1, 0.9999999, 1.0000001]))
	assert not approx_equal(np.array([0.9, 1.0, 1.1, 1.2]), eps=0.2)

	# Vector + Scalar tests

	assert approx_equal(1.0, [1.0, 0.9999999, 1.0000001])
	assert not approx_equal(1.0, [0.9, 1.0, 1.1, 1.2], eps=0.2)
	assert approx_equal(1.0, np.array([1.0, 0.9999999, 1.0000001]))
	assert not approx_equal(1.0, np.array([0.9, 1.0, 1.1, 1.2]), eps=0.2)


_unit_tests.append(_test_misc)


def test(verbose=False):
	from unit_test import unit_test
	return unit_test.run_unit_tests(_unit_tests, verbose=verbose)


def main(args):
	test(args)
