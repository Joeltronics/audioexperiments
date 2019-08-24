#!/usr/bin/env python3

import numpy as np
import math
from typing import Union, Iterable, Optional


DEFAULT_EPS = 1.0e-9


def _approx_equal_abs(
		val1: Union[int, float],
		val2: Union[int, float],
		eps: Union[int, float, None]) -> bool:
	return abs(val1 - val2) < eps


"""
The symmetric relative error problem:

Relative error is defined as (x0 - x) / x = (x0 / x) - 1

However, this only works if you know x and x0
What if you just have 2 values - which is x0 and which is x?

The problem is, unlike in absolute space, the error in one direction isn't equal to the other.
e.g. x1 = 1, x2 = 1.1
x2/x1 = 1.1, relative error 0.1
x1/x2 = 0.90909..., relative error 0.090909...

These are close, especially for sufficiently small epsilon, but they're not the same.

Possible solutions:

1. Allow specifying which is the true value

The problem is, we still have to support cases where we don't know which is which,
e.g. for cases of "ensure these two functions return approximately the same value"

2. Always use first value as true value

Could do it this way - again, there's the whole "for sufficiently small epsilon" thing above, so maybe we don't need
to care. But still, you want something that is effectively an equality operator to behave the same way regardless of
order of operands

3. Average the two rel errors?

0.5 * (err(1.0, 1.1) + err(1.1, 1.0)) = 0.09545454

This is a possible fix, though it means calculating twice

4. Use average of the two as the true value (and then double the error)

e.g. 1.0 & 1.01, so true value is 1.05; 2*err(1.1) = 0.095238; 2*err(1.0) = 0.095238

This works, and could be one option. It likely means extra calculations for vector comparisons, though.

5. Require *both* relative errors be < eps

This way you have to calculate twice. Unless...

6. Do it in log space

err = 2^abs(log2(val1) - log2(val2)) - 1

Because of the abs, this will always yield the higher of the two errors - so it's actually effectively the same as #5.
This is what we'll use.
"""


def _approx_equal_rel(
		val1: Union[int, float],
		val2: Union[int, float],
		eps: Union[int, float, None]) -> bool:

	if val1 == val2 == 0.0:
		return True
	elif (val1 == 0.0) or (val2 == 0.0):
		return False
	elif (val1 > 0) != (val2 > 0):
		return False

	val1 = math.log2(abs(val1))
	val2 = math.log2(abs(val2))

	err = 2 ** abs(val1 - val2) - 1

	return err < eps


def _approx_equal(
		val1: Union[int, float],
		val2: Union[int, float],
		check_abs: bool,
		check_rel: bool,
		eps_abs: Union[int, float, None],
		eps_rel: Union[int, float, None]) -> bool:

	assert check_abs or check_rel

	if check_abs and _approx_equal_abs(val1, val2, eps_abs):
		return True

	if check_rel and _approx_equal_rel(val1, val2, eps_rel):
		return True

	return False


def approx_equal_scalar(
		*args: Union[int, float],
		eps: Union[int, float]=DEFAULT_EPS,
		rel=False,
		abs_rel=False,
		eps_abs: Union[int, float, None]=None,
		eps_rel: Union[int, float, None]=None) -> bool:
	"""
	Compare 2 or more values for equality within threshold

	:param args: values to be compared
	:param eps: comparison threshold (if abs_rel is given, will be used for both absolute and relative by default)
	:param rel: compare relative error instead of absolute error
	:param abs_rel: return True as long as relative *or* absolute is within eps
		(e.g. useful for values that may be close to zero)
	:param eps_abs: specify epsilon value for absolute comparison only (e.g. useful if abs_rel)
	:param eps_rel: specify epsilon value for relative comparison only (e.g. useful if abs_rel)
	:return: True if values are within eps of each other

	:note: with relative comparisons, exact zero values will be considered equal only to other zero values
	"""

	if abs_rel and rel:
		raise ValueError('Cannot set both abs_rel and rel!')

	if eps_abs is None:
		eps_abs = eps

	if eps_rel is None:
		eps_rel = eps

	check_rel = abs_rel or rel
	check_abs = abs_rel or not rel

	if len(args) == 1:
		raise ValueError('Must give more than 1 argument!')
	elif len(args) == 2:
		val1, val2 = args
	else:
		val1, val2 = min(args), max(args)

	return _approx_equal(
		val1=val1, val2=val2,
		check_abs=check_abs, eps_abs=eps_abs,
		check_rel=check_rel, eps_rel=eps_rel)


def approx_equal_vector(
		*args: Union[np.ndarray, Iterable],
		eps: Union[int, float]=DEFAULT_EPS,
		rel=False,
		abs_rel=False,
		eps_abs: Union[int, float, None]=None,
		eps_rel: Union[int, float, None]=None) -> bool:
	"""
	Compare 2 or more vectors for equality within threshold

	:param args: values to be compared
	:param eps: comparison threshold (if abs_rel is given, will be used for both absolute and relative by default)
	:param rel: compare relative error instead of absolute error
	:param abs_rel: return True as long as relative *or* absolute is within eps
		(e.g. useful for values that may be close to zero)
	:param eps_abs: specify epsilon value for absolute comparison only (e.g. useful if abs_rel)
	:param eps_rel: specify epsilon value for relative comparison only (e.g. useful if abs_rel)
	:return: True if values are within eps of each other

	:note: with relative comparisons, exact zero values will be considered equal only to other zero values
	"""

	if abs_rel and rel:
		raise ValueError('Cannot set both abs_rel and rel!')

	if eps_abs is None:
		eps_abs = eps

	if eps_rel is None:
		eps_rel = eps

	check_rel = abs_rel or rel
	check_abs = abs_rel or not rel

	len0 = len(args[0])
	if not all([len(vec) == len0 for vec in args[1:]]):
		raise ValueError('All vectors must have the same length!')

	lowest = np.minimum.reduce(args)
	highest = np.maximum.reduce(args)

	if check_abs and not check_rel:
		# Special case: if only checking absolute error, can do this vectorized
		err = highest - lowest
		assert np.amin(err) >= 0  # DEBUG
		return np.amax(err) < eps_abs

	else:
		# Otherwise, the logic has too many branches for it to really be worth vectorizing (due to handling zero cases
		# in rel comparison), so just do it iteratively
		# (I haven't actually done a performance comparison so it's possible I could be wrong here - but it's not really
		# worth the effort of micro-optimizing this anyway - this isn't really a performance-critical function)
		return all([
			_approx_equal(
				val1=val1, val2=val2,
				check_abs=check_abs, eps_abs=eps_abs,
				check_rel=check_rel, eps_rel=eps_rel)
			for val1, val2 in zip(lowest, highest)])


def approx_equal(
		*args: Union[int, float, np.ndarray, Iterable],
		eps: Union[int, float]=DEFAULT_EPS,
		rel=False,
		abs_rel=False,
		eps_abs: Union[int, float, None]=None,
		eps_rel: Union[int, float, None]=None) -> bool:
	"""
	Compare 2 or more values and/or vectors for equality within threshold

	:param args: values to be compared - either 1 vector, or multiple scalar or vector arguments
	:param eps: comparison threshold (if abs_rel is given, will be used for both absolute and relative by default)
	:param rel: compare relative error instead of absolute error
	:param abs_rel: return True as long as relative *or* absolute is within eps
		(e.g. useful for values that may be close to zero)
	:param eps_abs: specify epsilon value for absolute comparison only (e.g. useful if abs_rel)
	:param eps_rel: specify epsilon value for relative comparison only (e.g. useful if abs_rel)
	:return: True if values are within eps of each other

	:note: with relative comparisons, exact zero values will be considered equal only to other zero values
	"""

	kwargs = dict(rel=rel, abs_rel=abs_rel, eps=eps, eps_abs=eps_abs, eps_rel=eps_rel)

	if len(args) == 1:
		if np.isscalar(args[0]):
			raise ValueError('Must give array_like, or more than 1 argument!')
		else:
			return approx_equal_scalar(np.amin(args[0]), np.amax(args[0]), **kwargs)

	is_scalar = [np.isscalar(arg) for arg in args]
	num_scalar = sum(is_scalar)

	if num_scalar == len(args):
		# All scalar
		return approx_equal_scalar(*args, **kwargs)

	elif num_scalar == 0:
		# All vectors
		return approx_equal_vector(*args, **kwargs)

	else:
		# Mix of scalar and vector
		# For now, just extend scalars into vectors (not the most efficient, but not worth optimizing at the moment)
		first_vector = next(arg for (arg, arg_is_scalar) in zip(args, is_scalar) if not arg_is_scalar)
		vector_args = [
			arg*np.ones_like(first_vector) if arg_is_scalar else arg
			for (arg, arg_is_scalar) in zip(args, is_scalar)
		]
		return approx_equal_vector(*vector_args, **kwargs)


_unit_tests = []


def _test_scalar():
	from unit_test import unit_test

	def _test(*args, **kwargs):

		args_as_single_length_np_arrays = [
			np.array([val]) for val in args
		]

		equals = [
			approx_equal_scalar(*args, **kwargs),
			approx_equal_scalar(*reversed(args), **kwargs),
			approx_equal(*args, **kwargs),
			approx_equal(*reversed(args), **kwargs),
			approx_equal([*args], **kwargs),
			approx_equal([*reversed(args)], **kwargs),
			approx_equal_vector(*args_as_single_length_np_arrays, **kwargs),  # TODO: make this one pass
		]

		if not all([val == equals[0] for val in equals[1:]]):
			raise unit_test.UnitTestFailure(
				'Different approx_equal functions returned different values (%s)' % ','.join([str(val) for val in equals]))

		return equals[0]

	# Basic abs tests
	assert _test(1, 1 + 1e-12)
	assert not _test(1, 1.001)
	assert _test(1, 1.1, eps=0.11)
	assert _test(1, 1.1, eps_abs=0.11)
	assert not _test(1, 1.1, eps_rel=0.11)  # eps_rel but with abs comparison

	# Basic rel tests
	assert _test(1, 1.0000001, rel=True, eps=0.0001)
	assert _test(1, 1.1, rel=True, eps=0.11)
	assert _test(1, 1.1, rel=True, eps_rel=0.11)
	assert not _test(1.0, 1.1, rel=True, eps=0.09999999)
	assert not _test(1.1, 1.0, rel=True, eps=0.09999999)
	assert not _test(1.0, 1.1, rel=True, eps_abs=0.11)  # eps_abs but with rel comparison

	# abs_rel
	assert _test(1.0, 1.1, abs_rel=True, eps_rel=0.2, eps_abs=1e-9)
	assert _test(1.0, 1.1, abs_rel=True, eps_rel=1e-9, eps_abs=0.2)
	assert _test(1.0e-9, 1.1e-9)
	assert not _test(1.0e-9, 1.1e-9, rel=True)
	assert _test(1.0e-9, 1.1e-9, abs_rel=True)

	# Common fractions that aren't perfectly representable in float
	assert _test(0.33333333333, 1.0 / 3.0)
	assert _test(0.33333333333, 1.0 / 3.0, rel=True)

	# irrational numbers
	assert _test(math.pi, 3.1415926, eps=0.000001)
	assert _test(math.pi, 3.1415926, eps=0.000001, rel=True)
	assert not _test(math.pi, 3.1415926, eps=0.00000001)
	assert not _test(math.pi, 3.1415926, eps=0.00000001, rel=True)

	# Small numbers
	assert _test(1e-10, 1e-11, rel=False)
	assert not _test(1e-10, 1e-11, rel=True)

	# Negative numbers
	assert _test(-1e-10, -1e-11, rel=False)
	assert not _test(-1e-10, -1e-11, rel=True)

	# Different sign
	assert _test(-1e-10, 1e-10, rel=False)
	assert not _test(-1e-10, 1e-10, rel=True)

	# More than 2 args
	assert _test(1, 0.9999999, 1.0000001, eps=1e-6)
	assert _test(1, 0.9999999, 1.0000001, eps=1e-6, rel=True)
	assert not _test(1, 0.9999999, 1.0000001, eps=1e-12)
	assert not _test(1, 0.9999999, 1.0000001, eps=1e-12, rel=True)
	assert not _test(0.9, 1.0, 1.1, 1.2, eps=0.2)

	# Comparisons with zero
	assert _test(-1e-12, 1e-12, 0.0)
	assert not _test(-1e-12, 1e-12, 0.0, rel=True)
	assert not _test(1e-12, 0.0, rel=True)
	assert not _test(1e-12, 0.0, rel=True, eps=99999999999999.9)
	assert not _test(-1e-9, 1e-9, 0.0, rel=True)


_unit_tests.append(_test_scalar)


def _test_vector():
	from generation import signal_generation

	def test_vector(*args, rel=False, eps=DEFAULT_EPS):
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

	vec1 = np.array([10.00, 1001., -5.00, -1.000e-11, 1.000e-5])
	vec2 = np.array([10.02, 1000., -5.01, -1.001e-11, 1.000e-5])

	assert test_vector(vec1, vec2, eps=0.01, rel=True)
	assert not test_vector(vec1, vec2, eps=0.0001, rel=True)

	vec1 = np.array([10.00, 1001., -5.00, -1.000e-11, 1.000e-5])
	vec2 = np.array([10.02, 1000., -5.01,  1.001e-11, 1.000e-5])

	assert not test_vector(vec1, vec2, eps=9999.9, rel=True)

	vec2[2] = 0.0
	assert not test_vector(vec1, vec2, rel=True)


_unit_tests.append(_test_vector)


def _test_misc():
	# Single list tests

	assert approx_equal([1, 0.9999999999, 1.0000000001])
	assert not approx_equal([0.9, 1.0, 1.1, 1.2], eps=0.2)
	assert approx_equal(np.array([1, 0.9999999999, 1.0000000001]))
	assert not approx_equal(np.array([0.9, 1.0, 1.1, 1.2]), eps=0.2)

	# Vector + Scalar tests

	assert approx_equal(1.0, [1.0, 0.9999999999, 1.0000000001])
	assert approx_equal([1.0, 0.9999999999, 1.0000000001], 1.0)
	assert approx_equal(1.0, [1.0, 0.9999999999, 1.0000000001], rel=True)
	assert approx_equal([1.0, 0.9999999999, 1.0000000001], 1.0, rel=True)
	assert approx_equal(1.0, np.array([1.0, 0.9999999999, 1.0000000001]))
	assert approx_equal(np.array([1.0, 0.9999999999, 1.0000000001]), 1.0)
	assert approx_equal(1.0, np.array([1.0, 0.9999999999, 1.0000000001]), rel=True)
	assert approx_equal(np.array([1.0, 0.9999999999, 1.0000000001]), 1.0, rel=True)

	assert not approx_equal(1.0, [0.9, 1.0, 1.1, 1.2], eps=0.1)
	assert not approx_equal([0.9, 1.0, 1.1, 1.2], 1.0, eps=0.1)
	assert not approx_equal(1.0, [0.9, 1.0, 1.1, 1.2], eps=0.1, rel=True)
	assert not approx_equal([0.9, 1.0, 1.1, 1.2], 1.0, eps=0.1, rel=True)
	assert not approx_equal(1.0, np.array([0.9, 1.0, 1.1, 1.2]), eps=0.1)
	assert not approx_equal(np.array([0.9, 1.0, 1.1, 1.2]), 1.0, eps=0.1)
	assert not approx_equal(1.0, np.array([0.9, 1.0, 1.1, 1.2]), eps=0.1, rel=True)
	assert not approx_equal(np.array([0.9, 1.0, 1.1, 1.2]), 1.0, eps=0.1, rel=True)


_unit_tests.append(_test_misc)


def test(verbose=False):
	from unit_test import unit_test
	return unit_test.run_unit_tests(_unit_tests, verbose=verbose)


def main(args):
	test(args)
