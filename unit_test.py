#!/usr/bin/env python3

import traceback
from typing import Iterable, Callable
import sys
import approx_equal


def _get_func_str(func, *args, **kwargs):
	args_str = ', '.join(
		[str(arg) for arg in args] +
		['%s=%s' % (str(k), str(v)) for k, v in kwargs.items()])
	return '%s(%s)' % (func.__name__, args_str)



def test_is(val1, val2):
	"""

	:param val1:
	:param val2:
	:raises: UnitTestFailure if test failed
	"""
	if val1 is val2:
		return

	raise AssertionError('Expected %s is %s' % (str(val1), str(val2)))


def test_equal(val1, val2):
	"""

	:param val1:
	:param val2:
	:raises: AssertionError if test failed
	"""
	if val1 == val2:
		return

	raise AssertionError('Expected %s == %s' % (str(val1), str(val2)))


def test_approx_equal(val1, val2, eps=approx_equal.default_eps, rel=False) -> None:
	"""

	:param expected:
	:param actual:
	:param eps:
	:param rel:
	:raises: AssertionError if test failed
	"""
	if approx_equal.approx_equal(val1, val2, rel=rel, eps=eps):
		return

	raise AssertionError('Expected %s ~= %s' % (str(val1, val2)))


def expect_return(
		expected,
		func: Callable,
		approx: bool=True,
		rel: bool=False,
		eps: float=approx_equal.default_eps,
		*args,
		**kwargs) -> None:
	"""

	:param expected: expected return value of func
	:param func: function to be called
	:param approx: if True, approx_equal will be used
	:param rel: if approx, if relative comparison should be used
	:param eps: if approx, eps value for approx_equal
	:param args: args for function
	:param kwargs: kwargs for function
	:raises: AssertionError if test failed
	"""

	actual = func(*args, **kwargs)

	if expected is None or actual is None:
		if expected is None and actual is None:
			return

	elif approx:
		if approx_equal.approx_equal(expected, actual, rel=rel, eps=eps):
			return
	else:
		if expected == actual:
			return

	raise AssertionError('%s, expected %s%s, returned %s' % (
		_get_func_str(func, *args, **kwargs),
		str(expected),
		' (approximate)' if approx else '',
		str(actual)))


def threw(func: Callable, *args, **kwargs) -> bool:
	"""Return true if function threw

	:param func: function to be called
	:param args: args for function
	:param kwargs: kwargs for function
	:return: true if function threw
	"""
	try:
		func(*args, **kwargs)
		return False
	except Exception:
		return True


def test_threw(func: Callable, *args, **kwargs) -> None:
	"""Test a function expected to throw

	:param func: function to test
	:param args:
	:param kwargs:
	:raises: AssertionError if function did not throw
	"""

	if not threw(func, *args, **kwargs):
		raise AssertionError(
			'%s, expected throw' % _get_func_str(func, *args, **kwargs))


def run_unit_test(test_func: Callable, verbose=None) -> bool:
	"""Run unit test - test fails if exception (e.g. AssertionError) is thrown

	:param test_func: function to be called (no args)
	:param verbose: print even on success
	:return: True on success, False if an exception was thrown
	"""

	if verbose is None:
		verbose = '-v' in sys.argv or '--verbose' in sys.argv

	test_name = test_func.__name__

	# If it's a module private function, strip underscore from start of name
	if test_name.startswith('_'):
		test_name = test_name[1:]

	if verbose:
		print('Running test "%s"...' % test_name)

	success = False

	try:
		test_func()
		success = True
		if verbose:
			print('Test "%s" Passed' % test_name)

	except AssertionError as ex:
		print('Test "%s" failed' % test_name)
		traceback.print_exc()

	except Exception as ex:
		print('Test "%s" failed - threw %s' % (test_name, ex.__name__))
		traceback.print_exc()

	if verbose:
		print('')

	return success


def run_unit_tests(test_funcs: Iterable[Callable], verbose=None) -> bool:
	"""Run unit tests - test fails if exception (e.g. AssertionError) is thrown

	:param test_funcs: Iterable of functions to be called (no args)
	:param verbose: if True, print details of ever test even on success
	:return: True if all tests passed
	"""

	num_failures = 0
	num_tests = len(test_funcs)

	print('Running %i unit tests...' % num_tests)

	for test_func in test_funcs:
		if not run_unit_test(test_func, verbose=verbose):
			num_failures += 1

	if num_failures:
		print('%i/%i test failures' % (num_failures, num_tests))
		return False
	else:
		print('All %i tests passed!' % num_tests)
		return True


# TODO: unit test the unit test framework...
