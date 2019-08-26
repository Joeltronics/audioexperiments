#!/usr/bin/env python3

import traceback
from typing import Iterable, Callable
from utils import approx_equal
from matplotlib import pyplot as plt
import numpy as np


# TODO: just use pytest for this


class UnitTestFailure(Exception):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


def _get_func_str(func, *args, **kwargs):
	args_str = ', '.join(
		[str(arg) for arg in args] +
		['%s=%s' % (str(k), str(v)) for k, v in kwargs.items()])
	return '%s(%s)' % (func.__name__, args_str)


_unit_test_logs = []


def log(text):
	"""
	Log text that will only be printed if unit test fails
	"""
	global _unit_test_logs
	_unit_test_logs.append(text)


def _print_logs():
	global _unit_test_logs
	for log in _unit_test_logs:
		print(log)


def _clear_logs():
	global _unit_test_logs
	_unit_test_logs = []


def test_is(val1, val2):
	"""

	:param val1:
	:param val2:
	:raises: UnitTestFailure if test failed
	"""
	if val1 is val2:
		return

	raise UnitTestFailure('Expected %s is %s' % (val1, val2))


def test_equal(val1, val2):
	"""

	:param val1:
	:param val2:
	:raises: UnitTestFailure if test failed
	"""
	if val1 == val2:
		return

	raise UnitTestFailure('Expected %s == %s' % (val1, val2))


def test_approx_equal(val1, val2, **kwargs) -> None:
	"""

	:param val1:
	:param val2:
	:param kwargs: args for approx_equal
	:raises: UnitTestFailure if test failed
	"""
	if approx_equal.approx_equal(val1, val2, **kwargs):
		return

	raise UnitTestFailure('Expected %s ~= %s' % (val1, val2))


def expect_return(
		expected,
		func: Callable,
		approx: bool=True,
		rel: bool=False,
		eps: float=approx_equal.DEFAULT_EPS,
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
	:raises: UnitTestFailure if test failed
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

	raise UnitTestFailure('%s, expected %s%s, returned %s' % (
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
		raise UnitTestFailure(
			'%s, expected throw' % _get_func_str(func, *args, **kwargs))


def run_unit_test(test_func: Callable, verbose=None) -> bool:
	"""Run unit test - test fails if exception (e.g. AssertionError) is thrown

	:param test_func: function to be called (no args)
	:param verbose: print even on success
	:return: True on success, False if an exception was thrown
	"""

	_clear_logs()

	if hasattr(test_func, "name"):
		test_name = test_func.name
	elif hasattr(test_func, "__name__"):
		test_name = test_func.__name__
	elif hasattr(test_func, "__class__"):
		test_name = test_func.__class__
	else:
		test_name = type(test_func).__name__

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

	except UnitTestFailure as ex:
		print('Test "%s" failed' % test_name)
		traceback.print_exc()

	except AssertionError as ex:
		print('Test "%s" failed on assertion' % test_name)
		traceback.print_exc()

	except Exception as ex:
		print('Test "%s" failed - threw %s' % (test_name, type(ex).__name__))
		traceback.print_exc()

	if not success:
		_print_logs()

	if verbose:
		print('')

	return success


def run_unit_tests(test_funcs: Iterable[Callable], verbose=None) -> bool:
	"""Run unit tests - test fails if exception (e.g. AssertionError) is thrown

	:param test_funcs: Iterable of functions to be called (no args)
	:param verbose: if True, print details of every test, even on success
	:return: True if all tests passed
	"""

	num_failures = 0
	num_tests = len(test_funcs)

	if num_tests == 0:
		raise ValueError('No tests given!')

	print('Running %i unit test%s...' % (num_tests, 's' if num_tests > 1 else ''))

	for test_func in test_funcs:
		if not run_unit_test(test_func, verbose=verbose):
			num_failures += 1

	if num_failures:
		print('%i/%i test failures' % (num_failures, num_tests))
		return False
	else:
		if num_tests > 1:
			print('All %i tests passed!' % num_tests)
		else:
			print('Test passed!')

		return True


def plot_equal_failure(expected, actual, title=None, expected_label='expected', actual_label='actual', show=True):
	plt.figure()

	plt.subplot(211)

	plt.plot(expected, label=expected_label)
	plt.plot(actual, label=actual_label)

	plt.grid()

	if expected_label or actual_label:
		plt.legend()

	if title is None:
		plt.title('Unit test failure')
	else:
		plt.title('Unit test failure: %s' % title)

	plt.subplot(212)

	plt.plot(np.abs(actual-expected))
	plt.grid()
	plt.ylabel('abs error')

	plt.xlabel('Sample number')

	if show:
		plt.show()
