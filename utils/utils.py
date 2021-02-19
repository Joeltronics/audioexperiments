#!/usr/bin/env python3

from typing import Tuple, Any, List
from generation import signal_generation
from .approx_equal import *
from unit_test.unit_test import test_approx_equal

from typing import Optional
import numpy as np

_unit_tests = []


def is_even(value: int):
	return value % 2 == 0


def is_odd(value: int):
	return value % 2 != 0


def _test_is_even_odd():
	for val in [1, 3, -1, -10001]:
		assert is_odd(val) and not is_even(val)
	for val in [0, 2, -2, 10, -10000]:
		assert is_even(val) and not is_odd(val)


_unit_tests.append(_test_is_even_odd)


def to_pretty_str(val, num_decimals=6, point_zero=True) -> str:
	"""Convert float into nicely formatted string

	If another type is given, just calls str(val)
	"""

	if isinstance(val, float) or isinstance(val, np.floating):
		fmt = '%%.%if' % num_decimals
		s = fmt % val
		while s.endswith('0'):
			s = s[:-1]
		if s.endswith('.'):
			if point_zero:
				s = s + '0'
			else:
				s = s[:-1]
		return s
	else:
		return str(val)


def _test_to_pretty_str():
	assert to_pretty_str(1) == '1'
	assert to_pretty_str(0) == '0'
	assert to_pretty_str(-12345) == '-12345'
	assert to_pretty_str(-12345.0) == '-12345.0'
	assert to_pretty_str(-12345.0, point_zero=False) == '-12345'
	assert to_pretty_str(1.00000001) == '1.0'
	assert to_pretty_str(1.00000001, point_zero=False) == '1'
	assert to_pretty_str(0.0) == '0.0'
	assert to_pretty_str(0.00000001) == '0.0'
	assert to_pretty_str(0.00000001, point_zero=False) == '0'
	assert to_pretty_str(0.12345678) == '0.123457'
	assert to_pretty_str(0.1234567890123456789) == '0.123457'
	assert to_pretty_str(0.12, num_decimals=2) == '0.12'
	assert to_pretty_str('test') == 'test'


_unit_tests.append(_test_to_pretty_str)


def unit_str(val, unit='', num_decimals=3, point_zero=False):

	prefixes = [
		(1e12, None),
		(1e9, 'G'),
		(1e6, 'M'),
		(1e3, 'k'),
		(1, ''),
		(1e-3, 'm'),
		(1e-6, 'u'),
		(1e-9, 'n'),
		(1e-12, 'p'),
	]

	# Without round_val logic, 0.999999999 would return "1000 m"
	# e.g. round_val for num_decimals=3 is 0.9995
	round_val = 1.0 - 0.5 * (10.0 ** -num_decimals)

	for d, p in prefixes:
		if val >= (d * round_val):
			div = d
			prefix = p
			break
	else:
		div = prefix = None

	if prefix is None:
		fmt = '%%.%ig %%s' % num_decimals
		return fmt % (val, unit)
	else:
		return '%s %s%s' % (
			to_pretty_str(val / div, num_decimals=num_decimals, point_zero=point_zero),
			prefix,
			unit)


def _test_unit_str():
	assert unit_str(3, unit='A', num_decimals=6, point_zero=True) == '3.0 A'
	assert unit_str(3, unit='V', num_decimals=6, point_zero=False) == '3 V'
	assert unit_str(4700, unit='ohm') == '4.7 kohm'
	assert unit_str(.01 * 1e-6, unit='F', point_zero=False) == '10 nF'
	assert unit_str(.01 * 1e-6, unit='F', point_zero=True) == '10.0 nF'
	assert unit_str(2.1e13, unit='test') == '2.1e+13 test'
	assert unit_str(0.999, unit='V', point_zero=False) == '999 mV'
	assert unit_str(0.9999, unit='V', point_zero=False) == '1 V'
	assert unit_str(0.9999, unit='V', num_decimals=4, point_zero=False) == '999.9 mV'
	assert unit_str(0.99994, unit='V', num_decimals=4, point_zero=False) == '999.94 mV'


_unit_tests.append(_test_unit_str)


def integerize_if_int(
		num: Union[
			float, int,
			np.int8, np.int8, np.int32, np.int64,
			np.uint8, np.uint8, np.uint32, np.uint64,
			np.float32, np.float64
		]
) -> Union[float, int]:

	if isinstance(num, (int, np.int8, np.int8, np.int32, np.int64, np.uint8, np.uint8, np.uint32, np.uint64)):
		return int(num)

	elif isinstance(num, (float, np.float32, np.float64)):
		num = float(num)
		if num.is_integer():
			return int(num)
		else:
			return num

	else:
		raise TypeError


def _test_integerize_if_int():

	for val in [0, -3, 3]:

		ret = integerize_if_int(val)
		assert isinstance(ret, int)
		assert ret == val

		ret = integerize_if_int(float(val))
		assert isinstance(ret, int)
		assert ret == val

	for val in [0.0000001, -0.0000001, 0.5, -0.5, 2.1, -2.1]:
		ret = integerize_if_int(val)
		assert isinstance(ret, float)
		assert ret == val  # Should be exactly equal, no approx_equal needed


_unit_tests.append(_test_integerize_if_int)


def sgn(x: Union[float, int, np.ndarray]) -> Union[float, int, np.ndarray]:
	return np.sign(x)


def _test_sgn():
	assert sgn(2) == 1
	assert sgn(2.0) == 1.0

	assert sgn(-2) == -1
	assert sgn(-2.0) == -1.0

	assert sgn(0) == 0
	assert sgn(0.0) == 0.0


_unit_tests.append(_test_sgn)


def clip(val, range: Tuple[Union[float, int, None], Union[float, int, None]]):
	if (range[0] is not None) and (range[1] is not None) and (range[1] < range[0]):
		raise ValueError('range[1] must be >= range[0]')
	return np.clip(val, range[0], range[1])


def clip_in_place(val, range: Tuple[Union[float, int, None], Union[float, int, None]]) -> None:
	if (range[0] is not None) and (range[1] is not None) and (range[1] < range[0]):
		raise ValueError('range[1] must be >= range[0]')
	np.clip(val, range[0], range[1], out=val)


# Make alias for functions where clip is a function argument name that shadows
# TODO: see if there's a better way to do this
# (I don't think shadowing is actually that bad in this case, but it would be nice to have a cleaner workaround)
_clip = clip


def _test_clip():
	assert clip(1, (2, 4)) == 2
	assert clip(3, (2, 4)) == 3
	assert clip(5, (2, 4)) == 4

	assert clip(-2.0, (-1.0001, 1.0002)) == -1.0001

	for val in [-1, 0, 1, -1.0, 0.0, 1.0, 17.365, -17.365, 9999999999, 9999999999.0, -9999999999.0]:
		assert clip(val, (27.3, 27.3)) == 27.3


_unit_tests.append(_test_clip)


def lerp(vals: Tuple[Any, Any], x: float, clip=False) -> float:
	if clip:
		x = _clip(x, (0.0, 1.0))
	return (1.-x)*vals[0] + x*vals[1]


def reverse_lerp(vals: Tuple[Any, Any], y: float, clip=False) -> float:
	x = (y - vals[0]) / (vals[1] - vals[0])
	if clip:
		x = _clip(x, (0.0, 1.0))
	return x


def scale(val_in, range_in: Tuple[Any, Any], range_out: Tuple[Any, Any], clip=False) -> float:
	x = reverse_lerp(range_in, val_in, clip=clip)
	return lerp(range_out, x)


def _test_lerp():
	test_approx_equal(lerp((10, 20), 0.0), 10.0)
	test_approx_equal(lerp((10, 20), 0.5), 15.0)
	test_approx_equal(lerp((10, 20), 1.0), 20.0)
	test_approx_equal(lerp((10, 20), 1.5, clip=True), 20.0)
	test_approx_equal(lerp((10, 20), 1.5, clip=False), 25.0)
	test_approx_equal(lerp((10, 20), -0.5, clip=True), 10.0)
	test_approx_equal(lerp((10, 20), -0.5, clip=False), 5.0)

	test_approx_equal(reverse_lerp((10, 20), 10.0), 0.0)
	test_approx_equal(reverse_lerp((10, 20), 15.0), 0.5)
	test_approx_equal(reverse_lerp((10, 20), 20.0), 1.0)
	test_approx_equal(reverse_lerp((10, 20), 25.0, clip=True), 1.0)
	test_approx_equal(reverse_lerp((10, 20), 25.0, clip=False), 1.5)
	test_approx_equal(reverse_lerp((10, 20), 5.0, clip=True), 0.0)
	test_approx_equal(reverse_lerp((10, 20), 5.0, clip=False), -0.5)

	test_approx_equal(scale(10., (5., 25.), (1., 5.)), 2.)
	test_approx_equal(scale(30., (5., 25.), (1., 5.), clip=False), 6.)
	test_approx_equal(scale(30., (5., 25.), (1., 5.), clip=True), 5.)


_unit_tests.append(_test_lerp)


def log_lerp(vals: Tuple[Any, Any], x: float, clip=False) -> float:
	if clip:
		x = np.clip(x, 0.0, 1.0)
	lv = (1.-x)*math.log2(vals[0]) + x*math.log2(vals[1])
	return 2.0 ** lv


def _test_log_lerp():
	test_approx_equal(log_lerp((10., 100.), 0.0), 10.)
	test_approx_equal(log_lerp((10., 100.), 0.5), 31.62, eps=0.005)
	test_approx_equal(log_lerp((10., 100.), 1.0), 100.)
	test_approx_equal(log_lerp((10., 100.), 2.0, clip=True), 100.)
	test_approx_equal(log_lerp((10., 100.), 2.0, clip=False), 1000.)
	test_approx_equal(log_lerp((10., 100.), -1.0, clip=True), 10.)
	test_approx_equal(log_lerp((10., 100.), -1.0, clip=False), 1.)


_unit_tests.append(_test_log_lerp)


def maybe_make_integer(val: Union[float, int]):
	if isinstance(val, float) and val.is_integer():
		return int(val)
	else:
		return val


# Wrap value to range [-0.5, 0.5)
def wrap05(val):
	return (val + 0.5) % 1.0 - 0.5


def _test_wrap05():
	test_approx_equal(wrap05(0.6), -0.4)
	for val in [-0.5, -0.1, 0.0, 0.1, 0.45]:
		test_approx_equal(wrap05(val), val)
	test_approx_equal(wrap05(-0.6), 0.4)


_unit_tests.append(_test_wrap05)


def from_dB(val_dB):
	return np.power(10.0, val_dB / 20.0)


def to_dB(val_lin, min_dB: Optional[float]=None):
	"""
	:param val_lin: can be scalar or numpy array
	:param min_dB: clip to a minimum value, to prevent extremely low values and/or divide by zero
	:return: val_lin, in dB
	"""

	if min_dB is not None:
		val_lin = np.clip(val_lin, from_dB(min_dB), None)

	return 20.0*np.log10(val_lin)


def _test_dB():
	def _test(lin, dB):
		test_approx_equal(lin, from_dB(dB))
		test_approx_equal(dB, to_dB(lin))
		test_approx_equal(lin, from_dB(to_dB(lin)))
		test_approx_equal(dB, to_dB(from_dB(dB)))

	double_amp_dB = 20.0 * math.log10(2.0)
	test_approx_equal(double_amp_dB, 6.02, eps=0.005)  # Test the unit test logic itself

	_test(1.0, 0.0)
	_test(2.0, double_amp_dB)
	_test(0.5, -double_amp_dB)

	test_approx_equal(0.0, to_dB(0.0001, min_dB=0))


_unit_tests.append(_test_dB)


def rms(vec, dB=False):
	y = np.sqrt(np.mean(np.square(vec)))
	if dB:
		y = to_dB(y)
	return y


def _test_rms():
	for val in [-2.0, -1.0, 0.0, 0.0001, 1.0]:
		test_approx_equal(rms(val), abs(val))
	n_samp = 2048
	freq = 1.0 / n_samp
	test_approx_equal(rms(signal_generation.gen_sine(freq, n_samp)), 1.0 / math.sqrt(2.0), eps=1e-6)
	test_approx_equal(rms(signal_generation.gen_saw(freq, n_samp)), 1.0 / math.sqrt(3.0), eps=1e-6)
	test_approx_equal(rms(signal_generation.gen_square(freq, n_samp)), 1.0, eps=1e-6)


_unit_tests.append(_test_rms)


def normalize(vec):
	peak = np.amax(np.abs(vec))
	if peak == 0:
		return vec
	else:
		return vec / peak


def _test_normalize():
	n_samp = 2048
	freq = 1.0 / n_samp
	sig = signal_generation.gen_sine(freq, n_samp)
	test_approx_equal(normalize(sig * 0.13), sig)


_unit_tests.append(_test_normalize)


def shift_in_place(x: Union[np.ndarray, List], input_val=0.0, dir=1):
	"""Shift array 1 value to the right or left

	Like np.roll, but in-place

	:param x:
	:param input_val: new value to be added to x[0] or x[-1]
	:param dir: if positive, x[0] will be moved into x[1], etc
	:return:
	"""
	if dir == 0:
		raise ValueError('Dir must not be zero!')

	elif dir > 0:
		for n in range(len(x)-1, 0, -1):
			x[n] = x[n-1]
		x[0] = input_val

	else:
		for n in range(len(x)-1):
			x[n] = x[n+1]
		x[-1] = input_val


def _test_shift_in_place():
	from unit_test import unit_test

	x = np.array([1, 2, 3, 4, 5], dtype=np.float)

	y = np.copy(x)
	shift_in_place(y, dir=1)
	unit_test.test_approx_equal(y, [0, 1, 2, 3, 4])

	y = np.copy(x)
	shift_in_place(y, dir=-1)
	unit_test.test_approx_equal(y, [2, 3, 4, 5, 0])

	y = np.copy(x)
	shift_in_place(y, dir=1, input_val=573)
	unit_test.test_approx_equal(y, [573, 1, 2, 3, 4])

	y = np.copy(x)
	shift_in_place(y, dir=-1, input_val=573)
	unit_test.test_approx_equal(y, [2, 3, 4, 5, 573])


_unit_tests.append(_test_shift_in_place)


def derivatives(y: np.ndarray, x: np.ndarray, n_derivs=3, discontinuity_thresh=None):
	"""
	Calculate several orders of derivatives using numpy.gradient

	:param y:
	:param x:
	:param n_derivs:
	:param discontinuity_thresh: values with abs() above this will be corrected to np.nan
	:return: tuple, size matching n_derivs
	"""

	def fix_discontinuities(val):
		if np.isnan(val):
			return np.nan
		elif discontinuity_thresh is not None and abs(val) > discontinuity_thresh:
			return np.nan
		else:
			return val

	fix_discontinuities = np.vectorize(fix_discontinuities)

	derivs = []
	d = y
	for _ in range(n_derivs):
		d = np.gradient(d, x)
		d = fix_discontinuities(d)
		derivs.append(d)

	return tuple(derivs)


def test(verbose=False):
	from unit_test import unit_test
	return unit_test.run_unit_tests(_unit_tests, verbose=verbose)


def main(args):
	test(args)
