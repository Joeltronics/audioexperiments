#!/usr/bin/env python3

from typing import Tuple, Any, List
from generation import signal_generation
from .approx_equal import *

_unit_tests = []


def to_pretty_str(val, num_decimals=6, point_zero=True) -> str:
	"""Convert float into nicely formatted string

	If another type is given, just calls str(val)
	"""

	if isinstance(val, float) or isinstance(val, np.floating):
		format = '%%.%if' % num_decimals
		s = format % val
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


def clip(val, range: Tuple[Any, Any]):
	if range[1] < range[0]:
		raise ValueError('range[1] must be >= range[0]')
	return np.clip(val, range[0], range[1])


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
	assert approx_equal(lerp((10, 20), 0.0), 10.0)
	assert approx_equal(lerp((10, 20), 0.5), 15.0)
	assert approx_equal(lerp((10, 20), 1.0), 20.0)
	assert approx_equal(lerp((10, 20), 1.5, clip=True), 20.0)
	assert approx_equal(lerp((10, 20), 1.5, clip=False), 25.0)
	assert approx_equal(lerp((10, 20), -0.5, clip=True), 10.0)
	assert approx_equal(lerp((10, 20), -0.5, clip=False), 5.0)

	assert approx_equal(reverse_lerp((10, 20), 10.0), 0.0)
	assert approx_equal(reverse_lerp((10, 20), 15.0), 0.5)
	assert approx_equal(reverse_lerp((10, 20), 20.0), 1.0)
	assert approx_equal(reverse_lerp((10, 20), 25.0, clip=True), 1.0)
	assert approx_equal(reverse_lerp((10, 20), 25.0, clip=False), 1.5)
	assert approx_equal(reverse_lerp((10, 20), 5.0, clip=True), 0.0)
	assert approx_equal(reverse_lerp((10, 20), 5.0, clip=False), -0.5)

	assert approx_equal(scale(10., (5., 25.), (1., 5.)), 2.)
	assert approx_equal(scale(30., (5., 25.), (1., 5.), clip=False), 6.)
	assert approx_equal(scale(30., (5., 25.), (1., 5.), clip=True), 5.)


_unit_tests.append(_test_lerp)


def log_lerp(vals: Tuple[Any, Any], x: float, clip=False) -> float:
	if clip:
		x = np.clip(x, 0.0, 1.0)
	lv = (1.-x)*math.log2(vals[0]) + x*math.log2(vals[1])
	return 2.0 ** lv


def _test_log_lerp():
	assert approx_equal(log_lerp((10., 100.), 0.0), 10.)
	assert approx_equal(log_lerp((10., 100.), 0.5), 31.62, eps=0.005)
	assert approx_equal(log_lerp((10., 100.), 1.0), 100.)
	assert approx_equal(log_lerp((10., 100.), 2.0, clip=True), 100.)
	assert approx_equal(log_lerp((10., 100.), 2.0, clip=False), 1000.)
	assert approx_equal(log_lerp((10., 100.), -1.0, clip=True), 10.)
	assert approx_equal(log_lerp((10., 100.), -1.0, clip=False), 1.)


_unit_tests.append(_test_log_lerp)


# Wrap value to range [-0.5, 0.5)
def wrap05(val):
	return (val + 0.5) % 1.0 - 0.5


def _test_wrap05():
	assert approx_equal(wrap05(0.6), -0.4)
	for val in [-0.5, -0.1, 0.0, 0.1, 0.45]:
		assert approx_equal(wrap05(val), val)
	assert approx_equal(wrap05(-0.6), 0.4)


_unit_tests.append(_test_wrap05)


def to_dB(val_lin: Union[float, int]) -> float:
	return 20.0*np.log10(val_lin)


def from_dB(val_dB: Union[float, int]) -> float:
	return np.power(10.0, val_dB / 20.0)


def _test_dB():
	def test(lin, dB):
		assert approx_equal(lin, from_dB(dB))
		assert approx_equal(dB, to_dB(lin))
		assert approx_equal(lin, from_dB(to_dB(lin)))
		assert approx_equal(dB, to_dB(from_dB(dB)))

	double_amp_dB = 20.0 * math.log10(2.0)
	assert approx_equal(double_amp_dB, 6.02, eps=0.005)  # Test the unit test logic itself

	test(1.0, 0.0)
	test(2.0, double_amp_dB)
	test(0.5, -double_amp_dB)


_unit_tests.append(_test_dB)


def rms(vec, dB=False):
	y = np.sqrt(np.mean(np.square(vec)))
	if dB:
		y = to_dB(y)
	return y


def _test_rms():
	for val in [-2.0, -1.0, 0.0, 0.0001, 1.0]:
		assert approx_equal(rms(val), abs(val))
	n_samp = 2048
	freq = 1.0 / n_samp
	assert approx_equal(rms(signal_generation.gen_sine(freq, n_samp)), 1.0 / math.sqrt(2.0))
	assert approx_equal(rms(signal_generation.gen_saw(freq, n_samp)), 1.0 / math.sqrt(3.0))
	assert approx_equal(rms(signal_generation.gen_square(freq, n_samp)), 1.0)


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
	assert approx_equal(normalize(sig * 0.13), sig)


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


def test(verbose=False):
	from unit_test import unit_test
	return unit_test.run_unit_tests(_unit_tests, verbose=verbose)


def main(args):
	test(args)
