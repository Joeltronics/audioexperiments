#!/usr/bin/env python3

import numpy as np
import scipy.io.wavfile
import os.path
import math

from processor import Processor
from typing import Union, Tuple, Any
import signal_generation
from approx_equal import *

_unit_tests = []


def to_pretty_str(val) -> str:
	"""Convert float into nicely formatted string

	If another type is given, just calls str(val)
	"""
	if type(val) == float:
		s = '%.6f' % val
		while s.endswith('0'):
			s = s[:-1]
		if s.endswith('.'):
			s = s + '0'
		return s
	else:
		return str(val)


def _test_to_pretty_str():
	assert to_pretty_str(1) == '1'
	assert to_pretty_str(0) == '0'
	assert to_pretty_str(-12345) == '-12345'
	assert to_pretty_str(1.00000001) == '1.0'
	assert to_pretty_str(0.0) == '0.0'
	assert to_pretty_str(0.00000001) == '0.0'


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
	import signal_generation
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
	import signal_generation
	n_samp = 2048
	freq = 1.0 / n_samp
	sig = signal_generation.gen_sine(freq, n_samp)
	assert approx_equal(normalize(sig * 0.13), sig)


_unit_tests.append(_test_normalize)


def import_wavfile(filename) -> Tuple[np.ndarray, int]:
	"""Import wav file and normalize to float values in range [-1, 1)

	:param filename:
	:return: data, sample rate (Hz)
	"""
	if not os.path.exists(filename):
		raise FileNotFoundError(filename)
	
	sample_rate, data = scipy.io.wavfile.read(filename)

	# Convert to range (-1,1)

	if data.dtype == np.dtype('int8'):
		data = data.astype('float') / 128.0
	elif data.dtype == np.dtype('uint8'):
		data = (data.astype('float') - 128.0) / 128.0
	elif data.dtype == np.dtype('int16'):
		data = data.astype('float') / float(2**15)
	elif data.dtype == np.dtype('int32'):
		data = data.astype('float') / float(2**31)
	elif data.dtype == np.dtype('float'):
		pass
	else:
		raise ValueError('Unknown data type: %s' % data.dtype)

	return data, sample_rate


def _single_freq_dft(x, cos_sig, sin_sig, return_mag_phase=True):
	# TODO: use Goertzel algo instead

	dft_mult = cos_sig - 1j*sin_sig
	
	xs = x * dft_mult
	xs = 2.0 * np.mean(xs)
	
	if return_mag_phase:
		return np.abs(xs), np.angle(xs)
	else:
		return xs


def single_freq_dft(x: np.ndarray, freq: float, return_mag_phase=True):
	cos_sig, sin_sig = signal_generation.gen_cos_sine(freq, len(x))
	return _single_freq_dft(x, cos_sig, sin_sig, return_mag_phase=return_mag_phase)


def phase_to_group_delay(freqs, phases_rad, sample_rate):
	phases_rad_unwrapped = np.unwrap(phases_rad)

	freqs_cycles_per_sample = freqs / sample_rate
	freqs_rads_per_sample = freqs_cycles_per_sample * 2.0 * math.pi

	np_version = [int(n) for n in np.__version__.split('.')]
	if np_version[0] <= 1 and np_version[1] < 13:
		delay_samples = -np.gradient(phases_rad_unwrapped) / np.gradient(freqs_rads_per_sample)
	else:
		delay_samples = -np.gradient(phases_rad_unwrapped, freqs_rads_per_sample)

	delay_seconds = delay_samples / sample_rate

	return delay_seconds


def get_freq_response(system: Processor, freqs, sample_rate, n_samp=None, n_cycles=40.0, amplitude=1.0, throw_if_nonlinear=False, group_delay=False):
	# n_samp overrides n_cycles

	mags = np.zeros(len(freqs))
	phases = np.zeros(len(freqs))

	for n, freq in enumerate(freqs):
		system.reset()

		f_norm = freq / sample_rate

		if n_samp is None:
			n_samp_this_freq = math.ceil(n_cycles / f_norm)
		else:
			n_samp_this_freq = n_samp

		x_cos, x_sin = signal_generation.gen_cos_sine(f_norm, n_samp_this_freq)

		# Use sine signal since it starts at 0 - less prone to problems from nonlinearities
		x = x_sin
		y = system.process_vector(x * amplitude) / amplitude

		mag, ph = _single_freq_dft(y, x_cos, x_sin, return_mag_phase=True)
		
		# Add 90 degrees because we used the sine as input
		ph += 0.5*np.pi

		mags[n] = mag
		phases[n] = ph

		if throw_if_nonlinear:
			# Compare RMS to mag, to see if output was non-sinusoidal
			# (if linear, rms_mag should be very close to mags[n])
			rms_mag = rms(y) * math.sqrt(2.0)

			mag_dB = to_dB(mag)
			rms_dB = to_dB(rms_mag)

			# This method isn't perfect. The further away from 0 dB, the more potential for numerical error.
			if not approx_equal(mag, rms_mag, eps=(max(0.05, abs(rms_mag)/30.))):
				raise Exception(
					'Non-linear function! (frequency %.1f, sample rate %.1f, DFT magnitude %f dB, RMS magnitude %f dB, diff %f dB)' %
					(freq, sample_rate, mag_dB, rms_dB, abs(mag_dB - rms_dB)))

	if group_delay:
		group_delay = phase_to_group_delay(freqs, phases, sample_rate)
		return mags, phases, group_delay

	else:
		return mags, phases


if __name__ == "__main__":
	import unit_test
	unit_test.run_unit_tests(_unit_tests)
