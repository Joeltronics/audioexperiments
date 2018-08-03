#!/usr/bin/env python3

import numpy as np
import scipy.io.wavfile
import os.path
import math

from processor import Processor
from typing import Union, Tuple, Any
import signal_generation


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


def approx_equal(*args, eps=0.0001, rel=False) -> bool:
	"""Compare 2 values for equality within threshold
	Meant for floats, but will work with int as well

	:param args: values to be compared
	:param eps: comparison threshold
	:param rel: if true, operation is performed in log domain, and eps is relative to log2
	:return: True if values are within eps of each other
	"""

	if len(args) == 1:
		if np.isscalar(args[0]):
			raise ValueError('Must give array_like, or more than 1 argument!')
		else:
			val1, val2 = np.amin(args[0]), np.amax(args[0])
	elif len(args) == 2:
		val1, val2 = args
	else:
		val1, val2 = np.amin(args), np.amax(args)

	if rel:
		if val1 == 0.0 or val2 == 0.0:
			raise ZeroDivisionError("Cannot call approx_equal(rel=True) for value 0")

		if (val1 > 0) != (val2 > 0):
			return False

		val1 = math.log2(abs(val1))
		val2 = math.log2(abs(val2))

	return abs(val1 - val2) < eps


def sgn(x: Union[float, int, np.ndarray]) -> Union[float, int, np.ndarray]:
	return np.sign(x)


def clip(val, range: Tuple[Any, Any]):
	if range[1] < range[0]:
		raise ValueError('range[1] must be > range[0]')
	return np.clip(val, range[0], range[1])


def lerp(vals: Tuple[Any, Any], x: float, clip=False):
	if clip:
		x = np.clip(x, 0.0, 1.0)
	return (1.-x)*vals[0] + x*vals[1]


def log_lerp(vals: Tuple[Any, Any], x: float, clip=False) -> float:
	if clip:
		x = np.clip(x, 0.0, 1.0)
	lv = (1.-x)*math.log2(vals[0]) + x*math.log2(vals[1])
	return 2.0 ** lv


# Wrap value to range [-0.5, 0.5)
def wrap05(val):
	return (val + 0.5) % 1.0 - 0.5


# inline unit tests
assert approx_equal(wrap05(0.6), -0.4)
assert approx_equal(wrap05(-0.6), 0.4)


def to_dB(val_lin: Union[float, int]) -> float:
	return 20.0*np.log10(val_lin)


def from_dB(val_dB: Union[float, int]) -> float:
	return np.power(10.0, val_dB / 20.0)


def rms(vec, dB=False):
	y = np.sqrt(np.mean(np.square(vec)))
	if dB:
		y = to_dB(y)
	return y


def normalize(vec):
	peak = np.amax(np.abs(vec))
	if peak == 0:
		return vec
	else:
		return vec / peak


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
