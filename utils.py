#!/usr/bin/env python3

import numpy as np
import scipy.io.wavfile
import os.path
import math

from processor import Processor


def to_pretty_str(val):
	if type(val) == float:
		s = '%.6f' % val
		while s.endswith('0'):
			s = s[:-1]
		if s.endswith('.'):
			s = s + '0'
		return s
	else:
		return str(val)


def approx_equal(val1, val2, eps=0.0001):
	return abs(val1 - val2) < eps


def approx_equal_rel(val1, val2, eps=0.0001):
	if val1 == 0.0 or val2 == 0.0:
		raise ZeroDivisionError("Cannot call approx_equal_rel for value == 0")
	
	if (val1 > 0) != (val2 > 0):
		return False

	return approx_equal(
		math.log2(abs(val1)),
		math.log2(abs(val2)),
		eps=eps)


def sgn(x):
	return np.sign(x)


def lerp(v1, v2, x):
	return (1.-x)*v1 + x*v2


def log_lerp(v1, v2, x):
	lv = (1.-x)*math.log(v1) + x*math.log(v2)
	return math.exp(lv)


def clip(val, range):
	if val < range[0]:
		return range[0]
	elif val > range[1]:
		return range[1]
	else:
		return val


# Wrap value to range [-0.5, 0.5)
def wrap05(val):
	return (val + 0.5) % 1.0 - 0.5


assert approx_equal(wrap05(0.6), -0.4)
assert approx_equal(wrap05(-0.6), 0.4)


def to_dB(val_lin):
	return 20.0*np.log10(val_lin)


def from_dB(val_dB):
	return np.power(10.0, val_dB/20.0)


def rms(vec, dB=False):
	y = np.sqrt(np.mean(np.square(vec)))
	if dB:
		y = to_dB(y)
	return y


def phase_to_sine(phase):
	return np.sin(phase * 2.0 * math.pi)


def phase_to_cos(phase):
	return phase_to_sine((phase + 0.25) % 1.0)


def gen_phase(freq, n_samp, start_phase=0.0):
	
	if (freq <= 0.0) or (freq >= 0.5):
		print("Warning: freq out of range %f" % freq)

	# This could be vectorized
	ph = np.zeros(n_samp)
	phase = start_phase

	# TODO: vectorize this
	for n in range(n_samp):
		ph[n] = phase
		phase += freq
	
	ph = np.mod(ph, 1.0)
	
	return ph


def gen_sine(freq, n_samp, start_phase=0.0):
	phase = gen_phase(freq, n_samp, start_phase=start_phase)
	return phase_to_sine(phase)


def gen_square(freq, n_samp):
	p = gen_phase(freq, n_samp)
	return ((p >= 0.5) * 2 - 1).astype(float)


def normalize(vec):
	peak = np.amax(np.abs(vec))
	return vec / peak


def import_wavfile(filename):
	if not os.path.exists(filename):
		raise FileNotFoundError(filename)
	
	sample_rate, data = scipy.io.wavfile.read(filename)

	# Convert to range (-1,1)

	if data.dtype == np.dtype('int8'):
		return sample_rate, data.astype('float') / 128.0
	elif data.dtype == np.dtype('uint8'):
		return sample_rate, (data.astype('float') - 128.0) / 128.0
	elif data.dtype == np.dtype('int16'):
		return sample_rate, data.astype('float') / float(2**15)
	elif data.dtype == np.dtype('int32'):
		return sample_rate, data.astype('float') / float(2**31)
	elif data.dtype == np.dtype('float'):
		return sample_rate, data
	else:
		raise ValueError('Unknown data type: %s' % data.dtype)


def gen_cos_sine(freq, n_samp):
	ph = gen_phase(freq, n_samp)
	return phase_to_cos(ph), phase_to_sine(ph)


def _single_freq_dft(x, cos_sig, sin_sig, return_mag_phase=True):
	# TODO: use Goertzel algo instead

	dft_mult = cos_sig - 1j*sin_sig
	
	xs = x * dft_mult
	xs = 2.0 * np.mean(xs)
	
	if return_mag_phase:
		return np.abs(xs), np.angle(xs)
	else:
		return xs


def single_freq_dft(x: np.array, freq: float, return_mag_phase=True):

	ph_sin = gen_phase(freq, len(x))
	ph_cos = (ph_sin + 0.25) % 1.0

	cos_sig, sin_sig = gen_cos_sine(freq, len(x))
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

		x_cos, x_sin = gen_cos_sine(f_norm, n_samp_this_freq)

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
