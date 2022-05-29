#!/usr/bin/env python3


"""
See freq_response.md for details
"""

import argparse
from dataclasses import dataclass
import fractions
import math
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from analysis import linearity, time_domain_response
from utils import utils
from unit_test import unit_test
from processor import ProcessorBase
from generation import signal_generation


PI = math.pi
HALF_PI = 0.5 * math.pi
TWOPI = 2.0 * math.pi

SQRT2 = math.sqrt(2.0)
INV_SQRT2 = 1.0 / SQRT2

# Square wave has THD+N of sqrt(pi^2 / 8 - 1) ~= 0.483 ~= -6.32 dB
# https://en.wikipedia.org/wiki/Total_harmonic_distortion#Examples
SQUARE_THDN = utils.to_dB(math.sqrt((math.pi ** 2.) / 8 - 1))


_unit_tests_short = []
_unit_tests_full = []


@dataclass
class FreqResponse:
	freqs: np.ndarray
	sample_rate: float

	amplitude: Optional[float] = None  # Amplitude frequency response was performed at (relevant for nonlinear systems)

	mag: Optional[np.ndarray] = None  # Magnitude
	rms: Optional[np.ndarray] = None  # RMS response (only relevant for nonlinear system)
	phase: Optional[np.ndarray] = None  # Phase response, in radians
	group_delay: Optional[np.ndarray] = None
	thdn: Optional[np.ndarray] = None  # THD + Noise (linear, not dB)


def dft_num_samples(
		freq: Union[int, float],
		sample_rate: Union[int, float],
		min_num_samples=0,
		max_num_samples: Optional[int]=None,
		maximize=False,
		round_up=False,
		) -> int:
	"""
	Determine optimum DFT size at a given frequency and sample rate, in order to get an exact number of cycles at the
	frequency, or as close as possible.

	:param freq: frequency, in whatever units you want (must be same units as sample rate)

	:param sample_rate: sample rate (in same units as freq)

	:param min_num_samples:
		Minimum number of samples; default 0 (i.e. no minimum).
		Actual practical minimum will always be at least 1 cycle

	:param max_num_samples:
		Maximum number of samples; default is sample_rate or period at frequency, whichever is larger
		Must be > (sample_rate/freq).
		Must be specified if maximize

	:param maximize:
		By default, will come up with the minimum possible number of samples that satisfies the criteria sequence;
		if maximize, will come up with the longest instead.
		Must explicitly specify max_num_samples if maximize

	:param round_up:
		if True, will always round up instead of rounding to nearest
	"""

	if maximize and not max_num_samples:
		raise ValueError('Must provide max_num_samples if setting maximize')

	period = sample_rate / freq

	if min_num_samples < 0:
		raise ValueError('min_num_samples must be > 0')
	elif not isinstance(min_num_samples, int):
		raise ValueError('min_num_samples must be integer')

	if max_num_samples is None:
		max_num_samples = int(math.ceil(max(sample_rate, period)))
	elif not isinstance(max_num_samples, int):
		raise ValueError('max_num_samples must be integer')
	elif max_num_samples <= 0:
		raise ValueError('max_num_samples (%g) must be > 0' % max_num_samples)
	elif max_num_samples <= min_num_samples:
		raise ValueError('max_num_samples (%g) must be > min_num_samples (%g)' % (max_num_samples, min_num_samples))

	eps = 1e-12
	min_num_cycles = max(min_num_samples / period, 1.)
	min_num_cycles_int = int(math.ceil(min_num_cycles - eps))

	max_num_cycles = max_num_samples / period
	max_num_cycles_int = int(math.floor(max_num_cycles + eps))

	if max_num_cycles_int < 1:
		assert max_num_samples < period
		raise ValueError('max_num_samples (%u) must be >= period (%g)' % (max_num_samples, period))

	assert min_num_cycles_int * (period + eps) >= min_num_samples
	assert max_num_cycles_int * (period - eps) <= max_num_samples

	if max_num_cycles_int == min_num_cycles_int:
		# Special case: only 1 possible number of periods
		n_samples = max_num_cycles_int * period
		n_samples = int(math.ceil(n_samples) if round_up else round(n_samples))
		assert min_num_samples <= n_samples <= max_num_samples
		return n_samples

	elif max_num_cycles_int < min_num_cycles_int:
		# TODO: come up with good error message for this
		raise ValueError('freq %g, SR %g, min_num_cycles %f -> %u, max_num_cycles %f -> %u' % (
			freq, sample_rate,
			min_num_cycles, min_num_cycles_int, max_num_cycles, max_num_cycles_int
		))

	assert max_num_samples >= period  # Should be guaranteed by above conditions

	freq = utils.integerize_if_int(freq)
	sample_rate = utils.integerize_if_int(sample_rate)

	if isinstance(freq, int) and isinstance(sample_rate, int):
		period_as_fraction = fractions.Fraction(sample_rate, freq)
	else:
		period_as_fraction = fractions.Fraction.from_float(period)

	period_as_fraction = period_as_fraction.limit_denominator(max_denominator=max_num_cycles_int)

	n_samples_ideal = period * period_as_fraction.denominator
	assert utils.approx_equal(period_as_fraction.numerator, n_samples_ideal, eps=0.5)

	if maximize:
		if 2*n_samples_ideal <= max_num_samples:
			"""
			What's the largest integer we can multiply n_samples_ideal by to still be <= max_num_samples?
	
			n * k <= max
			k <= max / n
			k = floor(max / n)
			"""
			n_samples_ideal *= math.floor(max_num_samples / n_samples_ideal)

	elif n_samples_ideal < min_num_samples:
		"""
		What's the smallest integer we can multiply n_samples_ideal by to be >= min_num_samples?

		n * k >= min
		k >= min / n
		k = ceil(min / n)
		"""
		n_samples_ideal *= math.ceil(min_num_samples / n_samples_ideal)

	n_samples = int(math.ceil(n_samples_ideal) if round_up else round(n_samples_ideal))

	if not (min_num_samples <= n_samples <= max_num_samples):
		raise AssertionError('Check n_samples (%i, from %g, fraction %s) in range (%i, %i) failed!' % (
			n_samples, n_samples_ideal, period_as_fraction, min_num_samples, max_num_samples))

	return n_samples


def _test_dft_num_samples():
	from unit_test.unit_test import test_equal, test_threw

	"""
	Perfect divisors
	"""

	# 1 kHz @ 96 kHz
	# 1 period = 96 samples
	test_equal(dft_num_samples(1000, 96000), 96)
	test_equal(dft_num_samples(1000, 96000.), 96)
	test_equal(dft_num_samples(1000., 96000), 96)
	test_equal(dft_num_samples(1000., 96000.), 96)

	test_equal(dft_num_samples(1000., 96000., min_num_samples=100), 192)
	test_equal(dft_num_samples(1000., 96000., min_num_samples=384), 384)
	test_equal(dft_num_samples(1000., 96000., max_num_samples=400, maximize=True), 384)

	test_equal(dft_num_samples(1000., 96000., min_num_samples=380, max_num_samples=400), 384)
	test_threw(dft_num_samples, 1000., 96000., min_num_samples=398, max_num_samples=400)

	# 3.125 (25/8) @ 96 kHz
	# 1 period = 30,720 samples
	test_equal(dft_num_samples(3.125, 96000.), 30720)

	"""
	Rational numbers
	"""

	# 10 kHz @ 96 kHz
	# 1 period = 9.6 samples (48/5)
	test_equal(dft_num_samples(10000, 96000), 48)
	test_equal(dft_num_samples(10000, 96000, maximize=True, max_num_samples=96000), 96000)

	# 1 kHz @ 44.1 kHz
	# 1 period = 44.1 samples (441/10)
	test_equal(dft_num_samples(1000, 44100), 441)
	test_equal(dft_num_samples(1000, 44100, maximize=True, max_num_samples=44100), 44100)

	# 440 Hz @ 44.1 kHz
	# 1 period = 100.2272727 samples (2205/22)
	test_equal(dft_num_samples(440, 44100), 2205)
	test_equal(dft_num_samples(440, 44100, maximize=True, max_num_samples=44100), 44100)
	test_equal(dft_num_samples(440, 44100, max_num_samples=102), 100)
	test_equal(dft_num_samples(440, 44100, max_num_samples=102, round_up=True), 101)
	test_equal(dft_num_samples(440, 44100, max_num_samples=510, maximize=True), 401)
	test_equal(dft_num_samples(440, 44100, max_num_samples=510, round_up=True, maximize=True), 401)

	# 100.125 Hz @ 96 kHz
	# 1 period = 958.80 samples (256000/267)
	test_equal(dft_num_samples(100.125, 96000, max_num_samples=1000000), 256000)
	test_equal(dft_num_samples(100.125, 96000, max_num_samples=1000000, maximize=True), 768000)
	test_equal(dft_num_samples(100.125, 96000), 92045)

	# 3010 Hz @ 96 kHz
	# 1 period = 31.89 samples (9600/301)
	test_equal(dft_num_samples(3010, 96000), 9600)
	test_equal(dft_num_samples(3010, 96000, maximize=True, max_num_samples=96000), 96000)

	# 1001 Hz @ 96 kHz (coprime)
	# 1 period = 95.904 samples (96000/1001)
	test_equal(dft_num_samples(1001, 96000), 96000)
	test_equal(dft_num_samples(1001, 96000, maximize=True, max_num_samples=96000), 96000)

	# 1000.1 Hz @ 96 kHz
	# 1 period = 95.99 samples (960,000/10,001)
	test_equal(dft_num_samples(1000.1, 96000), 59994)
	test_equal(dft_num_samples(1000.1, 96000, maximize=True, max_num_samples=96000), 59994)

	"""
	Irrational numbers
	"""

	# 1000*pi Hz @ 96 kHz
	# 1 period = 30.5577 samples
	test_equal(dft_num_samples(1000*PI, 96000), 30955)

	"""
	Rational numbers expressed as ratio of 2 irrational numbers
	"""

	test_equal(dft_num_samples(1000*PI, 96000*PI), 96)


_unit_tests_short.append(_test_dft_num_samples)
_unit_tests_full.append(_test_dft_num_samples)


def _single_freq_dft(
		x: np.ndarray,
		cos_sig: np.ndarray,
		sin_sig: np.ndarray,
		freq: Union[int, float],
		sample_rate: Union[int, float],
		mag=False,
		phase=False,
		adjust_num_samp=False,
		normalize=False):
	# TODO: use Goertzel algo instead

	# FIXME: properly deal with boundary conditions - i.e. extra samples at end that don't fit into a complete cycle
	# adjust_num_samp should mostly deal with that

	if adjust_num_samp:
		n_samp = dft_num_samples(freq, sample_rate, min_num_samples=(len(x) // 2), max_num_samples=len(x), maximize=True)
	else:
		n_samp = len(x)

	dft_mult = cos_sig[:n_samp] - 1j * sin_sig[:n_samp]

	xs = x[:n_samp] * dft_mult
	xs = np.mean(xs) if normalize else sum(xs)

	if mag and phase:
		return np.abs(xs), np.angle(xs)
	elif mag:
		return np.abs(xs)
	elif phase:
		return np.angle(xs)
	else:
		return xs


def single_freq_dft(
		x: np.ndarray,
		freq: float,
		sample_rate=1.0,
		mag=True,
		phase=True,
		adjust_num_samp=False,
		normalize=False):
	"""
	Perform DFT at a single arbitrary frequency

	:param x:
	:param freq:
	:param sample_rate:
	:param mag: return magnitude
	:param phase: return phase

	:param adjust_num_samp:
		if True, will not perform DFT on entire signal; rather, will find optimal number of samples to get as close
		to a zero-crossing as possible (though guaranteed to use at least half the samples).

		Recommend calling dft_num_samples to determine sample size instead, in order to get the optimal DFT size of the
		signal in the first place.

	:param normalize: divide by number of samples, i.e. return average power per sample instead of sum

	:return: (mag, phase) if mag and phase; magnitude if mag only; phase if phase only; complex result if neither
	"""
	cos_sig, sin_sig = signal_generation.gen_cos_sine(freq / sample_rate, len(x))
	return _single_freq_dft(
		x, cos_sig, sin_sig, freq, sample_rate,
		mag=mag, phase=phase, adjust_num_samp=adjust_num_samp, normalize=normalize)


def phase_to_group_delay(freqs: np.ndarray, phases_rad: np.ndarray, sample_rate: float) -> np.ndarray:
	phases_rad_unwrapped = np.unwrap(phases_rad)

	freqs_cycles_per_sample = freqs / sample_rate
	freqs_rads_per_sample = freqs_cycles_per_sample * TWOPI

	np_version = [int(n) for n in np.__version__.split('.')]
	if np_version[0] <= 1 and np_version[1] < 13:
		delay_samples = -np.gradient(phases_rad_unwrapped) / np.gradient(freqs_rads_per_sample)
	else:
		delay_samples = -np.gradient(phases_rad_unwrapped, freqs_rads_per_sample)

	delay_seconds = delay_samples / sample_rate

	return delay_seconds


def get_ir_freq_response(
		ir: np.ndarray,
		freqs: Iterable,
		sample_rate,
		mag=True,
		phase=True,
		group_delay=True) -> FreqResponse:
	"""
	Calculate frequency response based on impulse response

	:param ir: Impulse response

	:param freqs: frequencies to get response at. More frequencies will also lead to more precise group delay

	:param sample_rate: sample rate, in Hz

	:param mag: if False, does not calculate nor return magnitude
	:param rms: if False, does not calculate nor return RMS magnitude
	:param phase: if False, does not calculate nor return phase
	:param group_delay: if False, does not calculate nor return group delay

	:return: frequency response of system
	"""

	if group_delay and not phase:
		raise ValueError('Must calculate phase to calculate group delay!')

	freqs = np.array(freqs)

	freq_resp = FreqResponse(freqs=freqs, sample_rate=sample_rate)

	if mag:
		freq_resp.mag = np.zeros(len(freqs))

	if phase:
		freq_resp.phase = np.zeros(len(freqs))

	for n, f_norm in enumerate(freqs / sample_rate):

		ret = single_freq_dft(ir, f_norm, mag=mag, phase=phase, adjust_num_samp=True)
		if mag:
			freq_resp.mag[n] = ret[0]

		if phase:
			freq_resp.phase[n] = ret[-1]

	if group_delay:
		freq_resp.group_delay = phase_to_group_delay(freqs, freq_resp.phase, sample_rate)

	if phase:
		freq_resp.phase = ((freq_resp.phase + PI) % TWOPI) - PI

	return freq_resp


def _calc_thdn(y, f_norm, mag, phase, debug_assert=False):

	# Subtract fundamental from signal

	phase01 = np.mod(phase / TWOPI, 1.0)
	fundamental = signal_generation.gen_sine(f_norm, n_samp=len(y), start_phase=phase01) * mag

	if debug_assert:
		debug_mag, debug_phase = single_freq_dft(fundamental, f_norm, mag=True, phase=True, normalize=True, adjust_num_samp=False)
		assert utils.approx_equal(debug_mag, mag, eps=0.001)
		assert utils.approx_equal(debug_phase, phase, eps=0.01)

	thdn_sig = y - fundamental

	return utils.rms(thdn_sig) * SQRT2 / mag


def get_discrete_sine_sweep_freq_response(
		system: ProcessorBase,
		freqs: Iterable,
		sample_rate,
		n_cycles=40.0,
		n_samp_min: Optional[int]=None,
		n_samp=None,
		amplitude=1.0,
		mag=True,
		rms=True,
		phase=True,
		group_delay=None,
		thdn=None) -> FreqResponse:
	"""
	Calculate frequency response by passing sine waves at various frequencies through system

	Unlike impulse response analysis, this will work for nonlinear systems as well
	(Of course, the definition of "frequency response" is ill-defined for a nonlinear system - see freq_response.md)

	:param system: Processor to process
	:param freqs: frequencies to get response at. More frequencies will also lead to more precise group delay
	:param sample_rate: sample rate, in Hz
	:param n_cycles: how many cycles of waveform to calculate over
	:param n_samp_min: if using n_cycles, minimum n_samp
	:param n_samp: how many samples to calculate over - overrides n_cycles
	:param amplitude: amplitude of sine wave to pass in

	:param mag: if False, does not calculate nor return magnitude
	:param rms: if False, does not calculate nor return RMS magnitude
	:param phase: if False, does not calculate nor return phase
	:param group_delay: if False, does not calculate nor return group delay; default true if phase, else false
	:param thdn: if False, does not calculate THD+Noise; default true if mag & phase, else false

	:return:
		frequency response of system.
		mag, phase, and group delay are based on measurement of output at only that frequency.
		RMS is based on entire signal.
		So you can get a proxy for "how nonlinear" the system is by comparing difference between mag & RMS
		(if linear, output would be a sine wave, so RMS would be 1/sqrt(2) of magnitude)
	"""

	if group_delay is None:
		group_delay = phase
	elif group_delay and not phase:
		raise ValueError('Must calculate phase to calculate group delay!')

	if thdn is None:
		thdn = mag and phase
	elif thdn and (not mag or not phase):
		raise ValueError('Must calculate magnitude/phase to calculate THD+N')

	freqs = np.array(freqs)

	freq_resp = FreqResponse(freqs=freqs, sample_rate=sample_rate)

	if mag:
		freq_resp.mag = np.zeros(len(freqs))

	if rms:
		freq_resp.rms = np.zeros(len(freqs))

	if phase:
		freq_resp.phase = np.zeros(len(freqs))

	if thdn:
		freq_resp.thdn = np.zeros(len(freqs))

	debug_use_dft_of_input = True  # FIXME: false is broken

	for n, freq in enumerate(freqs):

		f_norm = freq / sample_rate
		period = sample_rate / freq

		if n_samp is None:
			max_num_samples = int(math.ceil(max(n_cycles * period, sample_rate)))
			#n_samp_this_freq = max(math.ceil(n_cycles / f_norm), n_samp_min)
			n_samp_this_freq = dft_num_samples(
				freq, sample_rate,
				min_num_samples=n_samp_min if (n_samp_min is not None) else 0,
				max_num_samples=max_num_samples)
		else:
			n_samp_this_freq = n_samp

		scaling = 2.0 / n_samp_this_freq

		# Input is actually double the number of samples, but for output we only take the 2nd half
		# TODO: be smarter about this, actually watch the output and wait for the system to hit steady-state

		# TODO: Can we reach steady-state faster if we ramp up the amplitude?
		#       (This would avoid the sudden impulse in 2nd, 3rd, etc derivatives)

		x_cos_full, x_sin_full = signal_generation.gen_cos_sine(f_norm, 2 * n_samp_this_freq)
		x_cos = x_cos_full[n_samp_this_freq:]
		x_sin = x_sin_full[n_samp_this_freq:]

		x_cos_dft_mag = x_cos_dft_phase = x_sin_dft_mag = x_sin_dft_phase = None
		if debug_use_dft_of_input:
			x_cos_dft_mag, x_cos_dft_phase = _single_freq_dft(x_cos, x_cos, x_sin, freq, sample_rate, mag=True, phase=True, adjust_num_samp=False)
			x_sin_dft_mag, x_sin_dft_phase = _single_freq_dft(x_sin, x_cos, x_sin, freq, sample_rate, mag=True, phase=True, adjust_num_samp=False)
			x_rms = utils.rms(x_sin) if rms else None
		else:
			x_cos_dft_mag = 1.0
			x_cos_dft_phase = HALF_PI
			x_sin_dft_mag = 1.0
			x_sin_dft_phase = 0
			x_rms = INV_SQRT2

		# TODO: remove y_cos once we know we don't need it anymore
		system.reset()
		y_cos = system.process_vector(x_cos_full * amplitude)[n_samp_this_freq:] / amplitude

		system.reset()
		y_sin = system.process_vector(x_sin_full * amplitude)[n_samp_this_freq:] / amplitude

		# TODO: use this? the results look really good - in some cases they look even better than impulse response results
		# (That doesn't really make sense though - IR should be perfect for linear?)
		#mag_sin_cos = np.sqrt(np.square(y_sin) + np.square(y_cos))

		if mag or phase:
			ret = _single_freq_dft(y_sin, x_cos, x_sin, freq, sample_rate, mag=mag, phase=phase, adjust_num_samp=False)

			if mag:
				freq_resp.mag[n] = ret[0] if (mag and phase) else ret

				if x_sin_dft_mag is not None:
					freq_resp.mag[n] /= x_sin_dft_mag
				else:
					freq_resp.mag[n] *= scaling

			if phase:
				# TODO: figure out if should use both sin & cos
				# TODO: use x_sin_dft_phase
				freq_resp.phase[n] = (ret[1] if (mag and phase) else ret) + HALF_PI

		if rms:
			freq_resp.rms[n] = utils.rms(y_sin) / x_rms

		if thdn:
			freq_resp.thdn[n] = _calc_thdn(
				y=y_sin,
				f_norm=f_norm,
				mag=freq_resp.mag[n],
				phase=freq_resp.phase[n])

	if group_delay:
		freq_resp.group_delay = phase_to_group_delay(freqs, freq_resp.phase, sample_rate)

	if phase:
		freq_resp.phase = ((freq_resp.phase + PI) % TWOPI) - PI

	return freq_resp


def _test_thdn():
	pass  # TODO


#_unit_tests_short.append(_test_thdn)
#_unit_tests_full.append(_test_thdn)


def get_white_noise_response(
		system: ProcessorBase,
		freqs: Iterable,
		sample_rate,
		n_samp: int,
		amplitude=1.0,
		gaussian=True,
		mag=True,
		phase=True,
		group_delay=True,
		relative_to_input=True) -> FreqResponse:

	freqs = np.array(freqs)

	freq_resp = FreqResponse(freqs=freqs, sample_rate=sample_rate)

	if not (mag or phase):
		return freq_resp

	if mag:
		freq_resp.mag = np.zeros(len(freqs))

	if phase:
		freq_resp.phase = np.zeros(len(freqs))

	x = signal_generation.gen_noise(n_samp, gaussian=gaussian, amp=amplitude)

	y = system.process_vector(x)

	for n, freq in enumerate(freqs):

		kwargs = dict(
			freq=freq, sample_rate=sample_rate,
			mag=mag, phase=phase,
			adjust_num_samp=True, normalize=(not relative_to_input))
		x_ret = single_freq_dft(x, **kwargs)
		y_ret = single_freq_dft(y, **kwargs)

		if mag and phase:
			x_mag, x_phase = x_ret
			y_mag, y_phase = y_ret

		elif mag:
			x_mag = x_ret
			y_mag = y_ret

		else:
			assert phase
			x_phase = x_ret
			y_phase = y_ret

		if mag:
			freq_resp.mag[n] = (y_mag / x_mag) if relative_to_input else y_mag

		if phase:
			freq_resp.phase[n] = y_phase - x_phase

	if group_delay:
		freq_resp.group_delay = phase_to_group_delay(freqs, freq_resp.phase, sample_rate)

	return freq_resp


def _test_sine_vs_noise(long: bool):
	from filters import one_pole
	from filters import biquad
	from overdrive import overdrive
	from processor import GainWrapper, CascadedProcessors, GainProcessor
	from utils.utils import to_dB

	sample_rate = 96000
	cutoff = 1000
	Q = 2.0
	wc = cutoff / sample_rate

	if long:
		n_samp_min = 4096
		n_samp_noise = 4 * sample_rate
		eps_dB = 3

		freqs = [
			10,
			30,
			100,
			300,
			1000,
			3000,
			10000,
			20000,
		]

		phase_eps = 0.1
		delay_eps = 0.1

	else:
		n_samp_min = 1024
		n_samp_noise = sample_rate
		eps_dB = 6

		freqs = [
			10,
			100,
			1000,
			10000,
			20000,
		]

		phase_eps = 0.1
		delay_eps = 0.1

	processors = [
		("pass-through processor", CascadedProcessors([])),
		("Basic one pole", one_pole.BasicOnePole(wc=wc)),
		("Trapz one pole", one_pole.TrapzOnePole(wc=wc)),
		("Basic one pole highpass", one_pole.BasicOnePoleHighpass(wc=wc)),
		("Biquad, Q=%g" % Q, biquad.BiquadLowpass(wc=wc, Q=Q)),
		("tanh overdrive", overdrive.TanhProcessor()),
		("tanh overdrive, 20 dB gain", GainWrapper(overdrive.TanhProcessor(), 10.)),
		("tanh overdrive, -20 dB gain", GainWrapper(overdrive.TanhProcessor(), 0.1)),
		("Squarizer", overdrive.Squarizer()),
		("Squarizer -20 dB", CascadedProcessors([overdrive.Squarizer(), GainProcessor(0.1)])),
		("One pole then tanh", CascadedProcessors([one_pole.BasicOnePole(wc=wc), overdrive.TanhProcessor(gain=2)])),
		("tanh then one pole", CascadedProcessors([overdrive.TanhProcessor(gain=2), one_pole.BasicOnePole(wc=wc)])),
		("Biquad, Q=%g, then hard clip at 1.1" % Q, CascadedProcessors([biquad.BiquadLowpass(wc=wc, Q=Q), overdrive.Clipper(gain=1.0/1.1)])),
		("Biquad, Q=%g, then hard clip at 1" % Q, CascadedProcessors([biquad.BiquadLowpass(wc=wc, Q=Q), overdrive.Clipper()])),
		("Rossum 92 Nonlinear Biquad, Q=%g, gain 10" % Q, GainWrapper(biquad.Rossum92Biquad(wc=wc, Q=Q), 10.)),
	]

	for name, processor in processors:
		sine_resp = get_discrete_sine_sweep_freq_response(
			processor, freqs, sample_rate=sample_rate, rms=False, thdn=False, n_samp_min=n_samp_min)
		noise_resp = get_white_noise_response(processor, freqs=freqs, sample_rate=sample_rate, n_samp=n_samp_noise)

		assert np.array_equal(sine_resp.freqs, freqs)
		assert np.array_equal(noise_resp.freqs, freqs)

		if False:
			print(sine_resp.mag)
			print(to_dB(sine_resp.mag))
			print(noise_resp.mag)
			print(to_dB(noise_resp.mag))

			print(sine_resp.phase)
			print(to_dB(sine_resp.phase))
			print(noise_resp.phase)
			print(to_dB(noise_resp.phase))

			print(sine_resp.group_delay)
			print(to_dB(sine_resp.group_delay))
			print(noise_resp.group_delay)
			print(to_dB(noise_resp.group_delay))

		unit_test.test_approx_equal(to_dB(sine_resp.mag), to_dB(noise_resp.mag), eps_abs=eps_dB)
		unit_test.test_approx_equal(sine_resp.phase, noise_resp.phase, eps_abs=phase_eps)
		unit_test.test_approx_equal(sine_resp.group_delay, noise_resp.group_delay, eps_abs=delay_eps)


_unit_tests_short.append(lambda: _test_sine_vs_noise(False))
_unit_tests_full.append(lambda: _test_sine_vs_noise(True))


def check_linear_and_get_freq_resp(
		system: ProcessorBase,
		freqs: Iterable,
		sample_rate,
		n_samp: Optional[int]=None,
		n_cycles=40.0,
		n_samp_min: Optional[int]=None,
		amplitude=1.0,
		eps=0.00001,
		mag=True,
		rms=True,
		phase=True,
		group_delay=True) -> Tuple[bool, FreqResponse]:
	"""
	Check if system is linear and calculate frequency response
	If linear, impulse response will be used
	If nonlinear, sine sweep will be used

	Linearity check is done by testing if impulse response is equal to derivative of step response

	:param system: Processor to process
	:param freqs: frequencies to get response at. More frequencies will also lead to more precise group delay
	:param sample_rate: sample rate, in Hz
	:param n_cycles: how many cycles of waveform to calculate over
		(if using impulse response, IR length will be based on lowest of freqs)
	:param n_samp_min: if using n_cycles, minimum n_samp
	:param n_samp: how many samples to calculate over - overrides n_cycles

	:param amplitude: amplitude of IR/step/sine wave
	:param eps: epsilon value for IR/step comparison

	:param mag: if False, does not calculate nor return magnitude
	:param rms: if False, does not calculate nor return RMS magnitude
	:param phase: if False, does not calculate nor return phase
	:param group_delay: if False, does not calculate nor return group delay

	:return: Tuple (True if linear, frequency response of system)
	"""

	if n_samp is None:
		lowest_freq = min(freqs)
		highest_period = sample_rate / lowest_freq
		n_samp = highest_period * n_cycles

	linear = linearity.check_linear(system, n_samp=n_samp, amplitude=amplitude, eps=eps)

	if linear:
		freq_resp = get_ir_freq_response(
			ir, freqs, sample_rate,
			mag=mag, phase=phase, group_delay=group_delay)

	else:
		freq_resp = get_discrete_sine_sweep_freq_response(
			system, freqs, sample_rate,
			n_cycles=n_cycles, n_samp=n_samp, n_samp_min=n_samp_min,
			amplitude=amplitude,
			mag=mag, rms=rms, phase=phase, group_delay=group_delay)

	return linear, freq_resp


def _test_dft_trivial():
	"""
	Test DFT using the same cos/sin signal as the DFT uses
	"""

	sample_rates = [32000., 44100., 96000., 192000.]
	freqs = [
		100., 100.125, 107., 440., 500.,
		1000., 1001., 2050., 3000., 3010., 5000.,
		10000., 20000.,
		1000 * PI,
	]

	for sample_rate in sample_rates:
		for freq in freqs:

			if (2 * freq) > sample_rate:
				continue

			f_norm = freq / sample_rate
			period = sample_rate / freq

			max_num_samples = int(math.ceil(max(period, sample_rate)))
			n_samp = dft_num_samples(
				freq, sample_rate,
				max_num_samples=max_num_samples)

			n_cycles = n_samp / period

			cycle_err = abs(n_cycles - round(n_cycles))

			# TODO: tighten up maximum clip values here
			eps_rel = utils.clip(cycle_err, (1e-12, 1e-5))
			eps_zero = utils.clip(10*cycle_err, (1e-11, 1e-3))

			x_cos, x_sin = signal_generation.gen_cos_sine(f_norm, n_samp)

			dft_cos = _single_freq_dft(
				x_cos, cos_sig=x_cos, sin_sig=x_sin, freq=freq, sample_rate=sample_rate, adjust_num_samp=False)
			dft_sin = _single_freq_dft(
				x_sin, cos_sig=x_cos, sin_sig=x_sin, freq=freq, sample_rate=sample_rate, adjust_num_samp=False)

			unit_test.test_approx_equal(np.real(dft_cos), 0.5*n_samp, eps=eps_rel, rel=True)
			unit_test.test_approx_equal(np.imag(dft_cos), 0., eps=eps_zero)

			unit_test.test_approx_equal(np.real(dft_sin), 0., eps=eps_zero)
			unit_test.test_approx_equal(np.imag(dft_sin), -0.5*n_samp, eps=eps_rel, rel=True)


_unit_tests_short.append(_test_dft_trivial)
_unit_tests_full.append(_test_dft_trivial)


def _test_dft_against_fft(long=True):
	"""
	Test single_freq_dft against numpy fft
	Note that this is only possible at frequencies that perfectly fit into n samples
	"""

	eps = 1e-6

	mags_short = [1.0, eps, PI, 100.1]
	mags_full = [1.0, 0.125, 0.1, eps, PI, 100., 100.1, 100 + eps]

	phases_short = [0, 0.1, 0.125, 0.24, 0.25, 0.9]
	phases_full = [0, eps, 0.1, 0.124, 0.125, 0.126, 0.25, 0.24, 0.26, 0.25 - eps, 0.25 + eps, 0.5, 0.75, 0.9, 1 - eps]

	tests = [
		dict(
			n_samples=512,
			bin_nums=[0, 1, 10, 11, 100, 255, 256],
			eps_abs=1e-9,
			eps_rel=1e-12,
			mags=mags_full,
			phases=phases_full,
		),
		dict(
			n_samples=4096,
			bin_nums=[0, 1, 10, 11, 100, 512, 1024, 2047, 2048],
			eps_abs=1e-9,
			eps_rel=1e-9,
			mags=mags_full,
			phases=phases_full,
		),
		dict(
			n_samples=65536,
			bin_nums=(
				[0, 1, 10, 11, 100, 512, 1024, 2047, 2048, 4096, 8191, 8192, 32767, 32768] if long else
				[0, 1, 10, 11, 100, 2048, 4096, 32767, 32768]
			),
			eps_abs=1e-9,
			eps_rel=1e-8,
			mags=(mags_full if long else mags_short),
			phases=(phases_full if long else phases_short),
		),
	]

	for test in tests:

		eps_abs = test['eps_abs']
		eps_rel = test['eps_rel']
		n_samples = test['n_samples']
		bin_nums = test['bin_nums']
		mags = test['mags']
		phases = test['phases']

		for bin_num in bin_nums:
			for mag in mags:
				for phase in phases:
					unit_test.log('n_samples %u, bin_num %u, mag %g, ph %g' % (n_samples, bin_num, mag, phase))

					f_norm = bin_num / n_samples

					sig = mag * signal_generation.gen_sine(f_norm, n_samp=n_samples, start_phase=phase)

					dft_sig = single_freq_dft(
						sig, f_norm, sample_rate=1.0,
						mag=False, phase=False,
						adjust_num_samp=False, normalize=False)

					fft_sig = np.fft.fft(sig)
					fft_at_bin = fft_sig[bin_num]

					unit_test.test_approx_equal(
						np.real(dft_sig), np.real(fft_at_bin),
						abs_rel=True, eps_abs=eps_abs, eps_rel=eps_rel)

					unit_test.test_approx_equal(
						np.imag(dft_sig), np.imag(fft_at_bin),
						abs_rel=True, eps_abs=eps_abs, eps_rel=eps_rel)

					unit_test.test_approx_equal(
						np.abs(dft_sig), np.abs(fft_at_bin),
						abs_rel=True, eps_abs=eps_abs, eps_rel=eps_rel
					)


_unit_tests_short.append(lambda: _test_dft_against_fft(long=False))
_unit_tests_full.append(lambda: _test_dft_against_fft(long=True))


def _test_dft_sine(long=True):
	"""
	Test single_freq_dft with sine waves at arbitrary frequency, phase, amplitude
	"""

	# Similar to both _test_dft_trivial and _test_dft_against_fft but covers some ground those don't
	# (arbitrary frequency)

	eps = 1e-6

	sample_rate = 96000.

	mags_short = [1.0, eps, PI, 100.1]
	mags_full = [1.0, 0.125, 0.1, eps, PI, 100., 100.1, 100 + eps]

	phases_short = [0, 0.1, 0.125, 0.24, 0.25, 0.9]
	phases_full = [0, eps, 0.1, 0.124, 0.125, 0.126, 0.25, 0.24, 0.26, 0.25 - eps, 0.25 + eps, 0.5, 0.75, 0.9, 1 - eps]




	simple_freqs = [
		20., 100., 1000., 10000., 20000., 32000.,
	]

	complex_freqs = [
		440., 440. + 0.1*PI, 1234., 5927., PI*10000.,
	]



	freqs = [
		20., 100., 440., 440. + 0.1*PI, 1000., 1234., 5927., 10000., 20000., PI*10000., 32000.,
	]

	for freq in freqs:

		#eps_abs = test['eps_abs']
		#eps_rel = test['eps_rel']
		#n_samples = test['n_samples']
		#bin_nums = test['bin_nums']
		#mags = test['mags']
		#phases = test['phases']

		pass  # TODO: like _test_dft_trivial() but with with non-trivial phase & magnitude


#_unit_tests_short.append(lambda: _test_dft_sine(long=False))  # TODO: enable when ready
#_unit_tests_full.append(lambda: _test_dft_sine(long=True))  # TODO: enable when ready


def _do_detail():
	from matplotlib import pyplot as plt

	sample_rate = 96000
	n_samp = None
	n_samp_min = 4096
	n_cycles = 128.0
	n_samp_plot = 128

	freqs = [100., 107., 500., 1000., 2050., 3000., 3010., 5000., 10000., 20000.]

	fig = plt.figure()

	print('%6s   %6s   %8s   %8s   %10s   %10s   %10s   %10s   %12s' % (
		'freq', 'phase',
		'num samp', 'num cyc',
		'real err', 'imag err',
		'mag err', 'ph err', 'max rec err',))

	for n, freq in enumerate(freqs):

		f_norm = freq / sample_rate
		period = sample_rate / freq

		if n_samp is None:
			max_num_samples = int(math.ceil(max(n_cycles * period, sample_rate)))
			n_samp_this_freq = dft_num_samples(
				freq, sample_rate,
				min_num_samples=n_samp_min,
				max_num_samples=max_num_samples)
		else:
			n_samp_this_freq = n_samp

		n_cycles_this_freq = n_samp_this_freq * f_norm

		x_cos, x_sin = signal_generation.gen_cos_sine(f_norm, n_samp_this_freq)
		#mag_sin, phase_sin = _single_freq_dft(x_sin, x_cos, x_sin, freq, sample_rate, mag=True, phase=True)

		dft_cos = _single_freq_dft(
			x_cos, cos_sig=x_cos, sin_sig=x_sin, freq=freq, sample_rate=sample_rate, adjust_num_samp=False)
		dft_sin = _single_freq_dft(
			x_sin, cos_sig=x_cos, sin_sig=x_sin, freq=freq, sample_rate=sample_rate, adjust_num_samp=False)

		dft_cos /= (0.5*n_samp_this_freq)
		dft_sin /= (0.5*n_samp_this_freq)

		mag_cos = np.abs(dft_cos)
		mag_sin = np.abs(dft_sin)
		phase_cos = np.angle(dft_cos)
		phase_sin = np.angle(dft_sin)

		# gen_sine takes phase 0-1, relative to sine (not cos)
		phase_cos_01 = np.mod((phase_cos / TWOPI) + 0.25, 1.0)
		phase_sin_01 = np.mod((phase_sin / TWOPI) + 0.25, 1.0)

		idx = np.arange(n_samp_plot)

		reconstructed_cos = signal_generation.gen_sine(f_norm, n_samp=n_samp_this_freq, start_phase=phase_cos_01) * mag_cos
		reconstructed_sin = signal_generation.gen_sine(f_norm, n_samp=n_samp_this_freq, start_phase=phase_sin_01) * mag_sin

		cos_real_err = np.real(dft_cos) - 1.0
		cos_imag_err = np.imag(dft_cos)

		sin_real_err = np.real(dft_sin)
		sin_imag_err = np.imag(dft_sin) + 1.0

		cos_mag_err = mag_cos - 1
		sin_mag_err = mag_sin - 1

		cos_phase_err = phase_cos
		sin_phase_err = phase_sin + HALF_PI

		cos_rec_err = reconstructed_cos - x_cos
		sin_rec_err = reconstructed_sin - x_sin

		plt.subplot(len(freqs), 2, 2*n+1)

		if n == 0:
			plt.title('cos')

		plt.plot(idx, x_cos[:n_samp_plot], label='input')
		plt.plot(idx, x_cos[:n_samp_plot], label='output')
		plt.plot(idx, reconstructed_cos[:n_samp_plot], label='reconstructed')
		plt.plot(idx, cos_rec_err[:n_samp_plot], label='reconst err' % np.amax(np.abs(cos_rec_err)))
		plt.grid()
		plt.legend()
		plt.ylabel('%g Hz' % freq)

		plt.subplot(len(freqs), 2, 2*n+2)

		if n == 0:
			plt.title('sin')

		plt.plot(idx, x_sin[:n_samp_plot], label='input')
		plt.plot(idx, x_sin[:n_samp_plot], label='output')
		plt.plot(idx, reconstructed_sin[:n_samp_plot], label='reconstructed')
		plt.plot(idx, sin_rec_err[:n_samp_plot], label='reconst err' % np.amax(np.abs(sin_rec_err)))
		plt.grid()
		plt.legend()

		print()

		print('%6g   %6s   %8g   %8g   %10.2e   %10.2e   %10.2e   %10.2e   %12.2e' % (
			freq, 'cos',
			n_samp_this_freq, n_cycles_this_freq,
			cos_real_err, cos_imag_err,
			cos_mag_err, cos_phase_err, np.amax(np.abs(cos_rec_err))))

		print('%6g   %6s   %8g   %8g   %10.2e   %10.2e   %10.2e   %10.2e   %12.2e' % (
			freq, 'sin',
			n_samp_this_freq, n_cycles_this_freq,
			sin_real_err, sin_imag_err,
			sin_mag_err, sin_phase_err, np.amax(np.abs(sin_rec_err))))

	plt.show()


def _do_main(trivial=True, linear=True, nonlin=True, do_noise=True):
	from matplotlib import pyplot as plt
	from filters import one_pole
	from filters import biquad
	from overdrive import overdrive
	from processor import GainWrapper, CascadedProcessors, GainProcessor
	from utils.utils import to_dB

	sample_rate = 96000
	cutoff = 1000
	Q = 2.0

	n_samp_ir = 16384
	n_samp_min = 4096
	n_samp_noise = 4 * sample_rate

	wc = cutoff / sample_rate

	filters_trivial = [
		("pass-through processor", CascadedProcessors([]), None),
	]

	filters_linear = [
		("Basic one pole", one_pole.BasicOnePole(wc=wc), None),
		("Trapz one pole", one_pole.TrapzOnePole(wc=wc), None),
		("Basic one pole highpass", one_pole.BasicOnePoleHighpass(wc=wc), None),
		("Biquad, Q=%g" % Q, biquad.BiquadLowpass(wc=wc, Q=Q), None),
	]

	filters_nonlin = [
		("tanh overdrive", overdrive.TanhProcessor(), None),
		("tanh overdrive, 20 dB gain", GainWrapper(overdrive.TanhProcessor(), 10.), None),
		("tanh overdrive, -20 dB gain", GainWrapper(overdrive.TanhProcessor(), 0.1), None),
		("Squarizer", overdrive.Squarizer(), SQUARE_THDN),
		("Squarizer -20 dB", CascadedProcessors([overdrive.Squarizer(), GainProcessor(0.1)]), SQUARE_THDN),
		("One pole then tanh", CascadedProcessors([one_pole.BasicOnePole(wc=wc), overdrive.TanhProcessor(gain=2)]), None),
		("tanh then one pole", CascadedProcessors([overdrive.TanhProcessor(gain=2), one_pole.BasicOnePole(wc=wc)]), None),
		("Biquad, Q=%g, then hard clip at 1.1" % Q, CascadedProcessors([biquad.BiquadLowpass(wc=wc, Q=Q), overdrive.Clipper(gain=1.0/1.1)]), None),
		("Biquad, Q=%g, then hard clip at 1" % Q, CascadedProcessors([biquad.BiquadLowpass(wc=wc, Q=Q), overdrive.Clipper()]), None),
		("Rossum 92 Nonlinear Biquad, Q=%g, gain 10" % Q, GainWrapper(biquad.Rossum92Biquad(wc=wc, Q=Q), 10.), None),
	]

	filters = []

	if trivial:
		filters += filters_trivial
	if linear:
		filters += filters_linear
	if nonlin:
		filters += filters_nonlin

	freqs = np.array([
		10., 20., 30., 50.,
		100., 200., 300., 500., 700., 800., 900., 950.,
		1000., 1050., 1100., 1200., 1300., 1500., 2000., 3000., 5000.,
		10000., 11000., 13000., 15000., 20000., 25000., 30000., 40000.])

	for filter_name, filter, expected_thdn_dB in filters:

		print('Processing filter "%s"' % filter_name)

		ir = time_domain_response.get_impulse_response(filter, n_samp_ir, amplitude=1.0, reset=True)

		ir_freq_resp = get_ir_freq_response(
			ir, freqs, sample_rate,
			mag=True, phase=True, group_delay=True)

		sweep_freq_resp = get_discrete_sine_sweep_freq_response(
			filter, freqs, sample_rate,
			n_cycles=128.0, n_samp=None, n_samp_min=n_samp_min,
			amplitude=1.0,
			mag=True, rms=True, phase=True, group_delay=True)

		mag_rms_err = np.abs(sweep_freq_resp.mag - sweep_freq_resp.rms)
		sweep_ir_err = np.abs(sweep_freq_resp.mag - ir_freq_resp.mag)

		if do_noise:
			uniform_noise_freq_resp = get_white_noise_response(
				filter, freqs, sample_rate, n_samp_noise,
				gaussian=False,
				mag=True, phase=True, group_delay=True)

			gaussian_noise_freq_resp = get_white_noise_response(
				filter, freqs, sample_rate, n_samp_noise,
				gaussian=True,
				mag=True, phase=True, group_delay=True)

		fft_bin_freqs = np.fft.fftfreq(n_samp_ir, 1.0 / sample_rate)[:n_samp_ir // 2]
		ir_fft = np.fft.fft(ir)[:n_samp_ir // 2]
		ir_fft_group_delay = phase_to_group_delay(fft_bin_freqs, np.angle(ir_fft), sample_rate)

		#
		# Plotting
		#

		fig = plt.figure()
		fig.suptitle(filter_name)

		plt.subplot(4, 1, 1)
		plt.semilogx(fft_bin_freqs, to_dB(np.abs(ir_fft)), label='IR magnitude from FFT')
		plt.semilogx(freqs, to_dB(ir_freq_resp.mag), label='IR magnitude')
		plt.semilogx(freqs, to_dB(sweep_freq_resp.mag), label='Sine sweep magnitude')
		plt.semilogx(freqs, to_dB(sweep_freq_resp.rms), label='Sine sweep RMS')
		if do_noise:
			plt.semilogx(freqs, to_dB(uniform_noise_freq_resp.mag), label='Uniform noise')
			plt.semilogx(freqs, to_dB(gaussian_noise_freq_resp.mag), label='Gaussian noise')
		plt.grid()
		plt.legend()
		plt.ylabel('dB')

		plt.subplot(4, 1, 2)
		if expected_thdn_dB is not None:
			plt.axhline(expected_thdn_dB, color='g', label='Expected THD + Noise')
		plt.semilogx(freqs, to_dB(sweep_freq_resp.thdn, min_dB=-120), label='Measured THD + Noise')
		plt.semilogx(freqs, to_dB(sweep_ir_err, min_dB=-120), label='sine vs IR error')
		plt.semilogx(freqs, to_dB(mag_rms_err, min_dB=-120), label='sine mag vs RMS error')
		plt.grid()
		plt.legend()
		plt.ylabel('dB')

		plt.subplot(4, 1, 3)
		plt.semilogx(fft_bin_freqs, np.rad2deg(np.angle(ir_fft)), label='IR phase from FFT')
		plt.semilogx(freqs, np.rad2deg(ir_freq_resp.phase), label='IR phase')
		plt.semilogx(freqs, np.rad2deg(sweep_freq_resp.phase), label='Sine sweep phase')
		if do_noise:
			plt.semilogx(freqs, np.rad2deg(uniform_noise_freq_resp.phase), label='Uniform noise')
			plt.semilogx(freqs, np.rad2deg(gaussian_noise_freq_resp.phase), label='Gaussian noise')
		plt.grid()
		plt.legend()
		plt.ylabel('Degrees')

		plt.subplot(4, 1, 4)
		plt.semilogx(fft_bin_freqs, ir_fft_group_delay, label='IR group delay from FFT')
		plt.semilogx(freqs, ir_freq_resp.group_delay, label='IR group delay')
		plt.semilogx(freqs, sweep_freq_resp.group_delay, label='Sine sweep group delay')
		if do_noise:
			plt.semilogx(freqs, uniform_noise_freq_resp.group_delay, label='Uniform noise')
			plt.semilogx(freqs, gaussian_noise_freq_resp.group_delay, label='Gaussian noise')
		plt.grid()
		plt.legend()
		plt.ylabel('Seconds')

	print('Showing plots')
	plt.show()


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)

	grp = parser.add_argument_group('Standard analysis')
	grp.add_argument('--noise', action='store_true', help='Include noise analysis (slow)')
	grp.add_argument('--trivial', action='store_true', help='Process only trivial processors')
	grp.add_argument('--linear', action='store_true', help='Process only linear filters')
	grp.add_argument('--nonlin', action='store_true', help='Process only nonlinear processors')

	grp = parser.add_argument_group('Other functions')
	grp.add_argument('--detail', action='store_true')

	return parser


def main(args):

	if args.test:
		return test(verbose=args.verbose, long=args.long)

	if args.detail:
		return _do_detail()

	num_args = sum([args.trivial, args.linear, args.nonlin, args.detail])

	if num_args > 1:
		raise ValueError('Can only give max 1 of --trivial, --nonlin, --detail')
	elif not num_args:
		args.trivial = True
		args.linear = True
		args.nonlin = True

	trivial = (args.trivial or args.linear)
	linear = args.linear
	nonlin = args.nonlin
	return _do_main(trivial=trivial, linear=linear, nonlin=nonlin, do_noise=args.noise)


def test(verbose=False, long=False):
	if long:
		return unit_test.run_unit_tests(_unit_tests_full, verbose=verbose)
	else:
		return unit_test.run_unit_tests(_unit_tests_short, verbose=verbose)
