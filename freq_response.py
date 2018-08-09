#!/usr/bin/env python3

import numpy as np
import math
import utils

from processor import Processor
from typing import Union, Tuple, Any, Iterable
import signal_generation


def _single_freq_dft(
		x: np.ndarray,
		cos_sig: np.ndarray,
		sin_sig: np.ndarray,
		mag=False,
		phase=False):
	# TODO: use Goertzel algo instead

	dft_mult = cos_sig - 1j * sin_sig

	xs = x * dft_mult
	xs = 2.0 * np.mean(xs)

	if mag and phase:
		return np.abs(xs), np.angle(xs)
	elif mag:
		return np.abs(xs)
	elif phase:
		return np.angle(xs)
	else:
		return xs


def single_freq_dft(x: np.ndarray, freq: float, return_mag_phase=True):
	cos_sig, sin_sig = signal_generation.gen_cos_sine(freq, len(x))
	return _single_freq_dft(x, cos_sig, sin_sig, mag=return_mag_phase, phase=return_mag_phase)


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


def get_freq_response(
		system: Processor,
		freqs: Iterable,
		sample_rate,
		n_cycles=40.0,
		n_samp=None,
		amplitude=1.0,
		mag=True,
		rms=False,
		phase=True,
		group_delay=False):
	"""

	:param system: Processor to process
	:param freqs: frequencies to get response at
	:param sample_rate: sample rate, in Hz
	:param n_cycles: how many cycles of waveform to calculate over
	:param n_samp: how many samples to calculate over - overrides n_cycles
	:param amplitude: amplitude of sine wave to pass in
	:param throw_if_nonlinear:

	:param mag: if magnitude at frequency should be returned
	:param rms: if RMS amplitude (of whole signal) should be returned
	:param phase: if phase should be returned (in radians)
	:param group_delay: if group delay should be calculated & returned

	:return: mag, rms, phase, group_delay, depending on which values are set
	"""

	if mag:
		mags = np.zeros(len(freqs))

	if rms:
		rmses = np.zeros(len(freqs))

	if phase:
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

		if mag or phase:
			# TODO: could be a bit more efficient by not calculating phase if not necessary
			# Code might get messy though
			this_mag, ph = _single_freq_dft(y, x_cos, x_sin, mag=(mag or phase), phase=(mag or phase))

			if mag:
				mags[n] = this_mag

			if phase:
				# Add 90 degrees because we used the sine as input
				ph += 0.5 * np.pi
				phases[n] = ph

		if rms:
			rmses[n] = utils.rms(y)

	rets = []

	if mag:
		rets.append(mags)

	if rms:
		rets.append(rmses)

	if phase:
		rets.append(phases)

	if group_delay:
		group_delay = phase_to_group_delay(freqs, phases, sample_rate)
		rets.append(group_delay)

	return tuple(rets)
