#!/usr/bin/env python3

import numpy as np
import math
from dataclasses import dataclass

from utils import utils
from processor import ProcessorBase
from typing import Iterable, Optional
from generation import signal_generation


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


def phase_to_group_delay(freqs: np.ndarray, phases_rad: np.ndarray, sample_rate: float):
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


@dataclass
class FreqResponse:
	freqs: np.ndarray
	sample_rate: float

	amplitude: Optional[float] = None  # Amplitude frequency response was performed at (relevant for nonlinear systems)

	mag: Optional[np.ndarray] = None  # Magnitude
	rms: Optional[np.ndarray] = None  # RMS response (only relevant for nonlinear system)
	phase: Optional[np.ndarray] = None  # Phase response, in radians
	group_delay: Optional[np.ndarray] = None


def get_sine_sweep_freq_response(
		system: ProcessorBase,
		freqs: Iterable,
		sample_rate,
		n_cycles=40.0,
		n_samp=None,
		amplitude=1.0,
		mag=True,
		rms=True,
		phase=True,
		group_delay=True) -> FreqResponse:
	"""
	Calculate frequency response by passing through sine waves at various frequencies

	Unlike impulse response analysis, this will work for nonlinear systems as well
	(Of course, the definition of "frequency response" is ill-defined for a nonlinear system, as nonlinear systems
	will also add harmonic content)

	:param system: Processor to process
	:param freqs: frequencies to get response at. More frequencies will also lead to more precise group delay
	:param sample_rate: sample rate, in Hz
	:param n_cycles: how many cycles of waveform to calculate over
	:param n_samp: how many samples to calculate over - overrides n_cycles
	:param amplitude: amplitude of sine wave to pass in

	:param mag: if False, does not calculate nor return magnitude
	:param rms: if False, does not calculate nor return RMS magnitude
	:param phase: if False, does not calculate nor return phase
	:param group_delay: if False, does not calculate nor return group delay

	:return:
		frequency response of system.
		mag, phase, and group delay are based on measurement of output at only that frequency.
		RMS is based on entire signal.
		So you can get a proxy for "how nonlinear" the system is by comparing difference between mag & RMS
		(if linear, output would be a sine wave, so RMS would be 1/sqrt(2) of magnitude)
	"""

	freqs = np.array(freqs)

	freq_resp = FreqResponse(freqs=freqs, sample_rate=sample_rate)

	if mag:
		freq_resp.mag = np.zeros(len(freqs))

	if rms:
		freq_resp.rms = np.zeros(len(freqs))

	if phase:
		freq_resp.phase = np.zeros(len(freqs))

	for n, freq in enumerate(freqs):
		system.reset()

		f_norm = freq / sample_rate

		if n_samp is None:
			n_samp_this_freq = math.ceil(n_cycles / f_norm)
		else:
			n_samp_this_freq = n_samp

		x_cos, x_sin = signal_generation.gen_cos_sine(f_norm, n_samp_this_freq)

		# Use sine instead of cosine, since it starts at 0 - less prone to problems from nonlinearities
		x = x_sin
		y = system.process_vector(x * amplitude) / amplitude

		if mag or phase:
			ret = _single_freq_dft(y, x_cos, x_sin, mag=mag, phase=phase)

			if mag:
				freq_resp.mag[n] = ret[0] if (mag and phase) else ret

			if phase:
				ph = ret[1] if (mag and phase) else ret

				# Add 90 degrees because we used the sine as input
				ph += 0.5 * np.pi
				freq_resp.phase[n] = ph

		if rms:
			freq_resp.rms[n] = utils.rms(y)

	if group_delay:
		freq_resp.group_delay = phase_to_group_delay(freqs, freq_resp.phase, sample_rate)

	return freq_resp
