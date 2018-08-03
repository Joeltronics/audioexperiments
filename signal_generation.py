#!/usr/bin/env python3

import numpy as np
from typing import Tuple


def sample_time_index(n_samp: int, sample_rate: float) -> np.ndarray:
	return np.linspace(0., n_samp / sample_rate, n_samp)


def phase_to_sine(phase):
	"""Convert phase to sine signal

	:param phase: phase (expect np.ndarray, but float will work too), 0-1
	:return: cosine signal
	"""
	return np.sin(phase * 2.0 * np.pi)


def phase_to_cos(phase):
	"""Convert phase to cosine signal

	:param phase: phase (expect np.ndarray, but float will work too), 0-1
	:return: cosine signal
	"""
	return phase_to_sine((phase + 0.25) % 1.0)


def gen_phase(freq_norm: float, n_samp: int, start_phase=0.0, allow_aliasing=False) -> np.ndarray:
	"""

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp:
	:param start_phase:
	:param allow_aliasing: if False, will raise ValueError if freq_norm is out of range[0.0, 0.5]
	:return:
	"""

	if not allow_aliasing and (freq_norm <= 0.0 or freq_norm >= 0.5):
		raise ValueError("freq out of range: %f" % freq_norm)

	# TODO: vectorize this

	ph = np.zeros(n_samp)
	phase = start_phase

	for n in range(n_samp):
		ph[n] = phase
		phase += freq_norm

	ph = np.mod(ph, 1.0)

	return ph


def gen_sine(freq_norm: float, n_samp: int, start_phase=0.0, allow_aliasing=False) -> np.ndarray:
	"""Generate sine wave

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:param start_phase:
	:param allow_aliasing:
	:return:
	"""
	phase = gen_phase(freq_norm, n_samp, start_phase=start_phase, allow_aliasing=allow_aliasing)
	return phase_to_sine(phase)


def gen_cos_sine(freq_norm: float, n_samp: int) -> Tuple[np.ndarray, np.ndarray]:
	"""

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:return:
	"""
	ph = gen_phase(freq_norm, n_samp)
	return phase_to_cos(ph), phase_to_sine(ph)


def gen_saw(freq_norm: float, n_samp: int) -> np.ndarray:
	"""Generate naive sawtooth wave (no anti-aliasing)

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:return:
	"""
	return gen_phase(freq_norm, n_samp) * 2.0 - 1.0


def gen_square(freq_norm: float, n_samp: int) -> np.ndarray:
	"""Generate naive square wave (no anti-aliasing)

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:return:
	"""

	p = gen_phase(freq_norm, n_samp)
	return ((p >= 0.5) * 2 - 1).astype(float)


if __name__ == "__main__":
	from matplotlib import pyplot as plt
	import math

	freq = 440.
	sample_rate = 48000.
	freq_norm = freq / sample_rate

	# Period is around 110 samples
	n_samp = 256

	t = sample_time_index(n_samp, sample_rate)

	ph = gen_phase(freq_norm, n_samp=n_samp)
	c, s = gen_cos_sine(freq_norm, n_samp=n_samp)
	saw = gen_saw(freq_norm, n_samp=n_samp)
	squ = gen_square(freq_norm, n_samp=n_samp)

	plt.figure()

	plt.subplot(221)

	plt.plot(ph, label='phase')

	plt.title('signal_generation, 440 Hz @ 48 kHz')
	plt.ylabel('Phase')
	plt.grid()

	plt.subplot(223)
	plt.plot(t, s, label='sine')
	plt.plot(t, c, label='cosine')
	plt.plot(t, saw, label='sawtooth')
	plt.plot(t, squ, label='square')

	plt.grid()
	plt.legend()
	plt.ylabel('Signal')
	plt.xlabel('Time (seconds)')

	plt.subplot(122)

	n = math.floor(1.0 / freq_norm)

	plt.plot(ph[:n], s[:n], label='sine')
	plt.plot(ph[:n], c[:n], label='cosine')
	plt.plot(ph[:n], saw[:n], label='sawtooth')
	plt.plot(ph[:n], squ[:n], label='square')

	plt.grid()
	plt.legend()
	plt.xlabel('Phase')
	plt.ylabel('Signal')

	plt.show()
