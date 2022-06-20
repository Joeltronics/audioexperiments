#!/usr/bin/env python3

import numpy as np
from typing import Tuple, Union, Optional


def sample_time_index(n_samp: int, sample_rate: float) -> np.ndarray:
	return np.linspace(0., n_samp / sample_rate, n_samp)


def phase_to_sine(phase):
	"""
	Convert phase to sine signal

	:param phase: phase (expect np.ndarray, but float will work too), 0-1
	:return: sine signal
	"""
	return np.sin(phase * 2.0 * np.pi)


def phase_to_cos(phase):
	"""
	Convert phase to cosine signal

	:param phase: phase (expect np.ndarray, but float will work too), 0-1
	:return: cosine signal
	"""
	return phase_to_sine((phase + 0.25) % 1.0)


def gen_phase(freq_norm: Union[float, np.ndarray], n_samp: Optional[int]=None, start_phase=0.0, allow_aliasing=False) -> np.ndarray:
	"""
	Generate phase signal

	:param freq_norm: normalized frequency, i.e. freq / sample rate. Either a scalar or an nd.array of length n_samp
	:param n_samp:
	:param start_phase: 0-1
	:param allow_aliasing: if False, will raise ValueError if freq_norm is out of range[0.0, 0.5]
	:return:
	"""

	if not allow_aliasing and (np.amin(freq_norm) < 0.0 or np.amax(freq_norm) > 0.5):
		if np.isscalar(freq_norm):
			raise ValueError("freq out of range: %f" % freq_norm)
		else:
			raise ValueError("freq out of range: %f-%f" % (np.amin(freq_norm), np.amax(freq_norm)))

	if np.isscalar(freq_norm):
		if n_samp is None:
			raise ValueError("Must give n_samp if using scalar frequency")

		ph = np.arange(n_samp, dtype=np.float64) * freq_norm
	else:
		if n_samp and len(freq_norm) != n_samp:
			raise ValueError("n_samp given different size from len(freq_norm)")

		ph = np.cumsum(freq_norm)

	# TODO: could make this slightly more efficient by starting at correct phase instead of adding (not worth the work for now)
	ph += (start_phase - ph[0])
	ph = np.mod(ph, 1.0)

	return ph


def gen_sine(freq_norm: float, n_samp: int, start_phase=0.0, allow_aliasing=False) -> np.ndarray:
	"""
	Generate sine wave

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:param start_phase: 0-1
	:param allow_aliasing: if False, will raise ValueError if freq_norm is out of range[0.0, 0.5]
	:return:
	"""
	return phase_to_sine(gen_phase(freq_norm, n_samp, start_phase=start_phase, allow_aliasing=allow_aliasing))


def gen_cos_sine(freq_norm: float, n_samp: int) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Generate cosine and sine waves at same frequency

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:return: cosine wave, sine wave
	"""
	phase = gen_phase(freq_norm, n_samp)
	return phase_to_cos(phase), phase_to_sine(phase)


def gen_saw(freq_norm: float, n_samp: int, start_phase=0.0) -> np.ndarray:
	"""
	Generate naive sawtooth wave (no anti-aliasing)

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:return:
	"""
	return gen_phase(freq_norm, n_samp, start_phase=start_phase) * 2.0 - 1.0


def gen_square(freq_norm: float, n_samp: int) -> np.ndarray:
	"""
	Generate naive square wave (no anti-aliasing)

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:return:
	"""

	p = gen_phase(freq_norm, n_samp)
	return ((p >= 0.5) * 2 - 1).astype(float)


def gen_noise(n_samp: int, gaussian=False, amp=1.0) -> np.ndarray:
	"""Generate white or Gaussian noise"""
	if gaussian:
		return np.random.randn(n_samp) * amp
	else:
		return np.random.rand(n_samp) * amp


def gen_freq_sweep_phase(start_freq_norm: float, end_freq_norm: float, n_samp: int, log=True, start_phase=0.0) -> np.ndarray:
	if log:
		f = np.logspace(np.log2(start_freq_norm), np.log2(end_freq_norm), n_samp, base=2)
	else:
		f = np.linspace(start_freq_norm, end_freq_norm, n_samp)

	return gen_phase(f, start_phase=start_phase)


def gen_freq_sweep_sine(start_freq_norm: float, end_freq_norm: float, n_samp: int, log=True, start_phase=0.0) -> np.ndarray:
	ph = gen_freq_sweep_phase(start_freq_norm, end_freq_norm, n_samp, log=log, start_phase=start_phase)
	return phase_to_sine(ph)


def plot(args):
	from matplotlib import pyplot as plt
	import math

	freq = 110.
	sample_rate = 48000.
	freq_norm = freq / sample_rate

	# Period is around 436 samples
	n_samp = 1024

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

	plt.figure()

	sample_rate = 48000.
	n_samp = 8192
	start_freq = 20. / sample_rate
	end_freq = 20000. / sample_rate
	sweep_log = gen_freq_sweep_sine(start_freq, end_freq, n_samp, log=True)
	sweep_lin = gen_freq_sweep_sine(start_freq, end_freq, n_samp, log=False)

	t = sample_time_index(n_samp, sample_rate)

	plt.subplot(211)

	plt.plot(t, sweep_log)

	plt.grid()
	plt.title('Frequency sweeps')
	plt.ylabel('Log sweep')

	plt.subplot(212)

	plt.plot(t, sweep_lin)

	plt.grid()
	plt.ylabel('Lin sweep')
	plt.xlabel('Time (seconds)')

	plt.show()


def main(args):
	plot(args)
