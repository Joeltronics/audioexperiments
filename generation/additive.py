#!/usr/bin/env python3


from .signal_generation import gen_phase, phase_to_sine

import numpy as np
import math
from typing import Union, Optional


def _get_phase_n_harm(freq_norm, n_samp, max_freq):
	if np.isscalar(freq_norm):
		if n_samp is None:
			raise ValueError('n_samp must be given if freq_norm is scalar')
		lowest_freq = freq_norm
	else:
		n_samp = len(freq_norm)
		lowest_freq = np.amin(freq_norm)

	n_harm = int(math.ceil(max_freq / lowest_freq)) - 1
	p = gen_phase(freq_norm, n_samp, allow_aliasing=True)

	return p, n_harm


def gen_saw(freq_norm: Union[float, np.ndarray], n_samp: Optional[int]=None, ampl=1.0, max_freq=0.5) -> np.ndarray:
	"""Generate sawtooth wave by additive synthesis
	Guaranteed to be alias-free if freq_norm is a scalar - otherwise, virtually alias-free as long as freq_norm is not
	modulated too quickly

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:param ampl: amplitude
	:return:
	"""

	p, n_harm = _get_phase_n_harm(freq_norm, n_samp, max_freq)
	ampl *= 2.0 / math.pi
	y = np.zeros_like(p)

	for n in range(1, n_harm + 1):
		y += phase_to_sine(p * n) * ampl / n * (n*freq_norm < max_freq)
	return y


def gen_square(freq_norm: Union[float, np.ndarray], n_samp: Optional[int]=None, ampl=1.0, max_freq=0.5) -> np.ndarray:
	"""Generate square wave by additive synthesis
	Guaranteed to be alias-free if freq_norm is a scalar - otherwise, virtually alias-free as long as freq_norm is not
	modulated too quickly

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:param ampl: amplitude
	:return:
	"""

	p, n_harm = _get_phase_n_harm(freq_norm, n_samp, max_freq)
	ampl *= 4.0 / math.pi
	y = np.zeros_like(p)

	for n in range(1, n_harm + 1, 2):
		y += phase_to_sine(p * n) * ampl / n * (n*freq_norm < max_freq)
	return y


def gen_tri(freq_norm: Union[float, np.ndarray], n_samp: Optional[int]=None, ampl=1.0, max_freq=0.5) -> np.ndarray:
	"""Generate triangle wave by additive synthesis
	Guaranteed to be alias-free if freq_norm is a scalar - otherwise, virtually alias-free as long as freq_norm is not
	modulated too quickly

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param n_samp: number of samples
	:param ampl: amplitude
	:return:
	"""

	p, n_harm = _get_phase_n_harm(freq_norm, n_samp, max_freq)
	ampl *= 8.0 / (math.pi ** 2.0)
	y = np.zeros_like(p)

	odd = False
	for n in range(1, n_harm + 1, 2):
		y += phase_to_sine(p * n) * ampl / (n ** 2) * (-1.0 if odd else 1.0) * (n*freq_norm < max_freq)
		odd = not odd
	return y


def plot(args):
	from matplotlib import pyplot as plt
	from .signal_generation import sample_time_index
	from utils import plot_utils

	freq = 1760.
	sample_rate = 48000.
	freq_norm = freq / sample_rate

	n_samp = 128

	saw = gen_saw(freq_norm, n_samp=n_samp)
	squ = gen_square(freq_norm, n_samp=n_samp)
	tri = gen_tri(freq_norm, n_samp=n_samp)

	# Sweep

	start_freq_norm = 20. / sample_rate
	end_freq_norm = 24000. / sample_rate
	n_samp_sweep = 1024
	max_freq = 20. / 48.

	f = np.logspace(np.log2(start_freq_norm), np.log2(end_freq_norm), n_samp_sweep, base=2)

	saw_sweep = gen_saw(f, max_freq=max_freq)
	squ_sweep = gen_square(f, max_freq=max_freq)
	tri_sweep = gen_tri(f, max_freq=max_freq)


	plt.figure()
	t = sample_time_index(n_samp, sample_rate)

	plt.subplot(211)
	plt.plot(t, saw, '.-', label='sawtooth')
	plt.plot(t, squ, '.-', label='square')
	plt.plot(t, tri, '.-', label='triangle')

	plt.grid()
	plt.legend()
	plt.ylabel('Signal')
	plt.xlabel('Time (seconds)')
	plt.title('Alias-free signals, 1.76 kHz @ 48 kHz')

	plt.subplot(212)

	plot_utils.plot_fft(saw, 48000, log=False)
	plot_utils.plot_fft(squ, 48000, log=False)
	plot_utils.plot_fft(tri, 48000, log=False)

	plt.ylabel('FFT (dB)')
	plt.xlabel('Frequency (Hz)')

	plt.figure()
	t = sample_time_index(n_samp_sweep, sample_rate)

	plt.subplot(311)
	plt.plot(t, saw_sweep, '.-')
	plt.grid()
	plt.ylabel('Saw')
	plt.title('Alias-free log frequency sweep, 20 Hz - 24 kHz @ 48 kHz, max 20 kHz')

	plt.subplot(312)
	plt.plot(t, squ_sweep, '.-')
	plt.grid()
	plt.ylabel('Square')

	plt.subplot(313)
	plt.plot(t, tri_sweep, '.-')
	plt.grid()
	plt.xlabel('Time (seconds)')
	plt.ylabel('Triangle')

	plt.show()


def main(args):
	plot(args)
