#!/usr/bin/env python3

from typing import Callable, Optional

from matplotlib import pyplot as plt
import numpy as np

from generation.additive import gen_saw as gen_bandlimited_saw
from generation.additive import gen_square as gen_bandlimited_square
from generation.signal_generation import gen_sine, gen_saw, gen_square
from utils.utils import to_dB, from_dB


EPS = 1e-12


def get_real_cepstrum_from_fft(f: np.ndarray, eps=EPS) -> np.ndarray:
	mag = np.abs(f)
	mag = np.maximum(mag, eps)
	log_mag = np.log(mag)
	return np.real(np.fft.ifft(log_mag))


def get_power_cepstrum_from_fft(f: np.ndarray, eps=EPS) -> np.ndarray:
	mag = np.abs(f)
	mag = np.maximum(mag, eps)
	power = np.square(mag)
	log_power = np.log(power)
	c = np.fft.ifft(log_power)
	return np.square(np.abs(c))


def get_complex_cepstrum_from_fft(f: np.ndarray, eps=EPS) -> np.ndarray:
	mag = np.abs(f)
	mag = np.maximum(mag, eps)
	phase = np.angle(f)
	complex_log_f = np.log(mag) + 1j * phase
	return np.fft.ifft(complex_log_f)


def get_cepstrum(x: np.ndarray, window_func: Optional[Callable] = None, cepstrum_type = 'real') -> np.ndarray:

	if window_func is not None:
		x = x * window_func(len(x))

	f = np.fft.fft(x)

	if cepstrum_type == 'real':
		return get_real_cepstrum_from_fft(f)
	elif cepstrum_type == 'power':
		return get_power_cepstrum_from_fft(f)
	elif cepstrum_type == 'complex':
		return get_complex_cepstrum_from_fft(f)
	else:
		raise ValueError(f'Invalid cepstrum type: {cepstrum_type}')


def plot_cepstrum(x: np.ndarray, title: str):

	noise_floor_dB = -120

	num_samp = len(x)

	window = np.hamming(num_samp)
	#window = None

	xw = x * window if (window is not None) else x

	f = np.fft.fft(xw)
	mag_dB = to_dB(np.abs(f), min_dB=noise_floor_dB)

	f += from_dB(noise_floor_dB)
	real_cepstrum = get_real_cepstrum_from_fft(f)
	complex_cepstrum = get_complex_cepstrum_from_fft(f)
	power_cepstrum = get_power_cepstrum_from_fft(f)

	#
	# Plot
	#

	fig = plt.figure()
	fig.suptitle(title)

	gs = fig.add_gridspec(4, 2)

	#ax_t = subplots[0]
	#ax_f = subplots[1]
	#ax_c = subplots[2]
	#ax_cepstrum_1_over_freq = subplots[3]

	ax_t = fig.add_subplot(gs[0, :])

	ax_real_cepstrum = fig.add_subplot(gs[1, 0])
	ax_power_cepstrum = fig.add_subplot(gs[2, 0])
	ax_complex_cepstrum = fig.add_subplot(gs[3, 0])

	ax_f = fig.add_subplot(gs[1, 1])
	ax_f_log = fig.add_subplot(gs[2, 1])
	ax_cepstrum_1_over_freq = fig.add_subplot(gs[3, 1])

	t = np.arange(num_samp)
	freqs = np.arange(num_samp) / num_samp
	freqs += 0.5 * (freqs[1] - freqs[0])
	quefs = np.arange(num_samp)

	ax_t.plot(t, x, label='Signal')
	if window is not None:
		ax_t.plot(t, window, label='Window', zorder=-1)

	ax_f.plot(freqs, mag_dB, label='Magnitude')
	ax_f_log.semilogx(freqs, mag_dB, label='Magnitude')

	ax_real_cepstrum.plot(quefs, real_cepstrum, label='Real Cepstrum')
	ax_power_cepstrum.plot(quefs, power_cepstrum, label='Power Cepstrum')
	ax_complex_cepstrum.plot(quefs, np.real(complex_cepstrum), label='Complex Cepstrum (real)')
	ax_complex_cepstrum.plot(quefs, np.imag(complex_cepstrum), label='Complex Cepstrum (imag)')
	ax_complex_cepstrum.plot(quefs, np.abs(complex_cepstrum), label='Complex Cepstrum (mag)', zorder=-1)
	ax_complex_cepstrum.legend()

	x = 1.0 / quefs[1:num_samp//2]
	y = real_cepstrum[1:num_samp//2]
	#ax_cepstrum_1_over_freq.plot(x, y, label='Real Cepstrum')
	ax_cepstrum_1_over_freq.semilogx(x, y, label='Real Cepstrum')

	ax_t.legend()

	#ax_t.set_title('Time domain')
	#ax_f.set_title('Spectrum')
	#ax_c.set_title('Real Cepstrum')

	#ax_t.set_xlabel('time')

	#ax_f.set_xlabel('frequency')
	ax_f.set_ylabel('dB')

	ax_f_log.set_ylabel('dB')

	ax_cepstrum_1_over_freq.set_xlabel('1/quefrency')
	ax_cepstrum_1_over_freq.set_ylabel('Cepstrum')

	ax_t.set_xlim([0, num_samp])
	ax_real_cepstrum.set_xlim([0, num_samp//2])
	ax_power_cepstrum.set_xlim([0, num_samp//2])
	ax_complex_cepstrum.set_xlim([0, num_samp//2])
	ax_f.set_xlim([0, 0.5])
	ax_f_log.set_xlim([freqs[1], 0.5])
	ax_cepstrum_1_over_freq.set_xlim([freqs[1], 0.5])

	ax_real_cepstrum.set_ylabel('Real cepstrum')
	ax_power_cepstrum.set_ylabel('Power cepstrum')
	ax_complex_cepstrum.set_ylabel('Complex cepstrum')

	ax_t.grid()
	ax_f.grid()
	ax_f_log.grid()
	ax_cepstrum_1_over_freq.grid()
	ax_real_cepstrum.grid()
	ax_power_cepstrum.grid()
	ax_complex_cepstrum.grid()


def gen_test_sig(n_samp):

	# Basic harmonics
	x = 0.5 * gen_sine(0.01, n_samp, start_phase=0.0)
	x += 0.1 * gen_sine(0.02, n_samp, start_phase=0.1)
	x += 0.12 * gen_sine(0.03, n_samp, start_phase=0.667)
	x += 0.01 * gen_sine(0.04, n_samp, start_phase=0.923)
	x += 0.09 * gen_sine(0.05, n_samp, start_phase=0.31)
	x += 0.004 * gen_sine(0.06, n_samp, start_phase=0.719)
	x += 0.009 * gen_sine(0.07, n_samp, start_phase=0.118)
	x += 0.02 * gen_sine(0.08, n_samp, start_phase=0.52)

	# Inharmonic content
	x += 0.031 * gen_sine(0.0713294, n_samp)
	x += 0.0201 * gen_sine(0.0267, n_samp)

	return x


def plot(args):
	num_samp = 1024
	plot_cepstrum(gen_saw(0.01, num_samp), 'Naive saw')
	plot_cepstrum(gen_saw(10 / num_samp, num_samp), 'Naive saw, exact bin')
	plot_cepstrum(gen_bandlimited_saw(0.01, num_samp, 1.0, 0.5), 'Ideal bandlimited saw')
	plot_cepstrum(gen_bandlimited_saw(0.01, num_samp, 1.0, 0.25), '0.25 bandlimited saw')
	plot_cepstrum(gen_bandlimited_saw(0.01, num_samp, 1.0, 0.1), '0.1 bandlimited saw')
	plot_cepstrum(gen_square(0.01, num_samp), 'Naive square')
	plot_cepstrum(gen_square(10 / num_samp, num_samp), 'Naive square, exact bin')
	plot_cepstrum(gen_bandlimited_square(0.01, num_samp, 1.0, 0.5), 'Ideal bandlimited square')
	plot_cepstrum(gen_bandlimited_square(0.01, num_samp, 1.0, 0.1), '0.1 bandlimited square')
	plot_cepstrum(gen_sine(0.01, num_samp), 'Sine')
	plot_cepstrum(gen_test_sig(num_samp), 'Test signal')
	plt.show()


def main(args):
	plot(args)
