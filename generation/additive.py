#!/usr/bin/env python3


from .signal_generation import gen_phase, phase_to_sine
from utils import utils

import numpy as np
import math
from typing import Union, Optional


def _get_phase_n_harm(freq_norm, n_samp, max_freq):
	"""
	:return: phase signal, number of harmonics
	"""
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
	:param max_freq: freq to stop at
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
	:param max_freq: freq to stop at
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
	:param max_freq: freq to stop at
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


def gen_arbitrary(
		freq_norm: Union[float, np.ndarray],
		odd_power: Union[None, float, np.ndarray],
		even_power: Union[None, float, np.ndarray],
		n_samp: Optional[int]=None,
		ampl=1.0,
		fundamental_scale=1.0,
		max_freq=0.5,
		alternate_odd=False,
		alternate_even=False,
		) -> np.ndarray:
	"""
	Generate waveform with arbitrary harmonic slope

	Examples:
		Sawtooth: gen_arbitrary(odd_power=1, even_power=1)
		Square:   gen_arbitrary(odd_power=1, even_power=None)
		Triangle: gen_arbitrary(odd_power=2, even_power=None, alternate_odd=True)

	Guaranteed to be alias-free if freq_norm is a scalar - otherwise, virtually alias-free as long as freq_norm is not
	modulated too quickly

	:param freq_norm: normalized frequency, i.e. freq / sample rate
	:param odd_power: power decay of even harmonics - e.g. at 2, even harmonics scale by n**-2
	:param even_power: power decay of odd harmonics - e.g. at 2, odd harmonics scale by n**-2
	:param n_samp: number of samples
	:param ampl: amplitude
	:param fundamental_scale: amplitude of fundamental, relative to default
	:param max_freq: freq to stop at
	:param alternate_odd: if odd harmonics should alternate in polarity, as in a triangle wave
	:param alternate_even: if even harmonics should alternate in polarity
	:return:
	"""

	p, n_harm = _get_phase_n_harm(freq_norm, n_samp, max_freq)

	y = np.zeros_like(p)

	ampl_sum = 0.0

	odd_even = True
	odd_odd = True

	for n in range(1, n_harm + 1):

		polarity = 1.0

		if n == 1:
			this_harmonic_ampl = fundamental_scale
			ampl_sum += this_harmonic_ampl

		else:
			if utils.is_even(n):
				if even_power is None:
					continue
				this_harmonic_power = even_power

				if alternate_even:
					polarity = (-1.0 if odd_even else 1.0)
					odd_even = not odd_even

			elif utils.is_odd(n):
				if odd_power is None:
					continue
				this_harmonic_power = odd_power

				if alternate_odd:
					polarity = (-1.0 if odd_odd else 1.0)
				odd_odd = not odd_odd

			else:
				raise AssertionError('Number is neither even nor odd?!')

			this_harmonic_ampl = 1.0 / (n ** this_harmonic_power)

			ampl_sum += this_harmonic_ampl / (math.pi ** this_harmonic_power)

		y += phase_to_sine(p * n) * this_harmonic_ampl * polarity * (n*freq_norm < max_freq)

	y *= (ampl / ampl_sum)

	return y


def plot(args):
	from matplotlib import pyplot as plt
	from .signal_generation import sample_time_index
	from utils import plot_utils

	def _plot_basic_waves():
		n_samp = 128

		freq = 1760.
		sample_rate = 48000.
		freq_norm = freq / sample_rate

		saw = gen_saw(freq_norm, n_samp=n_samp)
		squ = gen_square(freq_norm, n_samp=n_samp)
		tri = gen_tri(freq_norm, n_samp=n_samp)

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
		plt.grid()
		plt.ylabel('FFT (dB)')
		plt.xlabel('Frequency (Hz)')

	def _plot_arbitrary():
		n_cycles = 4

		n_samp = 512
		freq_norm = n_cycles / n_samp

		saw = gen_arbitrary(freq_norm, n_samp=n_samp, odd_power=1, even_power=1)
		squ = gen_arbitrary(freq_norm, n_samp=n_samp, odd_power=1, even_power=None)
		tri = gen_arbitrary(freq_norm, n_samp=n_samp, odd_power=2, even_power=None, alternate_odd=True)
		anti_tri = gen_arbitrary(freq_norm, n_samp=n_samp, odd_power=None, even_power=2, alternate_even=True)
		full_tri = gen_arbitrary(freq_norm, n_samp=n_samp, odd_power=2, even_power=2, alternate_even=True, alternate_odd=True)
		i_tri = gen_arbitrary(freq_norm, n_samp=n_samp, odd_power=3, even_power=None, alternate_odd=True)

		# Why is this not an impulse train? (dirac comb)
		full_even_odd = gen_arbitrary(freq_norm, n_samp=n_samp, odd_power=0, even_power=0, alternate_even=True, alternate_odd=True)

		full_odd = gen_arbitrary(freq_norm, n_samp=n_samp, odd_power=0, even_power=None, alternate_even=True, alternate_odd=True)

		full_even = gen_arbitrary(freq_norm, n_samp=n_samp, odd_power=None, even_power=0, alternate_even=True, alternate_odd=True)

		plt.figure()

		plt.subplot(2, 2, 1)
		#fmt = '.-'
		fmt = '-'
		plt.plot(saw, fmt, label='sawtooth')
		plt.plot(squ, fmt, label='square')
		plt.plot(full_tri, fmt, label='full-triangle')
		plt.plot(tri, fmt, label='triangle')
		plt.plot(anti_tri, fmt, label='anti-triangle')
		plt.plot(i_tri, fmt, label='integral-triangle')
		plt.grid()
		plt.legend()
		plt.ylabel('Signal')
		plt.title('Alias-free arbitrary signals')

		plt.subplot(2, 2, 2)
		# fmt = '.-'
		fmt = '-'
		plt.plot(full_odd, fmt, label='full odd')
		plt.plot(full_even, fmt, label='full even')
		plt.plot(full_even_odd, fmt, label='full even+odd')
		plt.grid()
		plt.legend()
		plt.ylabel('Signal')
		plt.title('Alias-free arbitrary signals')

		plt.subplot(2, 2, 3)
		#kwargs = dict(log=False, nfft=n_samp, window=False, noise_floor=utils.from_dB(-120))
		kwargs = dict(log=True, nfft=n_samp, window=False, noise_floor=utils.from_dB(-120))
		#fmt = '.-'
		fmt = '.'
		plot_utils.plot_fft(saw, 1.0, fmt, **kwargs)
		plot_utils.plot_fft(squ, 1.0, fmt, **kwargs)
		plot_utils.plot_fft(full_tri, 1.0, fmt, **kwargs)
		plot_utils.plot_fft(tri, 1.0, fmt, **kwargs)
		plot_utils.plot_fft(anti_tri, 1.0, fmt, **kwargs)
		plot_utils.plot_fft(i_tri, 1.0, fmt, **kwargs)
		plt.grid()
		plt.ylabel('FFT (dB)')
		plt.xlabel('Frequency (Hz)')

		plt.subplot(2, 2, 4)
		# kwargs = dict(log=False, nfft=n_samp, window=False, noise_floor=utils.from_dB(-120))
		kwargs = dict(log=True, nfft=n_samp, window=False, noise_floor=utils.from_dB(-120))
		# fmt = '.-'
		fmt = '.'
		plot_utils.plot_fft(full_odd, 1.0, fmt, **kwargs)
		plot_utils.plot_fft(full_even, 1.0, fmt, **kwargs)
		plot_utils.plot_fft(full_even_odd, 1.0, fmt, **kwargs)
		plt.grid()
		plt.ylabel('FFT (dB)')
		plt.xlabel('Frequency (Hz)')

	def _plot_sweep():
		sample_rate = 48000.

		start_freq_norm = 20. / sample_rate
		end_freq_norm = 24000. / sample_rate
		n_samp_sweep = 1024
		max_freq = 20. / 48.

		f = np.logspace(np.log2(start_freq_norm), np.log2(end_freq_norm), n_samp_sweep, base=2)

		saw_sweep = gen_saw(f, max_freq=max_freq)
		squ_sweep = gen_square(f, max_freq=max_freq)
		tri_sweep = gen_tri(f, max_freq=max_freq)

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

	_plot_basic_waves()
	_plot_arbitrary()
	_plot_sweep()

	plt.show()


def main(args):
	plot(args)
