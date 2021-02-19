#!/usr/bin/env python3


from matplotlib import pyplot as plt
from typing import Callable, Optional, Iterable
import numpy as np
from utils import utils
from generation import signal_generation
from . import freq_response
from utils import plot_utils
import scipy.signal


def _plot_harmonics(f, gains_dB: Optional[Iterable[float]]=None, n_harmonics=20, noise_floor_dB=120):

	if gains_dB is None:
		gains_dB=[-24., -12., -6., 0., 6., 12., 24.]

	sample_rate = 192000.
	freq = 100.
	freq_norm = freq / sample_rate

	n_samp = freq_response.dft_num_samples(freq, sample_rate, max_num_samples=48000, maximize=True)

	for n_gain, gain_dB in enumerate(reversed(gains_dB)):

		gain_lin = utils.from_dB(gain_dB)

		x = signal_generation.gen_sine(freq_norm, n_samp) * gain_lin
		y = f(x)

		idxs = np.arange(1, n_harmonics + 1)

		vals = np.zeros_like(idxs)
		for n, idx in enumerate(idxs):
			vals[n] = freq_response.single_freq_dft(y, freq * idx, sample_rate, mag=True, phase=False, normalize=True)

		# TODO: test this - is multiplying by 2 correct?
		vals = utils.to_dB(2.0 * vals + utils.from_dB(-noise_floor_dB))

		plot_utils.stem_plot_freqs(
			idxs, vals,
			noise_floor_dB=noise_floor_dB,
			label=('%+g dB' % gain_dB),
			set_lims=(n_gain == 0))

	plt.grid()
	plt.legend()


def _plot_intermod(f, gains_dB: Optional[Iterable[float]]=None, noise_floor_dB=120, **kwargs):
	if gains_dB is None:
		gains_dB = [-12., -6., 0., 6., 12.]

	sample_rate = 192000.
	n_samp = nfft = 32768

	f1 = 1000.
	f2 = 1100.

	freqs = np.arange(0., 4000., 100.)

	"""
	Some harmonics:
	
	100:  2nd order IM (f1 - f2)
	900:  3rd order IM (2f1 - f2)
	1000: f1
	1100: f2
	1200: 3rd order IM (2f2 - f1)
	
	2000: f1 2nd harmonic
	2100: 2nd order IM (f1 + f2)
	2200: f2 2nd harmonic
	
	3000: f1 3rd harmonic
	3100: 3rd order IM (2f1 + f2)
	3200: 3rd order IM (2f2 + f1) 
	3300: f2 3rd harmonic
	"""

	w1 = f1 / sample_rate
	w2 = f2 / sample_rate

	random_start_phase = 0.31415926

	x = signal_generation.gen_sine(w1, n_samp) + signal_generation.gen_sine(w2, n_samp, start_phase=random_start_phase)
	x *= 0.5

	results = []
	for n_gain, gain_dB in enumerate(reversed(gains_dB)):
		y = f(x * utils.from_dB(gain_dB))
		dfty = np.zeros_like(freqs, dtype=np.float64)
		for n, freq in enumerate(freqs):
			dfty[n], _ = freq_response.single_freq_dft(y, freq / sample_rate)

		dfty = utils.to_dB(dfty + utils.from_dB(-noise_floor_dB))

		plot_utils.stem_plot_freqs(
			freqs, dfty,
			noise_floor_dB=noise_floor_dB,
			label=('%+g dB' % gain_dB),
			set_lims=(n_gain == 0))

		results.append(dfty)

	def annotate(freq, text):
		idx = [n for n, f in enumerate(freqs) if utils.approx_equal(f, freq)][0]
		y = max([result[idx] for result in results]) + noise_floor_dB + 5
		plt.text(freq, y, text, rotation=45)

	annotate(f1, 'f1')
	annotate(f2, 'f2')

	annotate(2*f1, 'H2')
	annotate(2*f2, 'H2')
	annotate(f2 - f1, 'IM2')
	annotate(f2 + f1, 'IM2')

	annotate(3*f1, 'H3')
	annotate(3*f2, 'H3')
	annotate(2*f1 - f2, 'IM3')
	annotate(2*f2 - f1, 'IM3')
	annotate(2*f1 + f2, 'IM3')
	annotate(2*f2 + f1, 'IM3')

	plt.xlim([0., 4000.])
	plt.grid()
	plt.legend()


def _plot_transfer_function(f):
	x = np.linspace(-10., 10., 20001)
	plt.plot(x, f(x), label="Transfer function")
	plt.title('Transfer function')
	plt.legend()
	plt.grid()


def _plot_derivatives(f, n_derivs):
	x = np.linspace(-10., 10., 80001)
	y = f(x)
	derivs = utils.derivatives(y, x, n_derivs=n_derivs, discontinuity_thresh=10.)

	for dn in range(n_derivs):
		for n in range(len(derivs[dn])):
			if abs(derivs[dn][n]) > 10.:
				derivs[dn][n] = np.nan

	color_cycler = plt.gca()._get_lines.prop_cycler

	for nd, d in enumerate(derivs):
		discont_x = []
		discont_y = []

		for n in range(len(x)):
			if n == 0:
				continue
			if np.isnan(d[n]) and (not np.isnan(d[n - 1])):
				discont_x.append(x[n-1])
				discont_y.append(d[n-1])
			elif (not np.isnan(d[n])) and np.isnan(d[n - 1]):
				discont_x.append(x[n])
				discont_y.append(d[n])

		color = next(color_cycler)['color']
		plt.plot(x, d, label=('Derivative %i' % (nd + 1)), color=color, zorder=-nd)
		plt.scatter(discont_x, discont_y, s=40, facecolors='none', edgecolors=color, zorder=-nd)

	plt.legend()
	plt.grid()


def plot_distortion(f: Callable, title='', n_derivs=4, noise_floor_dB=120):

	fig = plt.figure()

	plt.subplot(2, 2, 1)
	plt.title('Transfer function')
	_plot_transfer_function(f)

	plt.subplot(2, 2, 3)
	plt.title('Derivatives')
	_plot_derivatives(f, n_derivs)

	plt.subplot(2, 2, 4)
	plt.title('Intermodulation')
	_plot_intermod(f)

	plt.subplot(2, 2, 2)
	plt.title('Harmonic distortion')
	_plot_harmonics(f, noise_floor_dB=noise_floor_dB)

	if title:
		fig.suptitle(title)
