#!/usr/bin/env python3

from mpl_toolkits import mplot3d

from matplotlib import pyplot as plt
from matplotlib import cm
from typing import Callable, Optional, Iterable
import numpy as np
from utils import utils
from generation import signal_generation
from . import freq_response
from utils import plot_utils
import scipy.signal
from overdrive import overdrive


# TODO: there's a lot of redundant calculation in 2D & 3D plots; combine them
# TODO: separate plots for even & odd harmonics could be interesting


def _plot_harmonics_3d(ax, f, gains_dB: Optional[Iterable[float]]=None, n_harmonics=16, noise_floor_dB=None):

	if noise_floor_dB is None:
		noise_floor_dB = 200

	if gains_dB is None:
		gains_dB = np.arange(-48, 48, 3)

	n_samp = 1024
	freq_norm = 1.0 / n_samp

	harmonic_numbers = np.arange(1, n_harmonics + 1)

	x = signal_generation.gen_sine(freq_norm, n_samp)

	ax.set_xlabel('Harmonic')
	ax.set_ylabel('Input amplitude (dB)')
	ax.set_zlabel('Output harmonic magnitude (dB)')

	x_grid, y_grid = np.meshgrid(harmonic_numbers, gains_dB)
	z_grid = np.zeros_like(x_grid, dtype=np.float)

	for n_gain, gain_dB in enumerate(gains_dB):

		x_with_gain = x * utils.from_dB(gain_dB)
		y = f(x_with_gain)

		fft_y = np.fft.fft(y) / n_samp
		vals = np.abs(fft_y[1:n_harmonics+1])
		vals = utils.to_dB(vals, min_dB=-noise_floor_dB)

		z_grid[n_gain] = vals

	#ax.plot_wireframe(x_grid, y_grid, z_grid)
	ax.plot_surface(x_grid, y_grid, z_grid, cmap=cm.coolwarm)

	xticks = np.arange(0, n_harmonics, 2)
	ax.set_xticks(xticks)


def _plot_harmonics_2d(ax, f, gains_dB: Optional[Iterable[float]]=None, n_harmonics=16, noise_floor_dB=None):

	if noise_floor_dB is None:
		noise_floor_dB = 96

	if gains_dB is None:
		gains_dB=[-24., -12., -6., 0., 6., 12., 24.]

	n_samp = 1024
	freq_norm = 1.0 / n_samp

	harmonic_numbers = np.arange(1, n_harmonics + 1)

	x = signal_generation.gen_sine(freq_norm, n_samp)

	for n_gain, gain_dB in enumerate(reversed(gains_dB)):

		x_with_gain = x * utils.from_dB(gain_dB)
		y = f(x_with_gain)

		fft_y = np.fft.fft(y) / n_samp
		vals = np.abs(fft_y[1:n_harmonics+1])
		vals = utils.to_dB(vals, min_dB=-noise_floor_dB)

		plot_utils.stem_plot_freqs(
			harmonic_numbers, vals,
			noise_floor_dB=noise_floor_dB,
			label=('%+g dB' % gain_dB),
			set_lims=(n_gain == 0))

	xticks = np.arange(1, n_harmonics, 2)

	ax.set_xticks(xticks)

	ax.grid()
	ax.legend()


def _plot_intermod(ax, f, gains_dB: Optional[Iterable[float]]=None, noise_floor_dB=None):

	if noise_floor_dB is None:
		noise_floor_dB = 96

	# TODO: just use FFT for this, like for plot_harmonics

	if gains_dB is None:
		gains_dB = [-12., -6., 0., 6., 12.]

	sample_rate = 192000.
	n_samp = 32768

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

	#g1 = 0.75
	#g2 = 0.25

	g1 = g2 = 0.5

	x = \
		g1 * signal_generation.gen_sine(w1, n_samp) + \
		g2 * signal_generation.gen_sine(w2, n_samp, start_phase=random_start_phase)

	results = []
	for n_gain, gain_dB in enumerate(reversed(gains_dB)):
		x_with_gain = x * utils.from_dB(gain_dB)
		y = f(x_with_gain)
		dfty = np.zeros_like(freqs, dtype=np.float64)
		for n, freq in enumerate(freqs):
			dfty[n], _ = freq_response.single_freq_dft(y, freq / sample_rate, normalize=True)

		dfty = utils.to_dB(dfty, min_dB=-noise_floor_dB)

		plot_utils.stem_plot_freqs(
			freqs, dfty,
			noise_floor_dB=noise_floor_dB,
			label=('%+g dB' % gain_dB),
			set_lims=(n_gain == 0))

		results.append(dfty)

	def annotate(freq, text):
		idx = [n for n, f in enumerate(freqs) if utils.approx_equal(f, freq)][0]
		y = max([result[idx] for result in results]) + noise_floor_dB + 5
		ax.text(freq, y, text, rotation=45)

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

	ax.set_xlim([0., 4000.])
	ax.grid()
	ax.legend()


def _plot_transfer_function(ax, f):
	x = np.linspace(-10., 10., 10001)
	y = f(x)
	x_plot, y_plot = plot_utils.reduce_plot_points(x, y)
	ax.plot(x_plot, y_plot, label="Transfer function")
	ax.grid()


def _get_discontinuities(x, y):
	discont_x = []
	discont_y = []

	for n in range(1, len(x)):

		if np.isnan(y[n]) and (not np.isnan(y[n - 1])):
			# This is NaN, prev wasn't; use previous
			discont_x.append(x[n - 1])
			discont_y.append(y[n - 1])

		elif (not np.isnan(y[n])) and np.isnan(y[n - 1]):
			# Prev was NaN, this is; use this
			discont_x.append(x[n])
			discont_y.append(y[n])

	return discont_x, discont_y


def _plot_derivatives(ax, f, n_derivs):
	x = np.linspace(-10., 10., 40001)
	y = f(x)
	derivs = utils.derivatives(y, x, n_derivs=n_derivs, discontinuity_thresh=10.)

	for dn in range(n_derivs):
		for n in range(len(derivs[dn])):
			if abs(derivs[dn][n]) > 10.:
				derivs[dn][n] = np.nan

	color_cycler = plt.gca()._get_lines.prop_cycler  # FIXME HACK

	for nd, d in enumerate(derivs):

		x_plot, y_plot = plot_utils.reduce_plot_points(x, d)
		discont_x, discont_y = _get_discontinuities(x, d)

		color = next(color_cycler)['color']
		ax.plot(x_plot, y_plot, label=('Derivative %i' % (nd + 1)), color=color, zorder=-nd)
		ax.scatter(discont_x, discont_y, s=40, facecolors='none', edgecolors=color, zorder=-nd)

	ax.legend()
	ax.grid()


def plot_distortion(f: Callable, title='', n_derivs=4, n_harmonics=16, noise_floor_dB=None):
	"""
	:param f: distortion function; assumed to be memoryless
	:param title: plot title
	:param n_derivs: number of derivatives to plot
	:param n_harmonics: number of harmonics to plot
	:param noise_floor_dB: noise floor (positive dB value)
	:param do_3d_plot: do 3D plot of harmonics
	:return:
	"""

	# FIXME: noise_floor_dB should be negative. But then also have to fix in stem_plot_freqs - have to check if/where else that's used

	fig = plt.figure()

	gs = fig.add_gridspec(3, 3)

	if title:
		fig.suptitle(title)

	ax = fig.add_subplot(gs[0, 0])
	ax.set_title('Transfer function')
	_plot_transfer_function(ax, f)

	ax = fig.add_subplot(gs[1, 0])
	ax.set_title('Derivatives')
	_plot_derivatives(ax, f, n_derivs)

	ax = fig.add_subplot(gs[-1, 1:])
	ax.set_title('Intermodulation')
	_plot_intermod(ax, f, noise_floor_dB=noise_floor_dB)

	ax = fig.add_subplot(gs[:-1, 1:,], projection='3d')
	ax.set_title('Harmonic distortion (3D)')
	_plot_harmonics_3d(ax, f, n_harmonics=n_harmonics, noise_floor_dB=noise_floor_dB)

	ax = fig.add_subplot(gs[-1, 0])
	ax.set_title('Harmonic distortion (2D)')
	_plot_harmonics_2d(ax, f, n_harmonics=n_harmonics, noise_floor_dB=noise_floor_dB)


def main(args):

	asym_hardness = np.vectorize(lambda x: overdrive.clip(x) if x < 0 else overdrive.tanh(x))

	funcs = [
		(overdrive.clip, 'clip'),
		(overdrive.tanh, 'tanh'),
		(asym_hardness, 'Asymmetric hard/tanh'),
	]

	for func, name in funcs:
		print('Processing %s' % name)
		plot_distortion(func, title=name)

	print('Showing plots')
	plt.show()

