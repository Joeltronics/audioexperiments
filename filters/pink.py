#!/usr/bin/env python3

"""
"Pinking" filters, i.e. filters with a slope of -3 dB/octave, for generating pink noise from white noise
"""

from .filter_base import IIRFilterBase
import numpy as np
from utils import utils
from typing import Tuple, List
from .one_pole import BasicOnePole
from processor import ParallelProcessors
from .filter_base import ParallelFilters


class BasicPinkFilter(IIRFilterBase):
	"""Pink noise (1/f) filter
	Intended for 44.1-48 kHz; will work at higher sample rates but low frequency response will be off
	"""
	def __init__(self, wc=None, verbose=False):
		# https://ccrma.stanford.edu/~jos/sasp/Example_Synthesis_1_F_Noise.html
		super().__init__(
			a=[1, -2.494956002, 2.017265875, -0.522189400],
			b=[0.049922035, -0.095993537, 0.050612699, -0.004408786])


class PinkFilter(ParallelProcessors):
	def __init__(self, sample_rate, pole_zero_ratio=2.0):
		self.sample_rate = sample_rate
		freqs, gains = self._calc_individual_filters(self.sample_rate, pole_zero_ratio=pole_zero_ratio)
		filters = [BasicOnePole(wc=f/sample_rate, gain=g) for f, g in zip(freqs, gains)]
		super().__init__(filters)

	@staticmethod
	def _calc_individual_filters(sample_rate, pole_zero_ratio, start_freq_Hz=10.0, downward=False) -> Tuple[np.ndarray, np.ndarray]:
		# http://www.firstpr.com.au/dsp/pink-noise/#Filtering
		# http://www.cooperbaker.com/home/code/pink%20noise/

		freqs = []
		gains = []

		pzr_sq = pole_zero_ratio ** 2.0

		gain = 1.0

		if downward:
			hz = sample_rate / (2 * np.pi)
			while hz > start_freq_Hz:
				freqs.append(hz)
				gains.append(gain)
				gain *= pole_zero_ratio
				hz /= pzr_sq

			gain_norm = 1.0 / sum(gains)

		else:
			hz = start_freq_Hz
			while hz < (0.5*sample_rate):
				freqs.append(hz)
				gains.append(gain)
				gain /= pole_zero_ratio
				hz *= pzr_sq

			# FIXME
			# Figured this gain out empirically - works for 2 and sqrt(2), but nothing else
			#gain_norm = 0.125 * pzr_sq

			gain_norm = 1.0 / sum(gains)

		freqs = np.array(freqs)
		gains = np.array(gains) * gain_norm

		return freqs, gains


def test(verbose=False):
	import numpy as np
	from unit_test.processor_unit_test import ProcessorUnitTest
	from unit_test.unit_test import run_unit_tests
	from math import sqrt

	sqrt2 = sqrt(2.0)

	freqs = np.array([50., 100., 200., 400., 800., 1600., 3200., 6400., 12800.]) / 44100.

	tol = 0.5
	expected_vals = -3.0 * np.arange(len(freqs)) - 4.0
	expected_dB = [(val - tol, val + tol) for val in expected_vals]

	tests = [
		ProcessorUnitTest(
			"BasicPinkFilter()",
			lambda: BasicPinkFilter(),
			freqs_to_test=freqs,
			expected_freq_response_range_dB=expected_dB,
			expected_phase_response_range_degrees=None,
			deterministic=True,
			linear=True
		)
	]

	freqs = np.array([10., 31.6, 100., 316., 1000., 3162., 10000.]) / 44100.

	tol = 0.5
	expected_vals = -5.0 * np.arange(len(freqs))
	expected_dB = [(val - tol, val + tol) for val in expected_vals]
	expected_dB[0] = (-3., 0.)

	tests += [
		ProcessorUnitTest(
			"PinkFilter(44.1 kHz)",
			lambda: PinkFilter(44100.),
			freqs_to_test=freqs,
			expected_freq_response_range_dB=expected_dB,
			expected_phase_response_range_degrees=None,
			deterministic=True,
			linear=True,
			sample_rate=44100,
		),
		ProcessorUnitTest(
			"PinkFilter(44.1 kHz, sqrt2)",
			lambda: PinkFilter(44100., pole_zero_ratio=sqrt2),
			freqs_to_test=freqs,
			expected_freq_response_range_dB=expected_dB,
			expected_phase_response_range_degrees=None,
			deterministic=True,
			linear=True,
			sample_rate=44100,
		),
	]

	freqs = np.array([10., 31.6, 100., 316., 1000., 3162., 10000., 31623.]) / 96000.

	tol = 0.5
	expected_vals = -5.0 * np.arange(len(freqs))
	expected_dB = [(val - tol, val + tol) for val in expected_vals]
	expected_dB[0] = (-3., 0.)

	tests += [
		ProcessorUnitTest(
			"PinkFilter(96 kHz)",
			lambda: PinkFilter(96000.),
			freqs_to_test=freqs,
			expected_freq_response_range_dB=expected_dB,
			expected_phase_response_range_degrees=None,
			deterministic=True,
			linear=True,
			sample_rate=96000,
		),
		ProcessorUnitTest(
			"PinkFilter(96 kHz, sqrt2)",
			lambda: PinkFilter(96000., pole_zero_ratio=sqrt2),
			freqs_to_test=freqs,
			expected_freq_response_range_dB=expected_dB,
			expected_phase_response_range_degrees=None,
			deterministic=True,
			linear=True,
			sample_rate=96000,
		),
		ProcessorUnitTest(
			"PinkFilter(96 kHz, 1.1)",
			lambda: PinkFilter(96000., pole_zero_ratio=1.1),
			freqs_to_test=freqs,
			expected_freq_response_range_dB=expected_dB,
			expected_phase_response_range_degrees=None,
			deterministic=True,
			linear=True,
			sample_rate=96000,
		),
		ProcessorUnitTest(
			"PinkFilter(96 kHz, 4)",
			lambda: PinkFilter(96000., pole_zero_ratio=4),
			freqs_to_test=freqs,
			expected_freq_response_range_dB=expected_dB,
			expected_phase_response_range_degrees=None,
			deterministic=True,
			linear=True,
			sample_rate=96000,
		),
	]

	return run_unit_tests(tests, verbose=verbose)


def plot(args):
	from matplotlib import pyplot as plt
	import numpy as np
	from math import sqrt

	from utils.plot_utils import plot_freq_resp

	sqrt2 = sqrt(2.0)
	sample_rate = 48000.

	def setup_plot():
		fig, ax = plt.subplots(2, 1)
		x = [10, 10000]
		y = [0, -30]  # 10 dB/decade (often cited as 3 dB/octave, which is very close but not quite exact)
		ax[0].plot(x, y, 'g-')
		ax[1].axhline(-45, color='g')
		return fig, ax

	print('Processing & plotting...')

	plot_freqs = np.logspace(np.log10(1.0), np.log10(20000.0), 64, base=10)

	_, axes = setup_plot()
	plot_freq_resp(BasicPinkFilter, None, None, plot_freqs, sample_rate, n_samp=48000, phase=True, axes=axes)

	_, axes = setup_plot()
	plot_freq_resp(PinkFilter, None, dict(sample_rate=48000), plot_freqs, 48000, n_samp=48000, phase=True, axes=axes)

	_, axes = setup_plot()
	plot_freq_resp(PinkFilter, None, dict(sample_rate=96000), plot_freqs, 96000, n_samp=96000, phase=True, axes=axes)

	_, axes = setup_plot()
	plot_freq_resp(PinkFilter, None, dict(sample_rate=96000, pole_zero_ratio=sqrt2), plot_freqs, 96000, n_samp=96000, phase=True, axes=axes)

	_, axes = setup_plot()
	plot_freq_resp(PinkFilter, None, dict(sample_rate=96000, pole_zero_ratio=1.1), plot_freqs, 96000, n_samp=96000, phase=True, axes=axes)

	_, axes = setup_plot()
	plot_freq_resp(PinkFilter, None, dict(sample_rate=96000, pole_zero_ratio=4), plot_freqs, 96000, n_samp=96000, phase=True, axes=axes)

	for pole_zero_ratio in [2, sqrt2, 4]:
		cutoff_freqs, gains = PinkFilter._calc_individual_filters(sample_rate, pole_zero_ratio=pole_zero_ratio)
		args_list = [dict(wc=freq/sample_rate, gain=gain) for freq, gain in zip(cutoff_freqs, gains)]
		plot_freq_resp(BasicOnePole, None, args_list, plot_freqs, 48000, n_samp=48000, freq_args=['wc'])
		plt.title('Individual filters in PinkFilter, pole-zero ratio %g' % pole_zero_ratio)

	print('Showing plots')
	plt.show()


def main(args):
	test(args)
	plot(args)
