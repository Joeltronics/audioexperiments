#!/usr/bin/env python3

import utils
import freq_response
from enum import Enum, unique
from typing import Callable, Iterable, Tuple, Optional
from approx_equal import *
import numpy as np
import math
from matplotlib import pyplot as plt


class FilterUnitTest:
	def __init__(
			self,
			name,
			constructor: Callable,
			freqs_to_test: Iterable[float],
			expected_freq_response_range_dB: Iterable[Tuple[Optional[float], Optional[float]]],
			expected_phase_response_range_degrees: Optional[Iterable[Optional[Tuple[float, float]]]]=None,
			deterministic: bool=True,
			linear: bool=True):
		"""

		:param constructor:
		:param wc: cutoff frequency
		:param freqs_to_test: frequencies to test
		:param expected_freq_response_range_dB: expected response at frequencies, in dB - min and max, or None to leave one side open
		:param deterministic: True if filter is deterministic (aside from small floating-point and/or quantization error)
		:param linear: True if filter is linear (aside from small floating-point and/or quantization error)
		"""

		self.constructor = constructor
		self.name = name

		self.freqs_to_test = freqs_to_test
		self.expected_freq_response_range_dB = expected_freq_response_range_dB
		self.expected_phase_response_degrees = expected_phase_response_range_degrees
		self.deterministic = deterministic
		self.linear = linear

	def __call__(self, *args, **kwargs) -> None:
		"""

		:raises: AssertionError if test failed
		"""

		#plot_on_failure = False
		#if 'plot_on_failure' in kwargs:
		#	plot_on_failure = kwargs[plot_on_failure]

		plot_on_failure=True

		# Test reset first - if it doesn't work, remaining tests will be invalid
		self._test_reset(plot_on_failure=plot_on_failure)
		self._test_freq_response(plot_on_failure=plot_on_failure)
		self._test_samp_vs_vect(plot_on_failure=plot_on_failure)

		#if plot_on_failure:
		#	plt.show()

	@staticmethod
	def _test_resp(actual, expected_range, phase=False) -> bool:
		if expected_range is None:
			return True
		# TODO: use modular arithmetic if phase
		above_min = (actual >= expected_range[0]) if expected_range[0] is not None else True
		below_max = (actual <= expected_range[1]) if expected_range[1] is not None else True
		return above_min and below_max

	def _plot_mag_failure(self, resp):
		plt.figure()

		thresh_low = [val[0] for val in self.expected_freq_response_range_dB]
		thresh_high = [val[1] for val in self.expected_freq_response_range_dB]

		plt.semilogx(self.freqs_to_test, resp)
		plt.semilogx(self.freqs_to_test, thresh_low, 'r.')
		plt.semilogx(self.freqs_to_test, thresh_high, 'r.')

		max_amp = math.ceil(max(np.amax(resp), np.amax(thresh_low), np.amax(thresh_high)) / 6.0) * 6.0
		min_amp = math.floor(min(np.amin(resp), np.amin(thresh_low), np.amin(thresh_high)) / 6.0) * 6.0
		plt.yticks(np.arange(min_amp, max_amp + 6, 6))

		plt.xlabel("Frequency")
		plt.ylabel("Magnitude Response (dB)")

		plt.title('Unit test failure: %s' % self.name)
		plt.grid()

		plt.show()

	@staticmethod
	def _is_linear(f, mag_dB, rms_mag_dB) -> bool:
		# Dynamically determine eps based on f and magnitudes

		# TODO: figure out why all this is necessary
		# Shouldn't FFT vs RMS linearity check still work even at very small amplitudes and high freqs?

		# If amplitude is very low, increase eps
		# Amplitude   eps
		#    0 dB    0.25 dB
		#   -5 dB    0.25 dB
		#  -10 dB    0.5 dB
		#  -20 dB    1 dB
		#  -40 dB    2 dB
		eps = max(0.25, -mag_dB / 20.)

		if f >= 0.5:
			return True

		# For freqs above 0.1 cycles/sample, increase even further
		eps_adj = utils.scale(f, (0.1, 0.5), (1., 0.), clip=True)
		assert eps_adj != 0.0
		eps /= eps_adj

		return approx_equal_scalar(mag_dB, rms_mag_dB, eps=eps)

	def _test_freq_response(self, plot_on_failure=False) -> None:
		"""Test frequency response, and linearity if linear

		:raises: AssertionError if test failed
		"""

		# Also includes testing linearity

		filt = self.constructor()

		check_phase = self.expected_phase_response_degrees is not None

		freq_resp_ret = freq_response.get_freq_response(
			filt,
			self.freqs_to_test,
			sample_rate=1.0,
			mag=True,
			phase=check_phase,
			rms=self.linear,
			group_delay=False)

		freq_resp_mag = utils.to_dB(freq_resp_ret[0] if isinstance(freq_resp_ret, tuple) else freq_resp_ret)

		def test_linear():
			freq_resp_rms = utils.to_dB(freq_resp_ret[1] * math.sqrt(2.0))
			for f, f_mag, f_rms in zip(self.freqs_to_test, freq_resp_mag, freq_resp_rms):
				if not self._is_linear(f, f_mag, f_rms):
					raise AssertionError(
						'Test failed: %s is nonlinear - input sine wave freq %f, mag from DFT %.2f dB, mag from RMS %.2f dB' %
						(self.name, f, f_mag, f_rms))

		def test_freq_resp():
			for f, actual, expected in zip(self.freqs_to_test, freq_resp_mag, self.expected_freq_response_range_dB):
				if not self._test_resp(actual, expected):
					if plot_on_failure:
						self._plot_mag_failure(freq_resp_mag)
					raise AssertionError(
						'Test failed: %s, freq %f, expected (%s, %s) dB, actual %.2f dB' %
						(self.name, f, utils.to_pretty_str(expected[0]), utils.to_pretty_str(expected[1]), actual))

		def test_phase():
			freq_resp_phase_deg = np.rad2deg(freq_resp_ret[-1])
			for f, actual, expected in zip(self.freqs_to_test, freq_resp_phase_deg, self.expected_phase_response_degrees):
				if not self._test_resp(actual, expected, phase=True):
					if plot_on_failure:
						self._plot_phase_failure(freq_resp_phase_deg)
					raise AssertionError(
						'Test failed: %s, freq %f, expected phase (%s, %s) degrees, actual %.1f' %
						(self.name, f, utils.to_pretty_str(expected[0]), utils.to_pretty_str(expected[1]), actual))

		test_freq_resp()

		if self.linear:
			test_linear()

		if check_phase:
			test_phase()


	def _test_samp_vs_vect(self, plot_on_failure=False) -> None:
		"""

		:raises: AssertionError if test failed
		"""

		if not self.deterministic:
			return

		# TODO: test that these all return the same:
		# * process_sample
		# * process_vector on entire vector
		# * several process_vector in a row all return the same
		filt = self.constructor()
		pass

	def _test_reset(self, plot_on_failure=False) -> None:
		"""

		:raises: AssertionError if test failed
		"""

		if not self.deterministic:
			return

		# TODO: test that reset() and freshly constructed return same results
		filt = self.constructor()
		pass
