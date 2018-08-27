#!/usr/bin/env python3

import utils
import freq_response
from enum import Enum, unique
from typing import Callable, Iterable, Tuple, Optional
from approx_equal import *
import numpy as np
import math
from matplotlib import pyplot as plt
import signal_generation
from unit_test import UnitTestFailure


class FilterUnitTest:
	def __init__(
			self,
			name,
			constructor: Callable,
			freqs_to_test: Iterable[float],
			expected_freq_response_range_dB: Iterable[Tuple[Optional[float], Optional[float]]],
			expected_phase_response_range_degrees: Optional[Iterable[Optional[Tuple[float, float]]]]=None,
			deterministic: bool=True,
			linear: bool=True,
			amplitude: float=1.0):
		"""

		:param constructor:
		:param wc: cutoff frequency
		:param freqs_to_test: frequencies to test
		:param expected_freq_response_range_dB: expected response at frequencies, in dB - min and max, or None to leave one side open
		:param deterministic: True if filter is deterministic (aside from small floating-point and/or quantization error)
		:param linear: True if filter is linear (aside from small floating-point and/or quantization error)
		:param amplitude: amplitude to run unit test at (in case filter is nonlinear)
		"""

		self.constructor = constructor
		self.name = name

		self.freqs_to_test = freqs_to_test
		self.expected_freq_response_range_dB = expected_freq_response_range_dB
		self.expected_phase_response_degrees = expected_phase_response_range_degrees
		self.deterministic = deterministic
		self.linear = linear
		self.amplitude = amplitude

	def __call__(self, *args, **kwargs) -> None:
		"""

		:raises: AssertionError if test failed
		"""

		#plot_on_failure = False
		#if 'plot_on_failure' in kwargs:
		#	plot_on_failure = kwargs[plot_on_failure]

		plot_on_failure=True

		# Test sanity first - if it doesn't work, remaining tests will be invalid
		self._test_sanity(plot_on_failure=plot_on_failure)
		self._test_freq_response(plot_on_failure=plot_on_failure)

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

	def _plot_equal_failure(self, x1, x2, x1_label=None, x2_label=None):
		plt.figure()

		plt.subplot(211)

		plt.plot(x1, label=x1_label)
		plt.plot(x2, label=x2_label)

		plt.grid()

		if x1_label or x2_label:
			plt.legend()

		plt.title('Unit test failure: %s' % self.name)

		plt.subplot(212)

		plt.plot(np.abs(x2-x1))
		plt.grid()
		plt.ylabel('abs error')

		plt.xlabel('Sample number')

		plt.show()

	@staticmethod
	def _is_linear(f, mag_dB, rms_mag_dB) -> bool:
		# Dynamically determine eps based on f and magnitudes

		# FIXME: this doesn't even work properly

		# Shouldn't FFT vs RMS linearity check still work even at very small amplitudes and high freqs?
		# Even then, there are still some false failures

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
			group_delay=False,
			amplitude=self.amplitude)

		freq_resp_mag = utils.to_dB(freq_resp_ret[0] if isinstance(freq_resp_ret, tuple) else freq_resp_ret)

		def test_linear():
			freq_resp_rms = utils.to_dB(freq_resp_ret[1] * math.sqrt(2.0))
			for f, f_mag, f_rms in zip(self.freqs_to_test, freq_resp_mag, freq_resp_rms):
				if not self._is_linear(f, f_mag, f_rms):
					raise UnitTestFailure(
						'Test failed: %s is nonlinear - input sine wave freq %f, mag from DFT %.2f dB, mag from RMS %.2f dB' %
						(self.name, f, f_mag, f_rms))

		def test_freq_resp():
			for f, actual, expected in zip(self.freqs_to_test, freq_resp_mag, self.expected_freq_response_range_dB):
				if not self._test_resp(actual, expected):
					if plot_on_failure:
						self._plot_mag_failure(freq_resp_mag)
					raise UnitTestFailure(
						'Test failed: %s, freq %f, expected (%s, %s) dB, actual %.2f dB' %
						(self.name, f, utils.to_pretty_str(expected[0]), utils.to_pretty_str(expected[1]), actual))

		def test_phase():
			freq_resp_phase_deg = np.rad2deg(freq_resp_ret[-1])
			for f, actual, expected in zip(self.freqs_to_test, freq_resp_phase_deg, self.expected_phase_response_degrees):
				if not self._test_resp(actual, expected, phase=True):
					if plot_on_failure:
						self._plot_phase_failure(freq_resp_phase_deg)
					raise UnitTestFailure(
						'Test failed: %s, freq %f, expected phase (%s, %s) degrees, actual %.1f' %
						(self.name, f, utils.to_pretty_str(expected[0]), utils.to_pretty_str(expected[1]), actual))

		test_freq_resp()

		# In theory we should test linearity first, before freq response
		# Because if filter is nonlinear then freq response is invalid
		# Except that would only be correct behavior if linearity check actually worked properly
		if self.linear:
			test_linear()

		if check_phase:
			test_phase()

	def _test_sanity(self, plot_on_failure=False) -> None:
		"""

		:raises: AssertionError if test failed
		"""

		if not self.deterministic:
			return

		#
		# Test that reset() and freshly constructed return same results
		# Test that input vector is not modified
		#

		filt = self.constructor()

		n_samp = 8192

		# Generate frequency swept sawtooth wave
		ph = signal_generation.gen_freq_sweep_phase(10.0 / 48000., 20000. / 48000., n_samp, log=True)
		x = ph * 2.0 - 1.0
		x_copy = np.copy(x)

		yv = filt.process_vector(x)

		# Test exact equality, not just approx_equal
		if not np.array_equal(x, x_copy):
			raise UnitTestFailure('Filter must not change input values!')

		filt.reset()
		y = filt.process_vector(x)

		if not approx_equal_vector(yv, y):
			self._plot_equal_failure(yv, y, 'Newly constructed', 'after reset')
			raise UnitTestFailure('Filter gave different results after reset!')

		#
		# Test that process_sample and process_vector give same results
		#

		filt.reset()

		y = np.zeros_like(x)
		for n, xx in enumerate(x):
			y[n] = filt.process_sample(xx)

		if not approx_equal_vector(yv, y):
			self._plot_equal_failure(yv, y, 'process_vector', 'process_sample')
			raise UnitTestFailure('Filter process_sample and process_vector gave different results!')

		#
		# Test several process_vector in a row work
		#

		filt.reset()

		# Isn't it redundant to test over 8 repeats instead of just 2? Maybe. But we're testing a frequency sweep and
		# behavior may be different at different points in the spectrum (e.g. there is a discontinuity between
		# process_vector calls, but at certain freqs it's so small as to be below the approx_equal threshold), so
		# testing this ensures we hit more points in the spectrum. Could also be affected by if there happens to be a
		# phase discontinuity right near the transition point, so more tests ensures we're more likely to hit that.
		step = n_samp // 8

		y = filt.process_vector(x[0:step])
		for n in range(1, 8):
			this_x = x[n * step:(n+1) * step]
			y = np.concatenate((y, filt.process_vector(this_x)))

		if not approx_equal_vector(yv, y):
			self._plot_equal_failure(yv, y, 'expected', 'multiple process calls')
			raise UnitTestFailure('process_vector on full array gave different results from broken up into multiple process_vector calls')

		#
		# Final stress test
		# Mix of:
		# * process_sample
		# * process_vector on large aligned vector
		# * process_vector on weird size vector
		# * process_vector on size-1 vector
		#

		# TODO: regular sized vector but with alignment off

		# TODO: could combine this with steps above

		filt.reset()

		y = np.zeros_like(x)
		samples_processed = np.zeros_like(x, dtype=bool)  # For testing unit test logic

		# Same as above - this is technically redundant, but that's ok because we're at various points in spectrum

		assert n_samp % 2048 == 0  # test unit test logic itself
		for base in range(0, n_samp, 2048):

			def _test_process(first, last, vector=True):
				first += base
				last += base

				assert not any(samples_processed[first:last + 1])  # test unit test logic itself
				samples_processed[first:last + 1] = True

				if vector:
					y[first:last + 1] = filt.process_vector(x[first:last + 1])
				else:
					for n in range(first, last + 1):
						assert first <= n <= last  # test unit test logic itself
						y[n] = filt.process_sample(x[n])

			_test_process(0, 511, vector=True)  # size 512
			_test_process(512, 1023, vector=False)  # size 512
			_test_process(1024, 1055, vector=True)  # size 32

			# size 67
			assert (1122 - 1056) + 1 == 67  # test unit test logic itself
			_test_process(1056, 1122, vector=True)

			_test_process(1123, 1279, vector=True)

			# 1280-1535: process_vector on single sample
			for n in range(base+1280, base+1536):
				samples_processed[n] = True
				y[n] = filt.process_vector(np.array([x[n]]))

			_test_process(1536, 2047, vector=True)

		assert all(samples_processed)  # test unit test logic itself

		if not approx_equal_vector(yv, y):
			def _test_equal(first_samp, last_samp):
				return approx_equal_vector(yv[first_samp:last_samp+1], y[first_samp:last_samp+1])

			if not _test_equal(0, 1023):
				self._plot_equal_failure(yv[0:1024], y[0:1024], 'expected', 'process_vector to process_sample')
				raise UnitTestFailure('Switching from process_vector to process_sample failed!')

			elif not _test_equal(1024, 1055):
				self._plot_equal_failure(yv[512:1056], y[512:1056], 'expected', 'process_vector to process_sample')
				raise UnitTestFailure('Switching from process_sample to process_vector failed!')

			elif _test_equal(1056, 1279) and not _test_equal(1280, 1535):
				self._plot_equal_failure(yv[1056:1536], y[1056:1536], 'expected', 'process_vector to process_sample')
				raise UnitTestFailure('Running process_vector on single samples failed!')

			else:
				self._plot_equal_failure(yv, y, 'expected', 'stress test')
				raise UnitTestFailure('Final stress test failed!')
