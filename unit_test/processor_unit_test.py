#!/usr/bin/env python3

from utils import utils
from analysis import freq_response
from utils.approx_equal import *

from .unit_test import UnitTestFailure
from . import unit_test

from generation import signal_generation

from typing import Callable, Iterable, Tuple, Optional
import numpy as np
import math
from matplotlib import pyplot as plt


class ProcessorUnitTest:
	def __init__(
			self,
			name,
			constructor: Callable,
			freqs_to_test: Optional[Iterable[float]]=None,
			expected_freq_response_range_dB: Optional[Iterable[Tuple[Optional[float], Optional[float]]]]=None,
			expected_phase_response_range_degrees: Optional[Iterable[Optional[Tuple[float, float]]]]=None,
			deterministic: bool=True,
			linear: bool=True,
			amplitude: float=1.0,
			sample_rate=None):
		"""

		:param name: name for logging
		:param constructor:
		:param freqs_to_test: frequencies to test
		:param expected_freq_response_range_dB: expected response at frequencies, in dB - min and max, or None to leave one side open
		:param deterministic: True if processor is deterministic (aside from small floating-point and/or quantization error)
		:param linear: True if processor is linear (aside from small floating-point and/or quantization error)
		:param amplitude: amplitude to run unit test at (in case processor is nonlinear)
		"""

		self.constructor = constructor
		self.name = name

		self.freqs_to_test = freqs_to_test
		self.expected_freq_response_range_dB = expected_freq_response_range_dB
		self.expected_phase_response_degrees = expected_phase_response_range_degrees
		self.deterministic = deterministic
		self.linear = linear
		self.amplitude = amplitude
		self.sample_rate = sample_rate

	def __call__(self, *args, **kwargs) -> None:
		"""
		:raises: UnitTestFailure if test failed
		"""

		plot_on_failure = False
		if 'plot_on_failure' in kwargs:
			plot_on_failure = kwargs[plot_on_failure]

		#plot_on_failure=True

		# Test sanity first - if it doesn't work, remaining tests will be invalid
		self._test_sanity(plot_on_failure=plot_on_failure)

		if self.freqs_to_test is not None:
			self._test_freq_response(plot_on_failure=plot_on_failure)

	@staticmethod
	def _test_resp(actual, expected_range, phase=False) -> bool:
		if expected_range is None:
			return True
		# TODO: use modular arithmetic if phase
		above_min = (actual >= expected_range[0]) if expected_range[0] is not None else True
		below_max = (actual <= expected_range[1]) if expected_range[1] is not None else True
		return above_min and below_max

	def _plot_mag_failure_text(self, resp):

		sample_rate = self.sample_rate if self.sample_rate else 48000

		def _fmt_thresh(val: float) -> str:
			return '[%+8.2f]' % val

		def _fmt_val(val: Optional[float]) -> str:
			if val is None:
				return ' %8s ' % ''
			else:
				return ' %+8.2f ' % val

		def _fmt_freq(val: float) -> str:
			return ' %8.5f ' % val

		def _fmt_actual_freq(val: float) -> str:
			return ' %8.2f ' % (val * sample_rate)

		vals_above = []
		vals_in_range = []
		vals_below = []
		for val, (lower_bound, upper_bound) in zip(resp, self.expected_freq_response_range_dB):
			vals_above.append(val if val > upper_bound else None)
			vals_in_range.append(val if (lower_bound <= val <= upper_bound) else None)
			vals_below.append(val if val < lower_bound else None)

		above_thresh_line = '              ' + ' '.join([_fmt_val(val) for val in vals_above])
		upper_thresh_line = 'Upper thresh: ' + ' '.join([_fmt_thresh(val[1]) for val in self.expected_freq_response_range_dB])
		good_line         = '              ' + ' '.join([_fmt_val(val) for val in vals_in_range])
		lower_thresh_line = 'Lower thresh: ' + ' '.join([_fmt_thresh(val[0]) for val in self.expected_freq_response_range_dB])
		below_thresh_line = '              ' + ' '.join([_fmt_val(val) for val in vals_below])
		freq_line         = 'Freq:         ' + ' '.join([_fmt_freq(val) for val in self.freqs_to_test])
		actual_freq_line  = ('Freq @ %-7s' % '%gk:' % (sample_rate / 1000)) + ' '.join([_fmt_actual_freq(val) for val in self.freqs_to_test])

		print('Test failed:')
		print(self.name)
		print(above_thresh_line)
		print(upper_thresh_line)
		print(good_line)
		print(lower_thresh_line)
		print(below_thresh_line)
		print(freq_line)
		print(actual_freq_line)
		print('')

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

	def _plot_phase_failure(self, resp):
		pass  # TODO

	def _plot_equal_failure(self, x1, x2, x1_label=None, x2_label=None):
		# TODO: text-based option
		unit_test.plot_equal_failure(x1, x2, expected_label=x1_label, actual_label=x2_label, title=self.name)

	def _test_freq_response(self, plot_on_failure=False) -> None:
		"""Test frequency response, and linearity if linear

		:raises: AssertionError if test failed
		"""

		# Also includes testing linearity

		filt = self.constructor()

		check_phase = self.expected_phase_response_degrees is not None

		# Test linearity first, before freq response, because if filter is nonlinear then freq response is invalid
		if self.linear:
			if not freq_response.check_linear(filt, n_samp=8192, amplitude=10.0):
				raise UnitTestFailure('Test failed: %s is nonlinear' % self.name)

		filt.reset()

		freq_resp_ret = freq_response.get_discrete_sine_sweep_freq_response(
			filt,
			self.freqs_to_test,
			sample_rate=1.0,
			mag=True,
			phase=check_phase,
			rms=self.linear,
			group_delay=False,
			amplitude=self.amplitude)

		freq_resp_mag = utils.to_dB(freq_resp_ret.mag)

		def test_freq_resp():
			for f, actual, expected in zip(self.freqs_to_test, freq_resp_mag, self.expected_freq_response_range_dB):
				if not self._test_resp(actual, expected):

					if plot_on_failure:
						self._plot_mag_failure(freq_resp_mag)
					else:
						self._plot_mag_failure_text(freq_resp_mag)

					raise UnitTestFailure(
						'Test failed: %s, freq %f, expected (%s, %s) dB, actual %.2f dB' %
						(self.name, f, utils.to_pretty_str(expected[0]), utils.to_pretty_str(expected[1]), actual)
					)

		def test_phase():
			freq_resp_phase_deg = np.rad2deg(freq_resp_ret.phase)
			for f, actual, expected in zip(self.freqs_to_test, freq_resp_phase_deg, self.expected_phase_response_degrees):
				if not self._test_resp(actual, expected, phase=True):
					if plot_on_failure:
						self._plot_phase_failure(freq_resp_phase_deg)
					raise UnitTestFailure(
						'Test failed: %s, freq %f, expected phase (%s, %s) degrees, actual %.1f' %
						(self.name, f, utils.to_pretty_str(expected[0]), utils.to_pretty_str(expected[1]), actual))

		test_freq_resp()

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
		# Test get_state and set_state
		#

		crazy_vector = np.array([-123.4, 0.01, 9999.0, -0.000001, -987654321.0])
		crazy_vector2 = np.array([999.9, -0.1, -9999999.0, 0.1234567, 123456789.0])

		filt.reset()
		filt.process_vector(crazy_vector)

		state = filt.get_state()
		y1 = filt.process_vector(x)

		filt.reset()
		filt.process_vector(crazy_vector2)
		filt.set_state(state)
		y2 = filt.process_vector(x)

		if not approx_equal_vector(y1, y2):
			self._plot_equal_failure(y1, y2, 'Original', 'After get & set_state')
			raise UnitTestFailure('Filter gave different results after set_state!')

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


class FilterUnitTest(ProcessorUnitTest):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def __call__(self, *args, **kwargs) -> None:
		"""
		:raises: UnitTestFailure if test failed
		"""

		# FilterBase doesn't even have much extra functionality beyond ProcessorBase
		# The only one worth testing is probably basic set_freq sanity
		self._test_set_freq_sanity()

		super().__call__(*args, **kwargs)

	def _test_set_freq_sanity(self) -> None:

		filt = self.constructor()
		try:
			filt.set_freq(0.25)
		except NotImplementedError:
			raise UnitTestFailure('Test failed: %s does not have set_freq implemented' % self.name)
		except AttributeError:
			raise UnitTestFailure('Test failed: %s does not have a set_freq function' % self.name)

		# Freqs exactly 0 and 0.5 behavior is undefined. Most will throw, but filter is allowed to accept it too

		# Test freqs close to 0 or 0.5
		eps = 0.000001
		for freq in [eps, 0.5 - eps]:
			filt.set_freq(freq)

		# Test invalid freqs
		invalid_freqs = [
			-1.0-eps, -1.0, -1.0+eps,
			-0.5-eps, -0.5, -0.5+eps,
			-eps,
			0.5+eps,
			1.0-eps, 1.0, 1.0+eps,
			1.5-eps, 1.5, 1.5+eps,
		]
		for freq in invalid_freqs:
			unit_test.test_threw(filt.set_freq, freq)
