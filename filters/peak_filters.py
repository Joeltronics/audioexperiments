#!/usr/bin/env python3

import numpy as np
from math import exp, pi

from utils import utils
from processor import ProcessorBase


_unit_tests = []


class BidirectionalOnePoleFilter(ProcessorBase):
	def __init__(self, rise_time, fall_time, gain=1.0, verbose=False):
		"""
		:param rise_time: 63% (1-1/e) rise time, in samples. Equivalent cutoff frequency 1 / (2*pi*time)
		:param fall_time: 63% (1-1/e) fall time, in samples. Equivalent cutoff frequency 1 / (2*pi*time)
		:param gain: optional gain to add to the system
		:param verbose:
		"""

		if rise_time == 0.0 or fall_time == 0.0:
			raise ValueError('Rise & fall times must not be zero')
		elif rise_time < 0.0 or fall_time < 0.0:
			raise ValueError('Rise & fall times must be positive')

		self.z1 = 0.0

		self.na1_up = 0.0
		self.b0_up = 0.0

		self.na1_down = 0.0
		self.b0_down = 0.0

		self.gain = gain

		self._set_freqs(rise_time, fall_time, gain=gain)

		if verbose:
			print('Bidirectional one pole filter: gain %s, rise(%.1f samples, b0=%f, a1=%f), fall(%.1f samples, b0=%f, a1=%f)' % (
				utils.to_pretty_str(gain), rise_time, self.b0_up, -self.na1_up, fall_time, self.b0_down, -self.na1_down))

	def reset(self):
		self.z1 = 0.0

	def _set_freqs(self, rise_time, fall_time, gain=None):

		if rise_time == 0.0 or fall_time == 0.0:
			raise ValueError('Rise & fall times must not be zero')
		elif rise_time < 0.0 or fall_time < 0.0:
			raise ValueError('Rise & fall times must be positive')

		if gain is not None:
			self.gain = gain

		self.na1_up = exp(-1.0 / rise_time)
		self.na1_down = exp(-1.0 / fall_time)

		self.b0_up = (1.0 - self.na1_up) * self.gain
		self.b0_down = (1.0 - self.na1_down) * self.gain

		assert self.na1_down > 0.0
		assert self.b0_down > 0.0
		assert self.na1_up > 0.0
		assert self.b0_up > 0.0

	def process_sample(self, x):

		if x > self.z1:
			b0 = self.b0_up
			na1 = self.na1_up
		else:
			b0 = self.b0_down
			na1 = self.na1_down

		y = self.z1 = (b0 * x) + (na1 * self.z1)
		return y


def test(verbose=False):
	from unit_test import unit_test
	from unit_test.processor_unit_test import ProcessorUnitTest

	# If rise time = fall time, BidirectionalFilter is identical to one-pole filter

	twopi = 2.0 * np.pi

	tests = [
		ProcessorUnitTest(
			"BidirectionalOnePoleFilter(both 1 ms @ 44.1 kHz)",
			lambda: BidirectionalOnePoleFilter(44.1 / twopi, 44.1 / twopi),
			freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
			expected_freq_response_range_dB=[(-0.1, 0.0), (-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0)],
			expected_phase_response_range_degrees=None,  # [(), (), (), None],
			deterministic=True,
			linear=True
		),
		ProcessorUnitTest(
			"BidirectionalOnePoleFilter(both 1 ms @ 44.1 kHz, 3 dB gain)",
			lambda: BidirectionalOnePoleFilter(44.1 / twopi, 44.1 / twopi, gain=utils.from_dB(3.0)),
			freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
			expected_freq_response_range_dB=[(2.9, 3.0), (0.0, 3.0), (-1.0, 1.0), (-21.0, -15.0)],
			expected_phase_response_range_degrees=None,  # [(), (), (), None],
			deterministic=True,
			linear=True
		),
		ProcessorUnitTest(
			"BidirectionalOnePoleFilter(both 10 ms @ 44.1 kHz)",
			lambda: BidirectionalOnePoleFilter(44100. / (100. * twopi), 44100. / (100. * twopi)),
			freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
			expected_freq_response_range_dB=[(-3.0, 0.0), (-4.0, -2.0), (-21.0, -20.0), (-48.0, -38.0)],
			expected_phase_response_range_degrees=None,  # [(), (), (), None],
			deterministic=True,
			linear=True
		),
	]

	return unit_test.run_unit_tests(tests, verbose=verbose)


def plot(args):
	from matplotlib import pyplot as plt
	from generation import signal_generation

	sample_rate = 48000.
	freq = 100.
	n_samp = int(round(sample_rate))
	n_samp_sin = int(round(sample_rate / freq * 5))

	rise_time_ms = 1.
	fall_time_ms = 200.

	rise_time = rise_time_ms / 1000.0 * sample_rate
	fall_time = fall_time_ms / 1000.0 * sample_rate

	print('Rise time: %.1f ms = %.1f samples' % (rise_time_ms, rise_time))
	print('Fall time: %.1f ms = %.1f samples' % (fall_time_ms, fall_time))
	filter = BidirectionalOnePoleFilter(rise_time=rise_time, fall_time=fall_time, verbose=True)

	x = np.concatenate((signal_generation.gen_sine(freq / sample_rate, n_samp_sin), np.zeros(n_samp - n_samp_sin)))
	x = np.abs(x)
	y = filter.process_vector(x)

	plt.figure()

	t = signal_generation.sample_time_index(n_samp, sample_rate) * 1000.

	plt.subplot(2, 1, 1)
	plt.plot(t, x, label='Input')
	plt.plot(t, y, label='Output')
	plt.grid()
	plt.xlim([0, (n_samp_sin * 1.25) / sample_rate * 1000])
	plt.title('Bidirectional filter, rise %s ms, fall %s ms' % (
		utils.to_pretty_str(rise_time_ms), utils.to_pretty_str(fall_time_ms)))
	plt.xlabel('Time (ms)')

	plt.subplot(2, 1, 2)
	plt.plot(t, x, label='Input')
	plt.plot(t, y, label='Output')
	plt.axhline(1.0 / exp(1.0), color='red', label='fall time measurement value')
	plt.legend()
	plt.grid()
	plt.xlabel('Time (ms)')

	plt.show()


def main(args):
	plot(args)
