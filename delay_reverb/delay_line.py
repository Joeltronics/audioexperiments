#!/usr/bin/env python3


from processor import ProcessorBase
from typing import Union
import numpy as np
from utils import utils
import math


class DelayLine(ProcessorBase):
	def __init__(self, max_delay_samples: int):
		self.max_delay_samples = max_delay_samples
		self.delay_line = None
		self.write_idx = 0
		self.reset()

	def process_sample(self, x: float):
		y = self.peek_front()
		self.push_back(x)
		return y

	# TODO: efficient process_vector

	def peek_front(self):
		return self.delay_line[self.write_idx]

	def push_back(self, x: float):
		self.delay_line[self.write_idx] = x
		self.write_idx += 1
		self.write_idx %= self.max_delay_samples

	def __getitem__(self, index: Union[int, float]):
		"""
		Get item from a certain point in the delay line
		:param index: may be int or float; if float, will use linear interpolation between samples
		"""

		if index > self.max_delay_samples:
			raise IndexError('Index out of range: %g > %i' % (index, self.max_delay_samples))
		elif index <= 0:
			raise IndexError('Index must be positive, non-zero')

		if isinstance(index, int):
			idx = (self.write_idx + index) % self.max_delay_samples
			return self.delay_line[idx]

		elif isinstance(index, float):
			idx_base = int(math.floor(index))
			idx_remainder = index - idx_base
			lerp_idx = 1.0 - idx_remainder

			idx0 = (self.write_idx + idx_base + 1) % self.max_delay_samples
			idx1 = (idx0 + 1) % self.max_delay_samples

			return utils.lerp((self.delay_line[idx0], self.delay_line[idx1]), lerp_idx)

		else:
			raise KeyError('Index must be int or float')

	def reset(self):
		# resetting write_idx not necessary
		self.delay_line = np.zeros(self.max_delay_samples)

	def get_state(self):
		return np.copy(self.delay_line), self.write_idx

	def set_state(self, state):
		self.delay_line, self.write_idx = state


def _delay_line_behavior_test(delay_len=123):
	from generation.signal_generation import gen_freq_sweep_sine
	from utils.approx_equal import approx_equal_vector
	from unit_test import unit_test

	sample_rate = 48000.
	n_samp = 1024

	dl = DelayLine(delay_len)

	x = gen_freq_sweep_sine(20./sample_rate, 20000./sample_rate, n_samp)
	y = dl.process_vector(x)
	y_expected = np.concatenate((np.zeros(delay_len), x[:n_samp-delay_len]))
	assert len(y) == len(y_expected)

	if not approx_equal_vector(y_expected, y):
		unit_test.plot_equal_failure(expected=y_expected, actual=y)
		raise unit_test.UnitTestFailure('Basic DelayLine behavior')

	front = dl.peek_front()

	unit_test.test_approx_equal(front, x[-delay_len])
	unit_test.test_approx_equal(front, dl[delay_len])
	unit_test.test_approx_equal(front, float(dl[delay_len]))

	expected_1st = x[-delay_len]
	expected_2nd = x[-delay_len + 1]

	unit_test.test_approx_equal(0.5*(expected_1st + expected_2nd), dl[delay_len - 0.5])
	unit_test.test_approx_equal(0.75*expected_1st + 0.25*expected_2nd, dl[delay_len - 0.25])


def test(verbose=False):
	from unit_test import unit_test, processor_unit_test
	return unit_test.run_unit_tests([
		processor_unit_test.ProcessorUnitTest("DelayLine ProcessorBase behavior", lambda: DelayLine(123)),
		processor_unit_test.ProcessorUnitTest("Single sample DelayLine ProcessorBase behavior", lambda: DelayLine(1)),
		_delay_line_behavior_test,
	], verbose=verbose)
