#!/usr/bin/env python3


from processor import ProcessorBase
from typing import Union, Optional
import numpy as np
from utils import utils
import math

from filters import allpass  # Somehow this works despite the circular dependency?!


class DelayLine(ProcessorBase):
	def __init__(self, delay_samples: Union[float, int], max_delay_samples: Optional[int]=None, verbose=False):
		"""
		:param delay_samples: if fractional, will use linear interpolation
		"""

		if delay_samples < 1:
			raise ValueError('Minimum delay time 1 sample')

		self.write_idx = 0

		if max_delay_samples is not None:
			if delay_samples > max_delay_samples:
				raise ValueError('delay_samples must be <= max_delay_samples')
			self.max_delay_samples = max_delay_samples
		else:
			self.max_delay_samples = math.ceil(delay_samples)

		self._validate_delay_time(delay_samples)
		self.delay_samples = delay_samples

		self.delay_line = np.zeros(self.max_delay_samples)

		if verbose:
			print('DelayLine(delay: %g samples, max delay: %g samples)' % (self.delay_samples, self.max_delay_samples))

	def set_delay(self, delay_samples: Union[float, int]):
		"""
		:param delay_samples: if fractional, will use linear interpolation
		"""
		self._validate_delay_time(delay_samples)
		self.delay_samples = delay_samples

	def _validate_delay_time(self, delay_samples: Union[float, int]):
		if delay_samples > self.max_delay_samples:
			raise IndexError('Delay time out of range %g > %i' % (delay_samples, self.max_delay_samples))
		elif delay_samples < 1:
			raise IndexError('Minimum delay time is 1 sample')

	def process_sample(self, x: float):
		y = self.__getitem__(self.delay_samples)
		self.push_back(x)
		return y

	# TODO: efficient process_vector

	def peek_front(self):
		return self.__getitem__(self.delay_samples)

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
		elif index < 1:
			raise IndexError('Minimum index is 1')

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
		self.delay_line.fill(0.)

	def get_state(self):
		return np.copy(self.delay_line), self.write_idx

	def set_state(self, state):
		self.delay_line, self.write_idx = state


class FractionalAllpassDelayLine(ProcessorBase):
	def __init__(self, delay_samples: Union[float, int], max_delay_samples: Optional[int]=None, verbose=False):

		self.int_delay = math.floor(delay_samples)
		self.frac_delay = delay_samples - self.int_delay
		assert isinstance(self.int_delay, int)  # this might behave differently in Python 2 vs 3
		assert utils.approx_equal(self.frac_delay, delay_samples % 1.0)

		self.dl = DelayLine(self.int_delay, max_delay_samples=max_delay_samples)
		self.ap = allpass.FractionalDelayAllpass(self.frac_delay)

		if verbose:
			print('FractionalAllpassDelayLine(int delay: %g + %g samples, max delay: %g + 0.999... samples)' % (
				self.int_delay, self.frac_delay, self.dl.max_delay_samples + 1))

	def _validate_delay_time(self, delay_samples: Union[float, int]):
		if delay_samples > self.dl.max_delay_samples:
			raise IndexError('Delay time out of range %g > %i' % (delay_samples, self.dl.max_delay_samples))
		elif delay_samples < 1:
			raise IndexError('Minimum delay time is 1 sample')

	def set_delay(self, delay_samples: Union[float, int]):
		self._validate_delay_time(delay_samples)

		self.int_delay = math.floor(delay_samples)
		self.frac_delay = delay_samples - self.int_delay

		self.dl.set_delay(self.int_delay)
		self.ap.set_delay(self.frac_delay)

	def process_sample(self, x: float):
		y = self.dl.process_sample(x)
		y = self.ap.process_sample(y)
		return y

	# TODO: efficient process_vector

	def peek_front(self):
		y = self.dl.peek_front()
		return self.ap.process_sample(y, update_state=False)

	def push_back(self, x: float):
		# process sample without using the output, just so ap updates its state
		y = self.dl.process_sample(x)
		self.ap.process_sample(y)

	def reset(self):
		self.dl.reset()
		self.ap.reset()

	def get_state(self):
		return self.dl.get_state(), self.ap.get_state()

	def set_state(self, state):
		self.dl.set_state(state[0])
		self.ap.set_state(state[1])


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

	# TODO: test operator [] at other positions (1, and somewhere in middle)
	# TODO: test push_back
	# TODO: test FractionalAllpassDelayLine


def test(verbose=False):
	from unit_test import unit_test, processor_unit_test
	return unit_test.run_unit_tests([
		processor_unit_test.ProcessorUnitTest("DelayLine ProcessorBase behavior", lambda: DelayLine(123)),
		processor_unit_test.ProcessorUnitTest("Single sample DelayLine ProcessorBase behavior", lambda: DelayLine(1)),
		processor_unit_test.ProcessorUnitTest(
			"FractionalAllpassDelayLine ProcessorBase behavior",
			lambda: FractionalAllpassDelayLine(123)),
		processor_unit_test.ProcessorUnitTest(
			"Single sample FractionalAllpassDelayLine ProcessorBase behavior",
			lambda: FractionalAllpassDelayLine(1)),
		_delay_line_behavior_test,
	], verbose=verbose)


def plot(args):
	from matplotlib import pyplot as plt
	import scipy.signal

	n_samp = 32
	d = 9.75
	d_int = round(d)
	assert isinstance(d_int, int)
	upsample_ratio = 16

	x = np.zeros(n_samp)
	x[0] = 1.

	t = np.arange(0, n_samp)

	dl_int = DelayLine(delay_samples=d_int, verbose=True)
	dl_lerp = DelayLine(delay_samples=d, verbose=True)
	dl_ap = FractionalAllpassDelayLine(delay_samples=d, verbose=True)

	y_int = dl_int.process_vector(x)
	y_lerp = dl_lerp.process_vector(x)
	y_ap = dl_ap.process_vector(x)

	def plot_it(y, label):
		plt.plot(t, x, '.-', label='Input', zorder=2)
		plt.plot(t, y, '.-', label=label, zorder=3)
		y_up, t_up = scipy.signal.resample(y, n_samp * upsample_ratio, t=t)
		plt.plot(t_up, y_up, '-', label='Delayed, upsampled', zorder=1)
		plt.legend()
		plt.grid()

	fig = plt.figure()
	fig.suptitle('DelayLine & FractionalAllpassDelayLine')

	plt.subplot(3, 1, 1)
	plot_it(y_int, 'Delay %i' % d_int)
	plt.subplot(3, 1, 2)
	plot_it(y_lerp, 'Delay %g (lerp)' % d)
	plt.subplot(3, 1, 3)
	plot_it(y_ap, 'Delay %g (allpass)' % d)

	plt.show()


def main(args):
	plot(args)
