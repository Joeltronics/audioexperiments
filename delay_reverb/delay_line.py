#!/usr/bin/env python3

import argparse
import math
from typing import Union, Optional

from matplotlib import pyplot as plt
import numpy as np
import scipy.signal

from generation.signal_generation import gen_freq_sweep_sine
from processor import ProcessorBase
from unit_test import unit_test, processor_unit_test
from utils import utils
from utils.approx_equal import approx_equal_vector

from filters import allpass  # Somehow this works despite the circular dependency?!


class DelayLine(ProcessorBase):
	"""Basic delay line; non-integer delay times are allowed, in which case linear interpolation will be used"""

	def __init__(self, delay_samples: Union[float, int], max_delay_samples: Optional[int]=None, verbose=False):
		"""
		:param delay_samples: if fractional, will use linear interpolation
		"""

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
	"""Delay line that allows non-integer delays using an allpass filter"""

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
		return self.ap.process_sample_no_state_update(y)[0]

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


def _generate_fir_kernel(fir_size: int, delay_samples: float, reverse=False, shift=0):
	# TODO: if shift, generate this in-place rather than generating then rolling

	if reverse:
		x = np.arange(fir_size - 1, -1, -1, dtype=np.float64) - delay_samples
	else:
		x = np.arange(fir_size, dtype=np.float64) - delay_samples

	if shift:
		x = np.roll(x, shift)

	# TODO: window this
	return np.sinc(x)


class FIRDelayLine(ProcessorBase):
	"""
	Delay line
	Non-integer delay times achieved using FIR interpolation (zero-phase)
	Quite inefficient!
	"""

	def __init__(
			self,
			delay_samples: Union[float, int],
			max_delay_samples: Optional[int]=None, verbose=False):
		self.write_idx = 0

		if max_delay_samples is None:
			max_delay_samples = delay_samples

		# TODO: delay line size can actually be 1 less than 2*max_delay_samples
		# but this makes the math here and everywhere else simpler

		# TODO: right now FIR size is same as delay line size - can make it smaller!
		# This also makes it asymmetric when delay time is not dead center of delay line, therefore not truly zero-phase
		# This does make the math/logic much simpler though - can just np.dot the entire arrays in self.__getitem__

		self.delay_line_size = math.ceil(2.0*max_delay_samples)
		assert isinstance(self.delay_line_size, int)
		self.max_delay_samples = self.delay_line_size - 0.5*self.delay_line_size

		self._validate_delay_time(delay_samples)
		self.delay_samples = delay_samples

		self.delay_line = np.zeros(self.delay_line_size)

		self.fir_kernel = None
		self.fir_kernel_valid = False
		self.set_delay(delay_samples)

		if verbose:
			print('FIRDelayLine(delay: %g samples, max delay: %g samples)' % (self.delay_samples, self.max_delay_samples))

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

	def set_delay(self, delay_samples: Union[float, int]):
		self._validate_delay_time(delay_samples)
		self.delay_samples = utils.maybe_make_integer(delay_samples)

	def peek_front(self):
		return self.__getitem__(self.delay_samples)

	def push_back(self, x: float):
		self.delay_line[self.write_idx] = x
		self.write_idx += 1
		self.write_idx %= self.delay_line_size

	def __getitem__(self, index: Union[int, float]):
		"""
		Get item from a certain point in the delay line
		:param index: may be int or float; if float, will use linear interpolation between samples
		"""

		self._validate_delay_time(index)
		index = utils.maybe_make_integer(index)

		if isinstance(index, int):
			idx = (self.write_idx + int(index)) % self.delay_line_size
			return self.delay_line[idx]

		elif isinstance(index, float):
			# TODO: don't regenerate kernel every time! Can just roll it in-place by 1 sample if delay time unchanged
			kernel = _generate_fir_kernel(len(self.delay_line), index - 1., reverse=True, shift=self.write_idx)
			return np.dot(self.delay_line, kernel)

		else:
			raise KeyError('Index must be int or float')

	def reset(self):
		# resetting write_idx not necessary
		self.delay_line.fill(0.)

	def get_state(self):
		return np.copy(self.delay_line), self.write_idx

	def set_state(self, state):
		self.delay_line, self.write_idx = state


def _delay_line_behavior_test(delay_len=123):
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
	# TODO: test FIRDelayLine


def test(verbose=False):
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


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--sanity', action='store_true')
	return parser


def plot(args):

	def plot_up(t, y, label, x=None, upsample_ratio=16, setup_plot=True):
		if x is not None:
			plt.plot(t, x, '.-', label='Input', zorder=2)
		plt.plot(t, y, '.-', label=label, zorder=3)
		y_up, t_up = scipy.signal.resample(y, len(y) * upsample_ratio, t=t)
		plt.plot(t_up, y_up, '-', label='Upsampled', zorder=1)
		if setup_plot:
			plt.legend()
			plt.grid()

	def plot_sanity(fir_size=31):
		#t = np.linspace(-0.5*fir_size, 0.5*fir_size, fir_size)
		t = np.arange(0, fir_size)

		#print(t)

		d = fir_size // 2 + 0.25

		k0 = _generate_fir_kernel(fir_size, math.floor(d))
		k1 = _generate_fir_kernel(fir_size, d)

		fig = plt.figure()
		fig.suptitle('Sanity checks')

		plt.subplot(2, 1, 1)
		plot_up(t, k0, label='FIR kernel, delay %i' % math.floor(d))
		plt.subplot(2, 1, 2)
		plot_up(t, k1, label='FIR kernel, delay %g' % d)

	def plot_main(n_samp=64, d=9.75):
		d_int = round(d)
		assert isinstance(d_int, int)

		x = np.zeros(n_samp)
		x[0] = 1.

		ramp = np.arange(0, n_samp, dtype=np.float64)

		t = np.arange(0, n_samp)

		dl_int = DelayLine(delay_samples=d_int, verbose=True)
		dl_lerp = DelayLine(delay_samples=d, verbose=True)
		dl_ap = FractionalAllpassDelayLine(delay_samples=d, verbose=True)
		dl_fir = FIRDelayLine(delay_samples=d)

		y_int = dl_int.process_vector(x)
		y_lerp = dl_lerp.process_vector(x)
		y_ap = dl_ap.process_vector(x)
		y_fir = dl_fir.process_vector(x)

		for dl in [dl_int, dl_lerp, dl_ap, dl_fir]:
			dl.reset()

		yr_int = dl_int.process_vector(ramp)
		yr_lerp = dl_lerp.process_vector(ramp)
		yr_ap = dl_ap.process_vector(ramp)
		yr_fir = dl_fir.process_vector(ramp)

		fig = plt.figure()
		fig.suptitle('DelayLine & FractionalAllpassDelayLine')

		plt.subplot(4, 1, 1)
		plot_up(t, y_int, 'Delay %i' % d_int, x=x)
		plt.subplot(4, 1, 2)
		plot_up(t, y_lerp, 'Delay %g (lerp)' % d, x=x)
		plt.subplot(4, 1, 3)
		plot_up(t, y_ap, 'Delay %g (allpass)' % d, x=x)
		plt.subplot(4, 1, 4)
		plot_up(t, y_fir, 'Delay %g (FIR)' % d, x=x)

		fig = plt.figure()
		fig.suptitle('DelayLine & FractionalAllpassDelayLine, ramp')

		plt.subplot(4, 1, 1)
		plt.plot(t, ramp, '.-', label='Input')
		plt.plot(t, ramp - d, '.-', label='Input - d')
		plot_up(t, yr_int, 'Delay %i' % d_int, setup_plot=False)
		plt.plot(t, np.maximum(ramp - d - yr_int, 0.), '.-', label='Error')
		plt.legend()
		plt.grid()

		plt.subplot(4, 1, 2)
		plt.plot(t, ramp, '.-', label='Input')
		plt.plot(t, ramp - d, '.-', label='Input - d')
		plot_up(t, yr_lerp, 'Delay %g (lerp)' % d, setup_plot=False)
		plt.plot(t, np.maximum(ramp - d - yr_lerp, 0.), '.-', label='Error')
		plt.legend()
		plt.grid()

		plt.subplot(4, 1, 3)
		plt.plot(t, ramp, '.-', label='Input')
		plt.plot(t, ramp - d, '.-', label='Input - d')
		plot_up(t, yr_ap, 'Delay %g (allpass)' % d, setup_plot=False)
		plt.plot(t, np.maximum(ramp - d - yr_ap, 0.), '.-', label='Error')
		plt.legend()
		plt.grid()

		plt.subplot(4, 1, 4)
		plt.plot(t, ramp, '.-', label='Input')
		plt.plot(t, ramp - d, '.-', label='Input - d')
		plot_up(t, yr_fir, 'Delay %g (FIR)' % d, setup_plot=False)
		plt.plot(t, np.maximum(ramp - d - yr_fir, 0.), '.-', label='Error')
		plt.legend()
		plt.grid()

	if args.sanity:
		plot_sanity()

	plot_main()

	plt.show()


def main(args):
	plot(args)
