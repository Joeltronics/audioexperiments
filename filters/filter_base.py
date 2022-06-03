#!/usr/bin/env python3

"""
Base classes for filters with arbitrary coefficients

Note: does not break down filters into cascaded second-order sections!
So may not be numerically stable for higher orders.
"""

from enum import Enum, unique
import scipy.signal

import processor
from utils.utils import *


@unique
class FilterForm(Enum):
	D1 = 'Direct form I',
	D2 = 'Direct form II',
	D1t = 'Transposed direct form I ',
	D2t = 'Transposed direct form II',


class FilterBase(processor.ProcessorBase):

	def set_freq(self, wc: float):
		"""Set filter cutoff frequency

		:param wc: normalized cutoff frequency, i.e. cutoff / sample rate
		"""
		# TODO: rename this set_cutoff
		raise NotImplementedError('set_freq() to be implemented by the child class!')

	def process_sample(self, sample: float) -> float:
		raise NotImplementedError('process_sample() to be implemented by the child class!')

	def process_freq_sweep(self, x: np.ndarray, wc_start: float, wc_end: float, log: bool=True) -> np.ndarray:
		# TODO: rename this process_cutoff_sweep
		if log:
			wcs = np.logspace(math.log2(wc_start), math.log2(wc_end), len(x), base=2.0)
		else:
			wcs = np.linspace(wc_start, wc_end, len(x))

		y = np.zeros_like(x)
		for n, (xx, wc) in enumerate(zip(x, wcs)):
			self.set_freq(wc)
			y[n] = self.process_sample(xx)
		return y

	def reset(self) -> None:
		raise NotImplementedError('reset() to be implemented by the child class!')

	def get_state(self) -> Any:
		raise NotImplementedError('get_state() to be implemented by the child class!')

	def set_state(self, state: Any) -> None:
		raise NotImplementedError('set_state() to be implemented by the child class!')

	@staticmethod
	def throw_if_invalid_freq(wc):
		if wc >= 0.5:
			raise ValueError('Tried to set cutoff frequency above Nyquist')
		elif wc <= 0.0:
			raise ValueError('Tried to set negative cutoff frequency')


class ResonantFilterBase(FilterBase):	
	def set_resonance(self, resonance: float):
		"""
		Set filter resonance
		:param resonance: >= 0, self-oscillation (infinite Q) at 1. Child class may internally limit maximum value.
		"""
		raise NotImplementedError('set_resonance() to be implemented by the child class!')


class CascadedFilters(FilterBase):
	# TODO: make this inherit both CascadedProcessors and FilterBase to reduce duplication
	def __init__(self, filters):
		self.filters = filters
	
	def reset(self):
		for f in self.filters:
			f.reset()

	def get_state(self):
		return [f.get_state() for f in self.filters]

	def set_state(self, state):
		for f, s in zip(self.filters, state):
			f.set_state(s)

	def set_freq(self, wc, **kwargs):
		for f in self.filters:
			f.set_freq(wc, **kwargs)

	def process_sample(self, x):
		y = x
		for f in self.filters:
			y = f.process_sample(y)
		return y


class ParallelFilters(FilterBase):
	# TODO: make this inherit both ParallelProcessors and FilterBase to reduce duplication
	def __init__(self, filters):
		self.filters = filters

	def reset(self):
		for f in self.filters:
			f.reset()

	def get_state(self):
		return [f.get_state() for f in self.filters]

	def set_state(self, state):
		for f, s in zip(self.filters, state):
			f.set_state(s)

	def set_freq(self, wc, **kwargs):
		for f in self.filters:
			f.set_freq(wc, **kwargs)

	def process_sample(self, x):
		return sum([f.process_sample(x) for f in self.filters])

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		# TODO: compare performance between this and sum()
		y = None
		for n, p in enumerate(self.filters):
			if n == 0:
				y = p.process_vector(vec)
			else:
				y += p.process_vector(vec)
		return y


class IIRFilterBase(processor.ProcessorBase):
	"""
	A generic IIR filter for given coefficients
	Can also be used as an FIR filter if given A coeffs [1, 0...], but will be less efficient
	"""

	def __init__(self, a: List[float], b: List[float], verbose=False, form=FilterForm.D2t):
		"""
		:param a: feedback coefficients; a[0] must not be zero
		:param b: feedforward coefficients; must be same length as a
		:param verbose:
		:param form:
		"""


		# Internal data type
		self.dtype = np.float64

		# D2t by default because it can be vectorized with scipy.lfilter
		self.form = form

		self.order = None
		self.a = self.b = None
		self.set_coeffs(a, b)

		self.zx = self.zy = self.z = None
		self.reset()

	def reset(self):
		if self.form in [FilterForm.D2, FilterForm.D2t]:
			self.z = np.zeros(self.order, dtype=self.dtype)

		elif self.form == FilterForm.D1:
			self.zx = np.zeros(self.order + 1, dtype=self.dtype)
			self.zy = np.zeros(self.order, dtype=self.dtype)

		elif self.form == FilterForm.D1t:
			self.zx = np.zeros(self.order, dtype=self.dtype)
			self.zy = np.zeros(self.order, dtype=self.dtype)

		else:
			raise ValueError('Unexpected filter form %s!' % str(self.form.value))

	def get_state(self) -> Any:
		if self.form in [FilterForm.D2, FilterForm.D2t]:
			return np.copy(self.z)
		elif self.form in [FilterForm.D1, FilterForm.D1t]:
			return np.copy(self.zx), np.copy(self.zy)
		else:
			raise ValueError('Unexpected filter form %s!' % str(self.form.value))

	def set_state(self, state: Any) -> None:
		if self.form in [FilterForm.D2, FilterForm.D2t]:
			self.z = state
		elif self.form in [FilterForm.D1, FilterForm.D1t]:
			self.zx, self.zy = state
		else:
			raise ValueError('Unexpected filter form %s!' % str(self.form.value))

	def set_coeffs(self, a: List[float], b: List[float]):
		"""
		Set coefficients

		:param a: feedback coefficients; a[0] must not be zero
		:param b: feedforward coefficients; must be same length as a
		"""

		# TODO: support a and b vectors of different sizes
		# e.g. for efficient FIR filters

		if self.order is None:
			if len(a) != len(b):
				raise ValueError('Filter a & b coeff vectors must have the same length!')
			self.order = len(a) - 1
		elif len(a) != len(b) != (self.order + 1):
			raise ValueError('Filter a & b coeff vectors must have length of (order + 1)')

		a0 = a[0]

		if a0 == 0.0:
			raise ValueError('Filter a0 coeff must not be zero!')

		self.a = np.array(a, dtype=self.dtype) / a0
		self.b = np.array(b, dtype=self.dtype) / a0

	def process_sample(self, x):

		if self.form == FilterForm.D1:
			shift_in_place(self.zx, input_val=x)
			y = np.dot(self.b, self.zx) - np.dot(self.a[1:], self.zy)
			shift_in_place(self.zy, input_val=y)

		elif self.form == FilterForm.D1t:
			v = x + self.zx[0]
			shift_in_place(self.zx, dir=-1, input_val=0.0)
			self.zx -= self.a[1:]*v

			y = self.b[0]*v + self.zy[0]
			shift_in_place(self.zy, dir=-1, input_val=0.0)
			self.zy += self.b[1:]*v

		elif self.form == FilterForm.D2:
			v = x - np.dot(self.a[1:], self.z)
			y = self.b[0]*v + np.dot(self.b[1:], self.z)
			shift_in_place(self.z, input_val=v)

		elif self.form == FilterForm.D2t:
			y = self.b[0]*x + self.z[0]
			shift_in_place(self.z, dir=-1, input_val=0.0)
			self.z += self.b[1:]*x - self.a[1:]*y

		else:
			raise ValueError('Unexpected filter form %s!' % str(self.form.value))

		return y

	def process_vector(self, vec: np.ndarray):
		if self.form == FilterForm.D2t:
			y, self.z = scipy.signal.lfilter(b=self.b, a=self.a, x=vec, zi=self.z)
			assert len(self.z) == self.order

		else:
			y = np.zeros_like(vec)
			for n, x in enumerate(vec):
				y[n] = self.process_sample(x)

		return y


def plot(args):
	import numpy as np
	from utils import plot_utils

	default_cutoff = 1000.
	sample_rate = 48000.

	# 4 1-pole filters in a row
	class AllPoleFilter(IIRFilterBase):
		def __init__(self, form, wc=(default_cutoff/sample_rate)):
			a1 = -math.exp(-2.0 * math.pi * wc)
			b0 = 1.0 + a1

			b = [b0**4, 0, 0, 0, 0]
			a = [1, 4*a1, 6*(a1**2), 4.0*(a1**3), a1**4]

			super().__init__(a=a, b=b, form=form)

	# Moving average filter
	# (Actually an FIR filter, but IIRFilter can do it too)
	class AllZeroFilter(IIRFilterBase):
		def __init__(self, form):
			print('Constructing AllZeroFilter, %s' % form.value)
			super().__init__(
				a=[1.0, 0.0, 0.0, 0.0, 0.0],
				b=[0.2, 0.2, 0.2, 0.2, 0.2],
				form=form)

	# Pink noise filter (1/f)
	# https://ccrma.stanford.edu/~jos/sasp/Example_Synthesis_1_F_Noise.html
	class PinkFilter(IIRFilterBase):
		def __init__(self, form):
			super().__init__(
				a=[1, -2.494956002, 2.017265875, -0.522189400],
				b=[0.049922035, -0.095993537, 0.050612699, -0.004408786],
				form=form)

	freqs = np.logspace(np.log10(20.0), np.log10(20000.0), 16, base=10)

	fig, axes = plot_utils.plt.subplots(2, 2)

	# flatten list (https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python)
	axes = [item for sublist in axes for item in sublist]

	for form, subplot in zip(FilterForm, axes):
		plot_utils.plot_freq_resp(
			[AllPoleFilter, AllZeroFilter, PinkFilter],
			None,
			dict(form=form),
			freqs, sample_rate,
			zoom=False, phase=False, group_delay=False,
			axes=[subplot])

		subplot.set_title('%s' % form.value)
		subplot.legend(['4 pole', '4 zero (moving average)', 'Pink filter (4 poles + 4 zero)'])
		subplot.set_ylim([-30., 6.])

	print('Showing plots')
	plot_utils.plt.show()


def main(args):
	plot(args)
