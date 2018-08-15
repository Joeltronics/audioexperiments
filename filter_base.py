#!/usr/bin/env python3

from processor import ProcessorBase

from typing import Tuple, List
from utils import *


class FilterBase(ProcessorBase):
	def set_freq(self, wc: float):
		"""Set filter cutoff frequency

		:param wc: normalized cutoff frequency, i.e. cutoff / sample rate
		"""
		raise NotImplementedError('set_freq() to be implemented by the child class!')

	def process_sample(self, sample: float) -> float:
		raise NotImplementedError('process_sample() to be implemented by the child class!')

	def process_freq_sweep(self, x: np.ndarray, wc_start: float, wc_end: float, log: bool=True) -> np.ndarray:
		if log:
			wcs = np.logspace(math.log2(wc_start), math.log2(wc_end), len(x), base=2.0)
		else:
			wcs = np.linspace(wc_start, wc_end, len(x))

		y = np.zeros_like(x)
		for n, xx, wc in zip(range(len(x)), x, wcs):
			self.set_freq(wc)
			y[n] = self.process_sample(xx)
		return y


class CascadedFilters(FilterBase):
	def __init__(self, filters):
		self.filters = filters
	
	def reset(self):
		for f in self.filters:
			f.reset()

	def set_freq(self, wc, **kwargs):
		for f in self.filters:
			f.set_freq(wc, **kwargs)

	def process_sample(self, x):
		y = x
		for f in self.filters:
			y = f.process_sample(y)
		return y


class ParallelFilters(FilterBase):
	def __init__(self, filters):
		self.filters = filters

	def reset(self):
		for f in self.filters:
			f.reset()

	def set_freq(self, wc, **kwargs):
		for f in self.filters:
			f.set_freq(wc, **kwargs)

	def process_sample(self, x):
		return sum([f.process_sample(x) for f in self.filters])


class HigherOrderFilter(ProcessorBase):
	def __init__(self, order, a, b, verbose=False):
		self.order = order
		self.reset()
		self.set_coeffs(a, b)

	# TODO: could have much more efficient process_vector using scipy.signal.lfilter
	# would need to deal with state updates with zi and zf

	def reset(self):
		self.x = np.zeros(self.order+1)
		self.y = np.zeros(self.order)

	def set_coeffs(self, a: List[float], b: List[float]):
		if len(a) != len(b) != (self.order + 1):
			raise ValueError('Filter a & b coeff vectors must have length of (order + 1)')
		if a[0] == 0.0:
			raise ValueError('Filter a0 coeff must not be zero!')
		a0 = a[0]
		self.a = np.array(a[1:]) / a0
		self.b = np.array(b) / a0

	def process_sample(self, x):

		self.x = np.roll(self.x, 1)
		self.x[0] = x

		y = np.dot(self.b, self.x) - np.dot(self.a, self.y)
		
		self.y = np.roll(self.y, 1)
		self.y[0] = y

		return y
