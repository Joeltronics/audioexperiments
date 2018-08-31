#!/usr/bin/env python3

import scipy.signal

from filter_base import FilterBase, CascadedFilters, IIRFilter
from biquad_filter import BiquadFilterBase


class ButterworthLowpass(FilterBase):
	def __init__(self, wc, order=4, cascade_sos=True, verbose=False):
		self.order = order
		self.cascade_sos = cascade_sos
		self._set(wc)

	def reset(self):
		self.filt.reset()

	def set_freq(self, wc):
		self._set(wc)

	def _set(self, wc):
		super().throw_if_invalid_freq(wc)
		if self.cascade_sos:
			sos = scipy.signal.butter(self.order, wc * 2.0, btype='lowpass', analog=False, output="sos")
			self.filt = CascadedFilters([
				BiquadFilterBase(
					(sos[n, 3], sos[n, 4], sos[n, 5]),
					(sos[n, 0], sos[n, 1], sos[n, 2]),
				) for n in range(sos.shape[0])
			])
		else:
			b, a = scipy.signal.butter(self.order, wc * 2.0, btype='lowpass', analog=False, output="ba")
			self.filt = IIRFilter(a, b)

	def process_sample(self, x):
		return self.filt.process_sample(x)

	def process_vector(self, v):
		return self.filt.process_vector(v)


class ButterworthHighpass(FilterBase):
	def __init__(self, wc, order=4, cascade_sos=True, verbose=False):
		self.order = order
		self.cascade_sos = cascade_sos
		self._set(wc)

	def reset(self):
		self.filt.reset()

	def set_freq(self, wc):
		self._set(wc)

	def _set(self, wc):
		super().throw_if_invalid_freq(wc)
		if self.cascade_sos:
			sos = scipy.signal.butter(self.order, wc * 2.0, btype='highpass', output="sos")
			self.filt = CascadedFilters([
				BiquadFilterBase(
					(sos[n, 3], sos[n, 4], sos[n, 5]),
					(sos[n, 0], sos[n, 1], sos[n, 2]),
				) for n in range(sos.shape[0])
			])
		else:
			b, a = scipy.signal.butter(self.order, wc * 2.0, btype='highpass', analog=False, output="ba")
			self.filt = IIRFilter(a, b)

	def process_sample(self, x):
		return self.filt.process_sample(x)

	def process_vector(self, v):
		return self.filt.process_vector(v)


def main():
	import numpy as np
	from matplotlib import pyplot as plt
	from plot_utils import plot_freq_resp

	default_cutoff = 1000.
	sample_rate = 48000.

	wc = default_cutoff / sample_rate

	common_args = dict(wc=wc)

	filter_list = [
		(ButterworthLowpass, [
			dict(order=1),
			dict(order=2),
			dict(order=4),
			dict(order=8),
			dict(order=12),
			dict(order=12, cascade_sos=False),
		]),
		(ButterworthHighpass, [
			dict(order=1),
			dict(order=2),
			dict(order=4),
			dict(order=8),
			dict(order=12),
			dict(order=12, cascade_sos=False),
		]),
	]

	freqs = np.array([
		10., 20., 30., 50.,
		100., 200., 300., 500., 700., 800., 900., 950.,
		1000., 1050., 1100., 1200., 1300., 1500., 2000., 3000., 5000.,
		10000., 11000., 13000., 15000., 20000.])

	for filter_types, extra_args_list in filter_list:
		plot_freq_resp(
			filter_types, common_args, extra_args_list,
			freqs, sample_rate,
			zoom=True, phase=True, group_delay=True,
			freq_args=['wc'])

	plt.show()


if __name__ == "__main__":
	main()
