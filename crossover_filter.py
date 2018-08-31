#!/usr/bin/env python3

import filter_base
import butterworth_filter
import one_pole_filters
from typing import Tuple


class CrossoverLpf(filter_base.CascadedFilters):
	"""Linkwitz-Riley lowpass filter"""

	def __init__(self, wc, order: int=2, verbose=False):
		if order % 2 != 0:
			raise ValueError('Order must be even')
		order //= 2
		if order == 1:
			super().__init__([one_pole_filters.BasicOnePole(wc, verbose=verbose) for _ in range(2)])
		else:
			super().__init__([butterworth_filter.ButterworthLowpass(wc, order, verbose=verbose) for _ in range(2)])


class CrossoverHpf(filter_base.CascadedFilters):
	"""Linkwitz-Riley highpass filter"""

	def __init__(self, wc, order: int=2, verbose=False):
		if order % 2 != 0:
			raise ValueError('Order must be even')
		order //= 2
		if order == 1:
			super().__init__([one_pole_filters.BasicOnePoleHighpass(wc, verbose=verbose) for _ in range(2)])
		else:
			super().__init__([butterworth_filter.ButterworthHighpass(wc, order, verbose=verbose) for _ in range(2)])


class _ParallelCrossover(filter_base.ParallelFilters):
	"""Crossover HPF & LPF in parallel, used for plotting & unit testing"""
	def __init__(self, *args, **kwargs):
		super().__init__([CrossoverLpf(*args, **kwargs), CrossoverHpf(*args, **kwargs)])


def make_crossover_pair(wc, order) -> Tuple[CrossoverLpf, CrossoverHpf]:
	return CrossoverLpf(wc, order), CrossoverHpf(wc, order)


def main():
	import numpy as np
	from matplotlib import pyplot as plt
	from plot_utils import plot_freq_resp

	default_cutoff = 1000.
	sample_rate = 48000.
	wc = default_cutoff / sample_rate

	common_args = dict(wc=wc)

	filter_list = [
		([CrossoverLpf, CrossoverHpf], [
			dict(order=2),
			dict(order=4),
			dict(order=6)]),
		(_ParallelCrossover, [
			dict(order=2),
			dict(order=4),
			dict(order=6)]),
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
			freq_args=['wc'],
			zoom=True, phase=True, group_delay=True)

	plt.show()


if __name__ == "__main__":
	main()
