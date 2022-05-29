#!/usr/bin/env python3

import scipy.signal

from .filter_base import FilterBase, CascadedFilters, IIRFilterBase
from .biquad import BiquadFilterBase
from typing import Optional, Union, Tuple


class IIRFilter(FilterBase):
	def __init__(
			self,
			wc: float,
			order,
			btype="lowpass",
			passband_ripple_dB: Optional[float]=None,
			stopband_ripple_dB: Optional[float]=None,
			cascade_sos=True,
			verbose=False):

		if (passband_ripple_dB is not None and passband_ripple_dB < 0) or \
				(stopband_ripple_dB is not None and stopband_ripple_dB < 0):
			raise ValueError('Ripple values must be positive!')

		self.order = order
		self.btype = btype
		self.passband_ripple_dB = passband_ripple_dB
		self.stopband_ripple_dB = stopband_ripple_dB
		self.cascade_sos = cascade_sos

		# TODO:
		# In theory, any of the other 3 filter forms can be generalized as an elliptic
		# filter. I'm just not sure how scipy.signal.ellip behaves with 0 ripple values
		# Should compare resulting coeffs of ellip() to other 3 when ripple values are 0,
		# remove other 3 if results/performance are identical

		if self.passband_ripple_dB is None:
			if self.stopband_ripple_dB is None:
				self.constructor = scipy.signal.butter
			else:
				self.constructor = scipy.signal.cheby2
		else:
			if self.stopband_ripple_dB is None:
				self.constructor = scipy.signal.cheby1
			else:
				self.constructor = scipy.signal.ellip

		self.filt = None
		self._set(wc)

	def reset(self):
		self.filt.reset()

	def get_state(self):
		return self.filt.get_state()

	def set_state(self, state):
		self.filt.set_state(state)

	def set_freq(self, wc):
		self._set(wc)

	def process_sample(self, x):
		return self.filt.process_sample(x)

	def process_vector(self, v):
		return self.filt.process_vector(v)

	def _set(self, wc):
		super().throw_if_invalid_freq(wc)

		args = [self.order]

		if self.passband_ripple_dB is not None:
			args.append(self.passband_ripple_dB)

		if self.stopband_ripple_dB is not None:
			args.append(self.stopband_ripple_dB)

		args.append(wc*2.0)

		kwargs = dict(
			btype=self.btype,
			analog=False,
			output="sos" if self.cascade_sos else "ba")

		coeffs = self.constructor(*args, **kwargs)
		if self.cascade_sos:
			sos = coeffs
			self.filt = CascadedFilters([
				BiquadFilterBase(
					(sos[n, 3], sos[n, 4], sos[n, 5]),
					(sos[n, 0], sos[n, 1], sos[n, 2]),
				) for n in range(sos.shape[0])
			])
		else:
			self.filt = IIRFilterBase(coeffs[1], coeffs[0])


class ButterworthLowpass(IIRFilter):
	def __init__(self, wc: float, order=4, cascade_sos=True, verbose=False):
		super().__init__(
			wc,
			order=order,
			passband_ripple_dB=None,
			stopband_ripple_dB=None,
			cascade_sos=cascade_sos,
			verbose=verbose)

class ButterworthHighpass(IIRFilter):
	def __init__(self, wc: float, order=4, cascade_sos=True, verbose=False):
		super().__init__(
			wc,
			order=order,
			btype="highpass",
			passband_ripple_dB=None,
			stopband_ripple_dB=None,
			cascade_sos=cascade_sos,
			verbose=verbose)


def get_parser():
	import argparse
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--butterworth', action='store_true')
	parser.add_argument('--generic', action='store_true')
	return parser


def plot(args):
	import numpy as np
	from matplotlib import pyplot as plt
	from utils.plot_utils import plot_freq_resp

	if not args.butterworth and not args.generic:
		args.butterworth = args.generic = True

	default_cutoff = 1000.
	sample_rate = 48000.

	wc = default_cutoff / sample_rate

	if args.butterworth:
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

	if args.generic:
		common_args = dict(wc=wc, order=8)

		filter_list = [
			(IIRFilter, [
				dict(passband_ripple_dB=None, stopband_ripple_dB=None),
				dict(passband_ripple_dB=1, stopband_ripple_dB=None),
				dict(passband_ripple_dB=None, stopband_ripple_dB=60),
				dict(passband_ripple_dB=3, stopband_ripple_dB=40),
				dict(passband_ripple_dB=1, stopband_ripple_dB=60),
				dict(order=7, passband_ripple_dB=1, stopband_ripple_dB=60),
			]),
		]

		freqs = np.array([
			10., 20., 30., 50.,
			100., 200., 300., 500., 600., 700., 750., 800., 850., 900., 950.,
			1000., 1050., 1100., 1150., 1200., 1250., 1300., 1400., 1500., 1700.,
			2000., 2500., 3000., 4000., 5000., 6000., 7000., 8000., 9000.,
			10000., 11000., 12000., 13000., 14000., 15000., 17000., 20000.])

		for filter_types, extra_args_list in filter_list:
			plot_freq_resp(
				filter_types, common_args, extra_args_list,
				freqs, sample_rate,
				zoom=True, phase=True, group_delay=True,
				freq_args=['wc', 'wc2'],
				main_plot_ylim=[-90, 6])

	plt.show()


def main(args):
	plot(args)
