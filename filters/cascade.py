#!/usr/bin/env python3

from .filter_base import FilterBase
from math import pi, tan
from typing import Optional


"""
# Cascade filter
# 4 TrapzOnePole cascaded in series, with feedback resonance
# This general form is equivalent to ladder filter, OTA cascade filter, or most IC filters - the only difference are the
# nonlinearities within the poles and in the feedback path

# Recursive base equations

m = 1.0 / (1.0 + g)

xr = x - (y * r)

y[0] = m*(g*xr + s[0])
y[1] = m*(g*y[0] + s[1])
y[2] = m*(g*y[1] + s[2])
y[3] = m*(g*y[2] + s[3])

# State variables (just for reference - not used in math below)
s[n] = 2.0*y[n] - s[n]

# Put them together
# (Using shorthand for powers, e.g. m4 = m ** 4)

y = y[3]
y = m*(g*m*(g*m*(g*m*(g*xr + s[0]) + s[1]) + s[2]) + s[3])
y = m4*g4*xr + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]
y = m4*g4*xr + ...
y = m4*g4*(x - y*r) + ...
y = m4*g4*x - m4*g4*y*r + ...
y + m4*g4*y*r = m4*g4*x + ...
y = ( m4*g4*x + ... ) / ( 1.0 + r*m4*g4 )
y = ( m4*g4*x + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3] ) / ( 1.0 + r*m4*g4 )

# Factored for multiply-accumulate operations:
y = ( mg*(mg*(mg*(mg*x + m*s[0]) + m*s[1]) + m*s[2]) + m*s[3] ) / ( 1.0 + r*m4*g4 )
"""


def res_to_q(res: float) -> Optional[float]:
	"""
	:param res: resonance value
	:return: Q factor, or None if res out of range [0, 1)
	"""
	if not 0.0 <= res < 1.0:
		return None
	return 1.25 / (1 - res) - 1


def q_to_res(Q: float) -> Optional[float]:
	"""
	:param Q: Q factor
	:return: res, or None if Q < 0.25
	"""
	res = 1 - 1.25 / (Q + 1)
	if res < 0.0:
		return None
	return res


class LinearCascadeFilter(FilterBase):
	"""
	4 trapezoidal-integration one pole filters cascaded in series, with feedback resonance
	Equivalent to a ladder filter, OTA cascade filter, or most IC filters, except without nonlinearities
	Should be completey clean/linear if res < 1
	"""

	def __init__(self, wc: float, res: Optional[float]=None, Q: Optional[float]=None, compensate_res=True, verbose=False):
		"""

		:param wc: Cutoff frequency
		:param res: resonance, self oscillation when >= 1.0
		:param Q: alternative to res
		:param compensate_res: compensate gain when resonance increases
		"""
		self.s = [0.0 for _ in range(4)]  # State vector
		self.gain_corr = compensate_res
		self.fb = 0.0
		self.set_freq(wc, res=res, Q=Q)
		if verbose:
			if Q is not None:
				print('LinearCascadeFilter: wc=%f -> g=%f, Q=%f -> fb=%f' % (wc, self.g, Q, self.fb))
			else:
				print('LinearCascadeFilter: wc=%f -> g=%f, fb=%f' % (wc, self.g, self.fb))

	def set_freq(self, wc, res=None, Q=None):
		super().throw_if_invalid_freq(wc)

		self.g = tan(pi * wc)

		if Q is not None:
			if res is not None:
				raise ValueError('Cannot set both res and Q')
			res = q_to_res(Q)
			if res is None:
				raise ValueError('Q out of range')

		# Resonance starts at fb=4; map this to res=1
		if res is not None:
			self.fb = res * 4.0

		# Precalculate some values to make computation more efficient
		self.m = 1.0 / (self.g + 1.0)
		self.mg = self.m * self.g
		self.mg4 = self.mg ** 4.0
		self.recipmg = 1.0 / self.mg

		self.gain_corr = 1.0 + self.fb if self.gain_corr else 1.0

	def reset(self):
		for n in range(4):
			self.s[n] = 0.0

	def process_sample(self, x):

		# Abbreviations for neater code
		g = self.g
		m = self.m
		mg = self.mg
		mg4 = self.mg4
		rmg = self.recipmg

		s = self.s
		r = self.fb

		# See comments above for math
		y = (mg * (mg * (mg * (mg*x + m*s[0]) + m*s[1]) + m*s[2]) + m*s[3]) / (1.0 + r*mg4)

		# These two methods are be the same
		# working backwards is probably slightly more efficient,
		# at least if frequency is constant
		if False:
			# Work forwards
			xr = x - (y * r)
			y0 = m*(g*xr + s[0])
			y1 = m*(g*y0 + s[1])
			y2 = m*(g*y1 + s[2])
			y3 = m*(g*y2 + s[3])

		else:
			# Work backwards
			y3 = y
			y2 = (y3 - m*s[3]) * rmg
			y1 = (y2 - m*s[2]) * rmg
			y0 = (y1 - m*s[1]) * rmg

		s[0] = 2.0*y0 - s[0]
		s[1] = 2.0*y1 - s[1]
		s[2] = 2.0*y2 - s[2]
		s[3] = 2.0*y3 - s[3]

		return y * self.gain_corr


def determine_res_q():
	# Empirically determine res-Q mapping

	from analysis import freq_response

	wc = 1000. / 48000.

	char_width = 53

	print('=' * char_width)
	print('  R      1/(1-R)       Q         Q+1      (Q+1)(1-R)')
	print('-' * char_width)
	for r in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.8, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]:
		filt = LinearCascadeFilter(wc=wc, res=r)
		# Q is equal to magnitude response at cutoff frequency (linear, not dB)
		Q = freq_response.get_sine_sweep_freq_response(
			filt, freqs=[1000.], sample_rate=48000., n_samp=48000, mag=True, rms=False, phase=False, group_delay=False).mag
		print('%.3f  %10.6f  %10.6f  %10.6f   %.3f' % (
			r, 1.0 / (1.0 - r), Q, Q + 1.0, (Q + 1.0)*(1.0 - r)))
	print('=' * char_width)


def plot(args):
	import numpy as np
	from matplotlib import pyplot as plt
	from utils.plot_utils import plot_freq_resp
	from math import sqrt

	default_cutoff = 1000.
	sample_rate = 48000.

	wc = default_cutoff / sample_rate

	common_args = dict(wc=wc)

	# Actually can't test resonance > 1 as this will be unstable and no longer linear
	filter_list = [
		(LinearCascadeFilter, [
			dict(res=0.0),
			dict(res=0.125),
			dict(res=0.25),
			dict(res=0.375),
			dict(res=0.5),
			dict(res=0.75),
			dict(res=0.95),
		], True),
		(LinearCascadeFilter, [
			dict(Q=0.25),
			dict(Q=0.5),
			dict(Q=1.0/sqrt(2.0)),
			dict(Q=1.0),
			dict(Q=4.0),
		], False),
	]

	freqs = np.array([
		10., 20., 30., 50.,
		100., 200., 300., 400.,
		500., 550., 600., 650., 700., 750., 800., 850.,
		900., 925., 950., 975., 980., 985., 990., 995.,
		1000., 1025., 1050., 1075.,
		1100., 1200., 1300., 1500., 2000., 3000., 5000.,
		10000., 11000., 13000., 15000., 20000.])

	for filter_types, extra_args_list, extra_plots in filter_list:
		plot_freq_resp(
			filter_types, common_args, extra_args_list,
			freqs, sample_rate,
			freq_args=['wc'],
			zoom=extra_plots, phase=extra_plots, group_delay=extra_plots)

	plt.show()


def main(args):
	plot(args)
