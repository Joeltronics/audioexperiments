#!/usr/bin/env python3

from filter_base import FilterBase
from math import pi, tan

"""
# Linear cascade filter
# 4 TrapzOnePole cascaded in series, with feedback resonance

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


class LinearCascadeFilter(FilterBase):
	"""
	4 trapezoidal-integration one pole filters cascaded in series, with feedback resonance
	Equivalent to a ladder filter, OTA cascade filter, or most IC filters, except without nonlinearities
	Should be completey clean/linear if res < 1
	"""

	def __init__(self, wc: float, res=0.0, compensate_res=True, verbose=False):
		"""

		:param wc: Cutoff frequency
		:param res: resonance, self oscillation when >= 1.0
		:param compensate_res: compensate gain when resonance increases
		"""
		self.s = [0.0 for _ in range(4)]  # State vector
		self.gain_corr = compensate_res
		self.fb = 0.0
		self.set_freq(wc, res=res)
		if verbose:
			print('LinearCascadeFilter: wc=%f, g=%f, fb=%f' % (wc, self.g, self.fb))

	def set_freq(self, wc, res=None):
		self.g = tan(pi * wc)

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

def main():
	import numpy as np
	from matplotlib import pyplot as plt
	from plot_filters import plot_filters

	default_cutoff = 1000.
	sample_rate = 48000.

	# Actually can't test resonance > 1 as this will be unstable and no longer linear
	filter_list = [
		(LinearCascadeFilter, [
			dict(res=0.0),
			dict(res=0.25),
			dict(res=0.5),
			dict(res=0.75),
			dict(res=0.95),
		]),
	]

	freqs = np.array([
		10., 20., 30., 50.,
		100., 200., 300., 500., 700., 800., 900., 950.,
		1000., 1050., 1100., 1200., 1300., 1500., 2000., 3000., 5000.,
		10000., 11000., 13000., 15000., 20000.])

	for filter_types, extra_args_list in filter_list:
		plot_filters(filter_types, extra_args_list, freqs, sample_rate, default_cutoff, zoom=True, phase=True, group_delay=True)

	plt.show()


if __name__ == "__main__":
	main()
