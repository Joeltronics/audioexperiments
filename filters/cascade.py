#!/usr/bin/env python3

from .filter_base import FilterBase
from math import pi, tan
import numpy as np
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
		self.compensate_res = compensate_res
		self.fb = 0.0

		# Define vars in init that will be set in set_freq
		self.g = None
		self.m = None
		self.mg = None
		self.mg4 = None
		self.recipmg = None
		self.gain_corr = None

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

		self.gain_corr = 1.0 + self.fb if self.compensate_res else 1.0

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
		if True:
			# Work forwards
			xr = x - (y * r)
			y0 = m*(g*xr + s[0])
			y1 = m*(g*y0 + s[1])
			y2 = m*(g*y1 + s[2])
			#y3 = m*(g*y2 + s[3])
			y3 = y

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

	# TODO: override process_freq_sweep, see if can improve performance a bit


def calc_analog_freq_resp(f: np.ndarray, fc: float, fb: float) -> np.ndarray:
	"""
	Calculate S-domain frequency response of LinearCascadeFilter
	"""

	w = f / fc

	w2 = w ** 2
	w3 = w ** 3
	w4 = w ** 4

	denom_real = w4 - 6*w2 + 1 + fb
	denom_imag = -4*w3 + 4*w

	return 1 / np.sqrt(denom_real*denom_real + denom_imag*denom_imag)


def calc_digital_freq_resp(f: np.ndarray, fc: float, fb: float) -> np.ndarray:
	"""
	Attempt to calculate Z-domain frequency response of LinearCascadeFilter

	Doesn't work - the math is complicated and there's almost certainly a mistake in it somewhere
	"""

	# Figure out consts

	g = tan(pi * fc)

	g2 = g ** 2
	g3 = g ** 3
	g4 = g ** 4

	m = 1.0 / (1.0 + g)

	m2 = m ** 2
	m3 = m ** 3
	m4 = m ** 4

	r = fb

	# Solve for z^-1

	jwT = 1j * 2 * np.pi * f
	z1 = 1.0 / np.exp(jwT)

	# Current equations:
	"""
	y = ( m4*g4*(x - r*y) + m4*g3*s0 + m3*g2*s1 + m2*g*s2 + m*s3 ) / ( 1.0 + r*m4*g4 )
	
	y0 = m * (g * (x - r*y) + s0)
	y1 = m * (g * y0 + s1)
	y2 = m * (g * y1 + s2)
	
	s0 = 2*y0 - s0*z1
	s1 = 2*y1 - s1*z1
	s2 = 2*y2 - s2*z1
	s3 = 2*y - s3*z1
	"""
	
	# Clean up:
	"""
	y = ( m4*g4*x - m4*g4*r*y + m4*g3*s0 + m3*g2*s1 + m2*g*s2 + m*s3 ) / ( 1.0 + r*m4*g4 )
	
	y0 = m*g*x - m*g*r*y + m*s0
	y1 = m*g*y0 + m*s1
	y2 = m*g*y1 + m*s2

	s0 = 2 * y0 / (1 + z1)
	s1 = 2 * y1 / (1 + z1)
	s2 = 2 * y2 / (1 + z1)
	s3 = 2 * y / (1 + z1)
	"""
	
	# Put y equation in terms of new consts:
	"""
	y = A*x - B*y + C*s0 + D*s1 + E*s2 + F*s3
	"""
	denom = 1.0 + r * m4 * g4
	A = m4*g4 / denom
	B = m4*g4*r / denom
	C = m4*g3 / denom
	D = m3*g2 / denom
	E = m2*g / denom
	F = m / denom

	# Current equations:
	"""
	y0 = m*g*x - m*g*r*y + m*s0
	y1 = m*g*y0 + m*s1
	y2 = m*g*y1 + m*s2

	s0 = 2 * y0 / (1 + z1)
	s1 = 2 * y1 / (1 + z1)
	s2 = 2 * y2 / (1 + z1)
	s3 = 2 * y / (1 + z1)
	
	y = A*x - B*y + C*s0 + D*s1 + E*s2 + F*s3
	"""
	
	#Remove s3:
	"""	
	y = A*x - B*y + C*s0 + D*s1 + E*s2 + F * 2 * y / (1 + z1)
	"""

	# Redefine F

	F = F * 2 / (1 + z1)

	"""
	y = A*x - B*y + C*s0 + D*s1 + E*s2 + F*y
	
	y0 = m*g*x - m*g*r*y + m*s0
	y1 = m*g*y0 + m*s1
	y2 = m*g*y1 + m*s2

	s0 = 2 * y0 / (1 + z1)
	s1 = 2 * y1 / (1 + z1)
	s2 = 2 * y2 / (1 + z1)
	"""
	
	# Remove s2:
	"""
	y = A*x - B*y + C*s0 + D*s1 + E*2*y2 / (1 + z1) + F*y
	
	y2 = m*g*y1 + 2*m*y2 / (1 + z1)
	y2 - 2*m*y2 / (1 + z1) = m*g*y1
	y2 * ( 1 - 2*m / (1 + z1) ) = m*g*y1
	y2 = m*g*y1 * (1 + z1) / ( 1 - 2*m )
	"""

	E = E * 2 / (1 + z1)

	G = m * g * (1 + z1) / (1 - 2 * m)

	"""
	y = A*x - B*y + C*s0 + D*s1 + E*y2 + F*y
	
	y0 = m*g*x - m*g*r*y + m*s0
	y1 = m*g*y0 + m*s1
	y2 = G*y1
	
	s0 = 2 * y0 / (1 + z1)
	s1 = 2 * y1 / (1 + z1)
	"""

	# Remove y2:
	"""
	y = A*x - B*y + C*s0 + D*s1 + E*G*y1 + F*y
	
	y0 = m*g*x - m*g*r*y + m*s0
	y1 = m*g*y0 + m*s1
	
	s0 = 2 * y0 / (1 + z1)
	s1 = 2 * y1 / (1 + z1)	
	"""

	# Remove s1:
	"""
	y = A*x - B*y + C*s0 + D*(2 * y1 / (1 + z1)) + E*G*y1 + F*y
	  = A*x - B*y + C*s0 + D*2*y1/(1 + z1) + E*G*y1 + F*y
	
	y1 = m*g*y0 + m*2*y1 / (1 + z1)
	y1 - m*2*y1 / (1 + z1) = m*g*y0
	y1 * (1 - 2*m / (1 + z1)) = m*g*y0
	y1 = m*g*y0 * (1 + z1) / (1 - 2*m)
	y1 = G*y0
	
	y0 = m*g*x - m*g*r*y + m*s0
	s0 = 2 * y0 / (1 + z1)
	"""

	# Redefine D

	D = D * 2 / (1 + z1)

	"""
	y = A*x - B*y + C*s0 + D*y1 + E*G*y1 + F*y
	y1 = G*y0
	y0 = m*g*x - m*g*r*y + m*s0
	s0 = 2 * y0 / (1 + z1)
	"""

	# Remove y1:
	"""
	y = A*x - B*y + C*s0 + D*G*y0 + E*G*G*y0 + F*y
	y0 = m*g*x - m*g*r*y + m*s0
	s0 = 2 * y0 / (1 + z1)
	"""

	# Remove s0:
	"""
	y = A*x - B*y + C*(2 * y0 / (1 + z1)) + D*G*y0 + E*G*G*y0 + F*y
	
	y0 = m*g*x - m*g*r*y + 2*m*y0 / (1 + z1)
	y0 - 2*m*y0 / (1 + z1) = m*g*x - m*g*r*y
	y0 * (1 - 2*m / (1 + z1)) = m*g*x - m*g*r*y
	y0 = m * g * (x - r*y) * (1 + z1) / (1 - 2*m)
	y0 = G * (x - r*y)
	y0 = G*x - G*r*y
	"""

	# Redefine C

	C = C * 2 / (1 + z1)

	"""
	y = A*x - B*y + C*y0 + D*G*y0 + E*G*G*y0 + F*y
	y0 = G * (x - r*y)
	"""

	# Clean up and remove y0:
	"""
	y = A*x - B*y + C*y0 + D*G*y0 + E*G*G*y0 + F*y
	0 = A*x - B*y + C*y0 + D*G*y0 + E*G*G*y0 + F*y - y
	0 = A*x + (F - B - 1)*y + (C + D*G + E*G*G)*y0
	
	0 = A*x + (F - B - 1)*y + (C + D*G + E*G*G)*(G*x - G*r*y)
	0 = A*x + (F - B - 1)*y + (C + D*G + E*G*G)*G*x - (C + D*G + E*G*G)*G*r*y
	0 = A*x + (F - B - 1)*y + (C*G + D*G2 + E*G3)*x - (C*G + D*G2 + E*G3)*r*y
	"""

	H = C*G + D*(G**2) + E*(G**3)

	"""
	0 = A*x + (F - B - 1)*y + H*x - H*r*y
	
	H*r*y - (F - B - 1)*y = A*x + H*x
	y*(H*r - F + B + 1) = x*(A + H)
	
	y / x = (A + H) / (H*r + F + B + 1)
	"""

	h = (A + H) / (H*r + F + B + 1.0)

	return np.abs(h)


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
		Q = freq_response.get_discrete_sine_sweep_freq_response(
			filt, freqs=[1000.], sample_rate=48000., n_samp=48000, mag=True, rms=False, phase=False, group_delay=False).mag
		print('%.3f  %10.6f  %10.6f  %10.6f   %.3f' % (
			r, 1.0 / (1.0 - r), Q, Q + 1.0, (Q + 1.0)*(1.0 - r)))
	print('=' * char_width)


def plot(args):
	import numpy as np
	from matplotlib import pyplot as plt
	from utils.plot_utils import plot_freq_resp
	from math import sqrt

	plot_z = False

	fig, subplots = plt.subplots(2 if plot_z else 1, 1)
	fig.suptitle('Analog prototype frequency response')

	subplot_s = subplots[0] if plot_z else subplots
	subplot_z = subplots[1] if plot_z else None

	f = np.logspace(np.log10(20), np.log10(20000), num=200, base=10)
	for fb in [0, 1, 2, 3, 3.99]:
		a = calc_analog_freq_resp(f, fc=1000.0, fb=fb)
		a = 20*np.log10(a)
		subplot_s.semilogx(f, a, label='fb=%g' % fb)

		if plot_z:
			ad = calc_digital_freq_resp(f / 44100.0, fc=1000.0/44100.0, fb=fb)
			#ad = calc_digital_freq_resp(f / 40000.0, fc=1000.0/40000.0, fb=fb)
			ad = 20*np.log10(ad)
			subplot_z.semilogx(f, ad, label='fb=%g' % fb)

	subplot_s.set_ylim([-60, 12])
	subplot_s.set_yticks(np.arange(-60, 12 + 6, 6))

	subplot_s.grid()
	subplot_s.legend()

	if plot_z:
		subplot_z.grid()
		subplot_z.legend()

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
