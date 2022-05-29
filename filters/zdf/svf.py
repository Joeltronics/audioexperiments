#!/usr/bin/env python3

"""
State-variable filters

Note: this file is old, incomplete, experimental code

Some parts are completely unfinished
Some parts don't work properly
A lot needs to be cleaned up
"""

from copy import deepcopy
from math import tanh, pi
import math
from matplotlib import pyplot as plt
import numpy as np

from generation.signal_generation import gen_saw
from solvers.iter_stats import IterStats
from utils.utils import to_dB

MAX_NUM_ITER = 20
EPS = 1e-5


# TODO: implement ResonantFilterBase
# (Need to figure out how to deal with this returning multiple outputs)


def svf_freq_to_gain(wc: float) -> float:
	return 2. * math.sin(pi * wc)


def svf_res_q_mapping(res: float) -> float:
	"""
	:param res: 0-4, self-osc at 3.6 (0.9*4)
	:return: q (lowercase)
	"""

	# Q = 0.5 # 0.5 (no res) to infinity (max res)
	# q = 1. / Q # 2.0 to 0

	adj_res = res/4.0 * 1.0/0.9
	adj_res = math.sqrt(adj_res)
	q = 2.0 - 2.0*adj_res

	if q < 0.0:
		q = 0.0

	if q > 2.0:
		q = 2.0

	#print("res=%.1f, q=%.1f" % (res, q))

	return q


# Basic SVF, Forward Euler
class BasicSvf:

	def __init__(self, wc, res=0.0):
		self.q = svf_res_q_mapping(res)
		self.f = 0.0
		self.s = [0., 0.]

		self.set_freq(wc)

	def set_freq(self, wc):
		self.f = svf_freq_to_gain(wc)

	def process_vector(self, input_sig, b_all_outs=False):

		if b_all_outs:
			y_lp, y_bp, y_hp = [np.zeros_like(input_sig) for n in range(2)]
			for n, x in enumerate(input_sig):
				y_lp[n], y_bp[n], y_hp[n] = self.process_sample(x)
			return y_lp, y_bp, y_hp
		else:
			y_lp = np.zeros_like(input_sig)
			for n, x in enumerate(input_sig):
				y_lp[n], _, _ = self.process_sample(x)
			return y_lp

	def process_sample(self, x):

		# Abbreviated names
		f = self.f
		s = self.s
		q = self.q

		y_lp = s[1] + f*s[0]
		y_hp = x - y_lp - q*s[0]
		y_bp = s[0] + f*y_hp

		s[1] = y_lp
		s[0] = y_bp

		return y_lp, y_bp, y_hp


class SvfLinear:

	def __init__(self, wc, res=0.0):

		self.q = svf_res_q_mapping(res)

		self.f = None
		self.set_freq(wc)

		self.stage1prev = 0.
		self.stage2prev = 0.

		self.s = [0., 0.]

	def set_freq(self, wc):
		self.f = svf_freq_to_gain(wc)

	def process_vector(self, input_sig):
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n], _, _ = self.process_sample(x)
		return y

	def process_sample(self, x):

		zdf = True
		#zdf = False

		# Abbreviated names
		f = self.f * 0.5  # f always shows up below as f/2 (due to trapezoidal integration)
		q = self.q
		s = self.s  # reference

		if zdf:
			y_hp = (x - (f + q)*s[0] - s[1]) / (1 + f*f + q*f)
		else:
			y_hp = x - s[1] - q*s[0]

		y_bp = f*y_hp + s[0]
		y_lp = f*y_bp + s[1]

		s[0] = f*y_hp + y_bp
		s[1] = f*y_bp + y_lp

		return y_lp, y_bp, y_hp


class SvfLinearInputMixing:

	def __init__(self, wc, res=0.0):

		self.q = svf_res_q_mapping(res)

		self.set_freq(wc)

		self.stage1prev = 0.
		self.stage2prev = 0.

		self.s = [0., 0.]

	def set_freq(self, wc):
		self.f = svf_freq_to_gain(wc)

	def process_vector(self, in_lp, in_bp=None, in_hp=None):

		n_samp = len(in_lp)

		if in_bp is None:
			in_bp = np.zeros_like(in_lp)
		else:
			assert(len(in_bp) == n_samp)

		if in_hp is None:
			in_hp = np.zeros_like(in_lp)
		else:
			assert(len(in_hp) == n_samp)

		y = np.zeros_like(in_lp)
		for n, x_lp, x_bp, x_hp in zip(range(n_samp), in_lp, in_bp, in_hp):
			y[n] = self.process_sample(x_lp, x_bp, x_hp)
		return y

	def process_sample(self, x_lp, x_bp, x_hp):

		#zdf = True
		zdf = False

		# Abbreviated names
		f = self.f * 0.5  # f always shows up below as f/2 (due to trapezoidal integration)
		q = self.q
		s = self.s  # reference

		s_prev = deepcopy(s)  # For debug outputs only

		# See http://www.cytomic.com/files/dsp/SvfInputMixing.pdf?

		"""
		Output mixing code:

		if zdf:
			y_hp = (x - (f + q)*s[0] - s[1]) / (1 + f*f + q*f)
		else:
			y_hp = x - s[1] - q*s[0]

		y_bp = f*y_hp + s[0]
		y_lp = f*y_bp + s[1]

		s[0] = f*y_hp + y_bp
		s[1] = f*y_bp + y_lp
		"""

		# In-progress stuff I don't understand anymore:
		"""
		Vlp = v1 + x_bp
		Ivccs = g * (Vlp - (Vout + Vln))
		Icap = (Vout - Vhp) + s

		y0 = f*(x_lp - y) + s[0]
		y1 =
		"""

		if abs(y) > 100.0:
			print('Error: integrator blew up')
			print('f = %f' % f)
			print('q = %f' % q)
			#print('y_hp = %f' % y_hp)
			#print('y_bp = %f' % y_bp)
			#print('y_lp = %f' % y_lp)
			print('Prev: s0 = %f, s1 = %f' % (s_prev[0], s_prev[1]))
			print('New:  s0 = %f, s1 = %f' % (s[0], s[1]))
			exit(1)

		return y


class NonlinSvf:
	"""
	Note: Unlimited resonance
	"""

	def __init__(self, wc, res=0.0, iter_stats=None, res_limit=None, fb_nonlin=False):
		# res_limit: one of None, 'tanh', 'hard'

		if not res_limit:
			self.res_limit = lambda x: x
		else:
			if res_limit == 'tanh':
				self.res_limit = lambda x: tanh(x)
			elif res_limit == 'hard':
				# Clip to 0.3, then
				clip_thresh = 0.3
				clip_mix = 0.75
				self.res_limit = lambda x: clip_mix*min(max(x, -clip_thresh), clip_thresh) + (1.0 - clip_mix)*x
			else:
				assert False, "Invalid res_limit: %s" % res_limit

		self.fb_nonlin = fb_nonlin

		if iter_stats is None:
			self.iter_stats = IterStats('tanh SVF')

		self.q = svf_res_q_mapping(res)

		self.set_freq(wc)

		self.stage1prev = 0.
		self.stage2prev = 0.

		self.s = [0., 0.]

	def set_freq(self, wc):
		self.f = 2.*math.sin(pi*wc)

	def process_vector(self, input_sig):
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n], _, _ = self.process_sample(x)
		return y

	def process_sample(self, x):

		# Abbreviated names
		f = self.f * 0.5  # f always shows up below as f/2 (due to trapezoidal integration)
		q = self.q
		s = self.s  # reference

		# First calculate linear code

		y_hp = (x - (f + q)*s[0] - s[1]) / (1. + f*f + q*f)
		y_bp = f * y_hp + s[0]
		y_lp = f * y_bp + s[1]

		# Now iterate

		for n in range(MAX_NUM_ITER):

			# TODO: exit early if close enough
			#if abs_err < EPS:
			#	success = True
			#	break

			# Bandpass is tanh'ed from Hp
			y_bp = f * tanh(y_hp) + s[0]

			# Lowpass is tanh'ed from Lp
			y_lp = f * tanh(y_bp) + s[1]

			# Highpass is only tanh'ed if resonance limited

			if self.fb_nonlin:
				res = q * tanh(y_bp)
			else:
				res = q * y_bp

			# Resonance limiting
			# (limits the max difference from y_bp, not the max res amount)

			# Like this, I think:
			#res = self.res_limit(res - y_hp) + y_hp

			# This appears to be another way:
			res += 0.07*self.res_limit(y_bp)

			y_hp = x - y_lp - res

		s[0] = f*y_hp + y_bp
		s[1] = f*y_bp + y_lp

		return y_lp, y_bp, y_hp


def do_fft(x, n_fft, window=False):
	if window:
		y = np.fft.fft(x * np.hamming(len(x)), n=n_fft)
	else:
		y = np.fft.fft(x, n=n_fft)

	f = np.fft.fftfreq(n_fft, 1.0)

	# Only take first half
	y = y[0:len(y) // 2]
	f = f[0:len(f) // 2]

	y = np.abs(y)

	return y, f


def plot_nonlin_filter(fc=0.1, f_saw=0.01, res=1.5, gain=2.0, n_samp=2048):

	# FIXME: 'SVF Nonlin, tanh res' blows up if gain is higher
	x = gen_saw(f_saw, n_samp) * gain * 0.5

	t = np.arange(n_samp)

	X, f = do_fft(x, n_fft=n_samp, window=True)
	X = to_dB(np.abs(X))

	fig, (time_plot, freq_plot) = plt.subplots(2, 1)
	fig.suptitle('f_in=%g, fc=%g, res=%g, gain=%g' % (f_saw, fc, res, gain))

	time_plot.plot(t, x, '.-')
	freq_plot.semilogx(f, X)

	filts = [
		dict(name='Basic SVF', filt=BasicSvf(fc, res)),
		dict(name='SVF Linear', filt=SvfLinear(fc, res)),
		dict(name='SVF Nonlin', filt=NonlinSvf(fc, res, res_limit=None)),
		dict(name='SVF Nonlin, res limit', filt=NonlinSvf(fc, res, res_limit='hard')),
		dict(name='SVF Nonlin, tanh res', filt=NonlinSvf(fc, res, res_limit=None, fb_nonlin=True)),
	]

	for filt in filts:

		y = filt['filt'].process_vector(x)

		Y, _ = do_fft(y, n_fft=n_samp, window=True)
		Y = to_dB(Y)

		time_plot.plot(t, y, '.-', label=filt['name'])

		freq_plot.semilogx(f, Y)

		time_plot.legend()
		time_plot.set_xlim([0, 256])
		time_plot.grid()

		freq_plot.grid()


def plot(args=None):
	#plot_impulse_response(fc=0.01, n_samp=32768)
	#freq_sweep(fc=0.01, n_samp=2048)
	plot_nonlin_filter(fc=0.1, f_saw=0.01, gain=4.0, n_samp=2048)
	plt.show()


def main(args=None):
	plot(args)
