#!/usr/bin/env python3

from processor import Processor

import numpy as np
import scipy.signal
from math import pi, cos, tan, sin, exp, log2, log10
import math
from typing import Tuple, List
from utils import *


class Filter(Processor):
	def set_freq(self, wc):
		raise NotImplementedError('set_freq() to be implemented by the child class!')


class CascadedFilters(Filter):
	def __init__(self, filters):
		self.filters = filters
	
	def reset(self):
		[f.reset() for f in self.filters]

	def set_freq(self, wc, **kwargs):
		[f.set_freq(wc, **kwargs) for f in self.filters]

	def process_sample(self, x):
		y = x
		for filt in self.filters:
			y = filt.process_sample(y)
		return y
	
	def process_vector(self, vec: np.array) -> np.array:
		y = np.zeros_like(vec)
		for n, x in enumerate(vec):
			for m, filt in enumerate(self.filters):
				y[n] = filt.process_sample(x if m == 0 else y[n])
		return y


class BasicOnePole(Filter):
	def __init__(self, wc, verbose=False):
		self.z1 = 0.0
		self.set_freq(wc)
		if verbose:
			print('Basic one pole filter: wc=%f, a1=%f, b0=%f' % (wc, self.a1, self.b0))
	
	def reset(self):
		self.z1 = 0.0

	def set_freq(self, wc):
		self.a1 = exp(-2.0*pi * wc)
		self.b0 = 1.0 - self.a1
	
	def process_sample(self, x):
		self.z1 = self.b0*x + self.a1*self.z1
		y = self.z1
		return y


class BasicOnePoleHighpass(Filter):
	def __init__(self, wc, verbose=False):
		self.lpf = BasicOnePole(wc)
		if verbose:
			print('Basic one pole highpass filter - underlying LPF: wc=%f, a1=%f, b0=%f' % (wc, self.lpf.a1, self.lpf.b0))

	def reset(self):
		self.lpf.reset()

	def set_freq(self, wc):
		self.lpf.set_freq(wc)

	def process_sample(self, x):
		return x - self.lpf.process_sample(x)


class TrapzOnePole(Filter):
	
	def __init__(self, wc, verbose=False):
		self.s = 0.0
		self.set_freq(wc)
		if verbose:
			print('Trapezoid filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))
	
	def reset(self):
		self.s = 0.0

	def set_freq(self, wc):
		pi_wc = pi*wc
		self.g = tan(pi_wc)
		self.multiplier = 1.0 / (self.g + 1.0)
	
	def process_vector(self, inputSig):
		
		y = np.zeros_like(inputSig)
		
		for n, x in enumerate(inputSig):
			
			# y = g*(x - y) + s
			#   = g*x - g*y + s
			#   = (g*x + s) / (g + 1)
			#   = m * (g*x + s)
			
			yn = self.multiplier * (self.g*x + self.s)
			
			self.s = 2.0*yn - self.s
			y[n] = yn
		
		return y
	
	def process_sample(self, x):
		y = self.multiplier * (self.g*x + self.s)
		self.s = 2.0*y - self.s
		return y


# FIXME: doesn't work
"""
class TrapzOnePoleHighpass(TrapzOnePole):
	def __init__(self, wc, verbose=False):
		super().__init__(wc, verbose=verbose)
	
	def reset(self):
		super().reset()

	def process_sample(self, x):
		return x - super().process_sample(x)
"""


class BiquadFilter(Processor):
	def __init__(self, a: Tuple[float, float, float], b: Tuple[float, float, float]):
		self.set_coeffs(a, b)

		# Previous 2 inputs
		self.x1 = self.x2 = 0.0

		# Previous 2 outputs
		self.y1 = self.y2 = 0.0

	def reset(self):
		self.x1 = self.x2 = self.y1 = self.y2 = 0.0

	def set_coeffs(self, a: Tuple[float, float, float], b: Tuple[float, float, float]):
		if len(a) != len(b) != 3:
			raise ValueError('biquad a & b coeff vectors must have length 3')

		a0, a1, a2 = a
		b0, b1, b2 = b

		if a0 == 0.0:
			raise ValueError('Biquad a0 coeff must not be 0')

		# Normalize so that self.a0 == 1
		self.b0 = b0/a0
		self.b1 = b1/a0
		self.b2 = b2/a0
		self.a1 = a1/a0
		self.a2 = a2/a0

	def process_sample(self, x):
		
		# Assign these to make the math readable below
		a1 = self.a1; a2 = self.a2
		b0 = self.b0; b1 = self.b1; b2 = self.b2
		
		x1 = self.x1; x2 = self.x2
		y1 = self.y1; y2 = self.y2
		
		# DF1
		y = b0*x + b1*x1 + b2*x2 - a1*y1 - a2*y2
		
		# Update state
		self.x2 = self.x1; self.x1 = x
		self.y2 = self.y1; self.y1 = y
		
		return y


class BiquadLowpass(BiquadFilter):
	
	def __init__(self, wc, Q=0.5, verbose=False):
		self.x1 = self.x2 = 0.0
		self.y1 = self.y2 = 0.0
		self.Q = Q
		self.set_freq(wc, Q)
		
		if verbose:
			print('Biquad lowpass filter: wc=%f, Q=%.2f, A:[1, %f, %f], B:[%f, %f, %f]' % (
				wc, self.Q,
				self.a1, self.a2,
				self.b0, self.b1, self.b2))

	def set_freq(self, wc, Q=None):

		if Q is not None:
			self.Q = Q

		# Formulae from audio EQ cookbook
		# http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

		w0 = 2.0*pi * wc

		alpha = sin(w0)/(2.0*self.Q)

		b0 = (1.0 - cos(w0))/2.0
		b1 =  1.0 - cos(w0)
		b2 = (1.0 - cos(w0))/2.0
		a0 =  1.0 + alpha
		a1 = -2.0*cos(w0)
		a2 =  1.0 - alpha

		self.set_coeffs((a0, a1, a2), (b0, b1, b2))


class BiquadHighpass(BiquadFilter):
	
	def __init__(self, wc, Q=0.5, verbose=False):
		self.x1 = self.x2 = 0.0
		self.y1 = self.y2 = 0.0
		self.Q = Q
		self.set_freq(wc, Q)
		
		if verbose:
			print('Biquad highpass filter: wc=%f, Q=%.2f, A:[1, %f, %f], B:[%f, %f, %f]' % (
				wc, self.Q,
				self.a1, self.a2,
				self.b0, self.b1, self.b2))
	
	def set_freq(self, wc, Q=None):

		if Q is not None:
			self.Q = Q

		# Formulae from audio EQ cookbook
		# http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

		w0 = 2.0*pi * wc

		alpha = sin(w0)/(2.0*self.Q)

		b0 =  (1.0 + cos(w0))/2.0
		b1 = -(1.0 + cos(w0))
		b2 =  (1.0 + cos(w0))/2.0
		a0 =   1.0 + alpha
		a1 =  -2.0*cos(w0)
		a2 =   1.0 - alpha

		self.set_coeffs((a0, a1, a2), (b0, b1, b2))


class BiquadBandpass(BiquadFilter):
	def __init__(self, wc, Q=0.5, peak_0dB=False, verbose=False):
		self.x1 = self.x2 = 0.0
		self.y1 = self.y2 = 0.0
		self.Q = Q
		self.peak_0dB = peak_0dB
		self.set_freq(wc, Q)

		if verbose:
			print('Biquad highpass filter: wc=%f, Q=%.2f, A:[1, %f, %f], B:[%f, 0, %f]' % (
				wc, self.Q,
				self.a1, self.a2,
				self.b0, self.b2))

	def set_freq(self, wc, Q=None):

		if Q is not None:
			self.Q = Q

		# Formulae from audio EQ cookbook
		# http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

		w0 = 2.0*pi * wc

		alpha = sin(w0)/(2.0*self.Q)

		b0 =  alpha
		b1 =  0.0
		b2 = -alpha
		a0 =  1.0 + alpha
		a1 = -2.0*cos(w0)
		a2 =  1.0 - alpha

		if not self.peak_0dB:
			b0 *= self.Q
			b2 *= self.Q

		self.set_coeffs((a0, a1, a2), (b0, b1, b2))


class HigherOrderFilter(Processor):
	def __init__(self, order, a, b, verbose=False):
		self.order = order
		self.reset()
		self.set_coeffs(a, b)
	
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


"""
class ButterworthLowpass(HigherOrderFilter):
	def __init__(self, wc, order=4, verbose=False):
		# TODO: use "sos" format and cascaded biquads for better numerical stability
		b, a = scipy.signal.butter(order, wc*2.0, btype='lowpass', analog=False)
		super().__init__(order=order, a=a, b=b)
	
	def set_freq(self, wc):
		b, a = scipy.signal.butter(self.order, wc, btype='lowpass', analog=False)
		self.set_coeffs(a, b)


class ButterworthHighpass(HigherOrderFilter):
	def __init__(self, wc, order=4, verbose=False):
		# TODO: use "sos" format and cascaded biquads for better numerical stability
		b, a = scipy.signal.butter(order, wc*2.0, btype='highpass')
		super().__init__(order=order, a=a, b=b)
	
	def set_freq(self, wc):
		b, a = scipy.signal.butter(self.order, wc, btype='highpass')
		self.set_coeffs(a, b)
"""


class ButterworthLowpass(Filter):
	def __init__(self, wc, order=4, cascade_sos=True, verbose=False):
		self.order = order
		self.cascade_sos = cascade_sos
		self._set(wc)
	
	def reset(self):
		self.filt.reset()

	def set_freq(self, wc):
		self._set(wc)
	
	def _set(self, wc):
		if self.cascade_sos:
			sos = scipy.signal.butter(self.order, wc*2.0, btype='lowpass', analog=False, output="sos")
			self.filt = CascadedFilters([
				BiquadFilter(
					(sos[n, 3], sos[n, 4], sos[n, 5]),
					(sos[n, 0], sos[n, 1], sos[n, 2]),
				) for n in range(sos.shape[0])
			])
		else:
			b, a = scipy.signal.butter(self.order, wc*2.0, btype='lowpass', analog=False)
			self.filt = HigherOrderFilter(self.order, a, b)

	def process_sample(self, x):
		return self.filt.process_sample(x)

	def process_vector(self, v):
		return self.filt.process_vector(v)


class ButterworthHighpass(Filter):
	def __init__(self, wc, order=4, cascade_sos=True, verbose=False):
		self.order = order
		self.cascade_sos = cascade_sos
		self._set(wc)
	
	def reset(self):
		self.filt.reset()

	def set_freq(self, wc):
		self._set(wc)
	
	def _set(self, wc):
		if self.cascade_sos:
			sos = scipy.signal.butter(self.order, wc*2.0, btype='highpass', output="sos")
			self.filt = CascadedFilters([
				BiquadFilter(
					(sos[n, 3], sos[n, 4], sos[n, 5]),
					(sos[n, 0], sos[n, 1], sos[n, 2]),
				) for n in range(sos.shape[0])
			])
		else:
			b, a = scipy.signal.butter(self.order, wc*2.0, btype='highpass', analog=False)
			self.filt = HigherOrderFilter(self.order, a, b)

	def process_sample(self, x):
		return self.filt.process_sample(x)

	def process_vector(self, v):
		return self.filt.process_vector(v)


class LeakyIntegrator(Filter):
	def __init__(self, wc, w_norm=None, verbose=False):
		"""
		w_norm: frequency at which gain will be normalized to 0 dB
		Note: only accurate assuming w_norm >> wc
		"""

		self.z1 = 0.0
		self.w_norm = w_norm
		self.set_freq(wc)
		if verbose:
			print('Leaky integrator: wc=%f, alpha=%f, w_norm=%s, gain=%.2f dB' % (wc, self.alpha, str(self.w_norm), to_dB(self.gain)))
	
	def reset(self):
		self.z1 = 0.0

	def set_freq(self, wc):
		self.alpha = exp(-2.0*pi * wc)
		self.one_minus_alpha = 1.0 - self.alpha

		# Now calculate gain
		if self.w_norm is None:
			self.gain = 1.0
		else:
			# Just use Bode plot approximation (i.e. 20 dB per decade)
			decades_above_w_norm = log10(self.w_norm / wc)
			self.gain = from_dB(decades_above_w_norm * 20.0)

	def process_sample(self, x):
		# FIXME: I think I need to multiply x by 1-alpha
		self.z1 = self.alpha*self.z1 + x*self.one_minus_alpha
		return self.gain * self.z1


def main():
	import utils
	from matplotlib import pyplot as plt
	import argparse

	one_over_sqrt2 = 1.0/math.sqrt(2.0)  # 0.7071

	default_cutoff = 1000.
	sample_rate = 48000.
	n_samp = int(round(sample_rate / 4.))

	filter_list_full = [
		(BasicOnePole, None),
		(BasicOnePoleHighpass, None),
		(TrapzOnePole, None),
		#(TrapzOnePoleHighpass, None),
		(BiquadLowpass, [
			dict(Q=0.5),
			dict(Q=one_over_sqrt2),
			dict(Q=1.0),
			dict(Q=2.0)]),
		(BiquadHighpass, [
			dict(Q=0.5),
			dict(Q=one_over_sqrt2),
			dict(Q=1.0),
			dict(Q=2.0)]),
		(BiquadBandpass, [
			dict(Q=0.5),
			dict(Q=one_over_sqrt2),
			dict(Q=1.0),
			dict(Q=2.0)]),
		(BiquadBandpass, [
			dict(Q=0.5, peak_0dB=True),
			dict(Q=one_over_sqrt2, peak_0dB=True),
			dict(Q=1.0, peak_0dB=True),
			dict(Q=2.0, peak_0dB=True)]),
		(ButterworthLowpass, [dict(order=4)]),
		(ButterworthHighpass, [dict(order=4)]),
		(LeakyIntegrator, [
			dict(cutoff=10.0, f_norm=1000.0),
			dict(cutoff=100.0, f_norm=1000.0)])
	]

	filter_list_basic = [
		(BasicOnePole, [
			dict(cutoff=100.0),
			dict(cutoff=1000.0),
			dict(cutoff=10000.0)]),
		(BasicOnePoleHighpass, [
			dict(cutoff=10.0),
			dict(cutoff=100.0),
			dict(cutoff=1000.0)]),
		(TrapzOnePole, [
			dict(cutoff=100.0),
			dict(cutoff=1000.0),
			dict(cutoff=10000.0)]),
		#(TrapzOnePoleHighpass, [
		#	dict(cutoff=10.0),
		#	dict(cutoff=100.0),
		#	dict(cutoff=1000.0))]),
	]

	filter_list_biquad = [
				(BiquadLowpass, [
			dict(Q=0.5),
			dict(Q=one_over_sqrt2),
			dict(Q=1.0),
			dict(Q=2.0)]),
		(BiquadHighpass, [
			dict(Q=0.5),
			dict(Q=one_over_sqrt2),
			dict(Q=1.0),
			dict(Q=2.0)]),
		(BiquadBandpass, [
			dict(Q=0.5),
			dict(Q=one_over_sqrt2),
			dict(Q=1.0),
			dict(Q=2.0)]),
		(BiquadBandpass, [
			dict(Q=0.5, peak_0dB=True),
			dict(Q=one_over_sqrt2, peak_0dB=True),
			dict(Q=1.0, peak_0dB=True),
			dict(Q=2.0, peak_0dB=True)]),
	]

	filter_list_integrator = [
		(LeakyIntegrator, [
			dict(cutoff=10.0, f_norm=100.0),
			dict(cutoff=10.0, f_norm=1000.0),
			dict(cutoff=100.0, f_norm=1000.0),
			dict(cutoff=100.0, f_norm=10000.0),
			dict(cutoff=1000.0, f_norm=10000.0)]),
	]

	filter_list_butterworth = [
		(ButterworthLowpass, [
			dict(order=2),
			dict(order=4),
			dict(order=8),
			dict(order=12),
			dict(order=12, cascade_sos=False),]),
		(ButterworthHighpass, [
			dict(order=2),
			dict(order=4),
			dict(order=8),
			dict(order=12),
			dict(order=12, cascade_sos=False),]),
	]

	parser = argparse.ArgumentParser()
	parser.add_argument('--basic', action='store_true')
	parser.add_argument('--biquad', action='store_true')
	parser.add_argument('--butter', action='store_true')
	parser.add_argument('--int', action='store_true')
	args = parser.parse_args()

	filter_list = []
	if args.basic:
		filter_list += filter_list_basic
	
	if args.biquad:
		filter_list += filter_list_biquad

	if args.butter:
		filter_list += filter_list_butterworth

	if args.int:
		filter_list += filter_list_integrator

	if not filter_list:
		filter_list = filter_list_full

	f = np.array([
		10., 20., 30., 50., 
		100., 200., 300., 500., 700., 800., 900., 950.,
		1000., 1050., 1100., 1200., 1300., 1500., 2000., 3000., 5000.,
		10000., 11000., 13000., 15000., 20000.])

	t = np.linspace(0., n_samp/sample_rate, n_samp)

	for filter_type, extra_args_list in filter_list:

		plt.figure()

		if extra_args_list is None:
			add_legend = False
			extra_args_list = [dict()]
		else:
			add_legend = True

		max_amp_seen = 0.0
		min_amp_seen = 0.0

		for extra_args in extra_args_list:

			label = ', '.join(['%s=%s' % (key, to_pretty_str(value)) for key, value in extra_args.items()])
			if label:
				print('Processing %s, %s' % (filter_type.__name__, label))
			else:
				print('Processing %s' % (filter_type.__name__))

			if 'cutoff' in extra_args.keys():
				cutoff = extra_args['cutoff']
				extra_args.pop('cutoff')
			else:
				cutoff = default_cutoff

			if 'f_norm' in extra_args.keys():
				extra_args['w_norm'] = extra_args['f_norm'] / sample_rate
				extra_args.pop('f_norm')

			filt = filter_type(wc=(cutoff / sample_rate), verbose=True, **extra_args)
			amps, phases, group_delay = utils.get_freq_response(filt, f, sample_rate, n_samp=n_samp, throw_if_nonlinear=True, group_delay=True)
			#amps, phases, group_delay = utils.get_freq_response(filt, f, sample_rate, throw_if_nonlinear=True, group_delay=True)
			
			amps = to_dB(amps)

			max_amp_seen = max(max_amp_seen, np.amax(amps))
			min_amp_seen = min(min_amp_seen, np.amin(amps))

			phases_deg = np.rad2deg(phases)
			phases_deg = (phases_deg + 180.) % 360 - 180.
			
			plt.subplot(311)
			plt.semilogx(f, amps, label=label)

			plt.subplot(312)
			plt.semilogx(f, phases_deg, label=label)

			plt.subplot(313)
			plt.semilogx(f, group_delay, label=label)
		
		plt.subplot(311)
		plt.title('%s, sample rate %.0f' % (filter_type.__name__, sample_rate))
		plt.ylabel('Amplitude (dB)')
		
		max_amp = np.ceil(max_amp_seen / 6.0) * 6.0
		min_amp = np.floor(min_amp_seen / 6.0) * 6.0
		
		plt.yticks(np.arange(min_amp, max_amp + 6, 6))
		plt.ylim([max(min_amp, -60.0), max(max_amp, 6.0)])
		plt.grid()
		if add_legend:
			plt.legend()
		
		plt.subplot(312)
		plt.ylabel('Phase')
		plt.grid()
		plt.yticks([-180, -90, 0, 90, 180])

		plt.subplot(313)
		plt.grid()
		plt.ylabel('Group delay')

		plt.xlabel('Freq')

	plt.show()


if __name__ == "__main__":
	main()
