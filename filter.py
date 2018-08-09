#!/usr/bin/env python3

from processor import Processor

import numpy as np
import scipy.signal
from math import pi, cos, tan, sin, exp, log2, log10, sqrt
import math
from typing import Tuple, List
from utils import *
from filter_unit_test import FilterUnitTest

# TODO: figure out why one pole filter response is way off, and tighten up test requirements
# I don't think it's just measurement error, because Butterworth filters are dead on
# Seems to be an error in calculating coeffs


# TODO: many of these could have much more efficient process_vector using scipy.signal.lfilter
# would need to deal with state updates with zi and zf

_unit_tests = []


class Filter(Processor):
	def set_freq(self, wc: float):
		"""Set filter cutoff frequency

		:param wc: normalized cutoff frequency, i.e. cutoff / sample rate
		"""
		raise NotImplementedError('set_freq() to be implemented by the child class!')

	def process_sample(self, sample: float) -> float:
		raise NotImplementedError('process_sample() to be implemented by the child class!')


class CascadedFilters(Filter):
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


class ParallelFilters(Filter):
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


class BasicOnePole(Filter):
	def __init__(self, wc, verbose=False):
		self.z1 = 0.0
		self.a1 = 0.0
		self.b0 = 0.0
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


_unit_tests.append(FilterUnitTest(
	"BasicOnePole(1 kHz @ 44.1 kHz)",
	lambda: BasicOnePole(1./44.1),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"BasicOnePole(100 Hz @ 44.1 kHz)",
	lambda: BasicOnePole(100./44100.),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-3.0, 0.0), (-4.0, -2.0), (-21.0, -20.0), (-48.0, -38.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


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

	def process_vector(self, vec):
		return vec - self.lpf.process_vector(vec)


_unit_tests.append(FilterUnitTest(
	"BasicOnePoleHighpass(1 kHz @ 44.1 kHz)",
	lambda: BasicOnePoleHighpass(1./44.1),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-48.0, -38.0), (-24.0, -18.0), (-4.0, -2.0), (-3.0, 0.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


class TrapzOnePole(Filter):
	
	def __init__(self, wc, verbose=False):
		self.s = 0.0
		self.g = 0.0
		self.multiplier = 0.0
		self.set_freq(wc)
		if verbose:
			print('Trapezoid filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))
	
	def reset(self):
		self.s = 0.0

	def set_freq(self, wc):
		pi_wc = pi*wc
		self.g = tan(pi_wc)
		self.multiplier = 1.0 / (self.g + 1.0)

	def process_sample(self, x):

		# y = g*(x - y) + s
		#   = g*x - g*y + s
		#   = (g*x + s) / (g + 1)
		#   = m * (g*x + s)

		y = self.multiplier * (self.g*x + self.s)
		self.s = 2.0*y - self.s
		return y


_unit_tests.append(FilterUnitTest(
	"TrapzOnePole(1 kHz @ 44.1 kHz)",
	lambda: TrapzOnePole(1./44.1),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"TrapzOnePole(100 Hz @ 44.1 kHz)",
	lambda: TrapzOnePole(100./44100.),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0), (-48.0, -38.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


class TrapzOnePoleHighpass(Filter):
	def __init__(self, wc, verbose=False):
		self.x_prev = 0.0
		self.lpf = TrapzOnePole(wc=wc, verbose=verbose)

	def set_freq(self, wc: float):
		self.lpf.set_freq(wc)

	def reset(self):
		self.lpf.reset()

	def process_sample(self, x):
		y = x - self.lpf.process_sample(0.5 * (x + self.x_prev))
		self.x_prev = x
		return y


_unit_tests.append(FilterUnitTest(
	"TrapzOnePoleHighpass(1 kHz @ 44.1 kHz)",
	lambda: TrapzOnePoleHighpass(1./44.1),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-48.0, -38.0), (-24.0, -18.0), (-4.0, -2.0), (-3.0, 0.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


class BiquadFilter(Processor):
	def __init__(self, a: Tuple[float, float, float], b: Tuple[float, float, float]):
		self.a1 = self.a2 = 0.0
		self.b0 = self.b1 = self.b2 = 0.0
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

# TODO: could add much more efficient process_vector using scipy.signal.lfilter
# would need to deal with state updates with zi and zf


class BiquadLowpass(BiquadFilter):
	
	def __init__(self, wc, Q=0.5, verbose=False):
		self.Q = Q
		a, b = self._get_coeffs(wc, self.Q)
		super().__init__(a, b)

		if verbose:
			print('Biquad lowpass filter: wc=%f, Q=%.2f, A:[1, %f, %f], B:[%f, %f, %f]' % (
				wc, self.Q,
				self.a1, self.a2,
				self.b0, self.b1, self.b2))

	@staticmethod
	def _get_coeffs(wc, Q):

		# Formulae from audio EQ cookbook
		# http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

		w0 = 2.0*pi * wc

		alpha = sin(w0)/(2.0*Q)

		b0 = (1.0 - cos(w0))/2.0
		b1 =  1.0 - cos(w0)
		b2 = (1.0 - cos(w0))/2.0
		a0 =  1.0 + alpha
		a1 = -2.0*cos(w0)
		a2 =  1.0 - alpha

		return (a0, a1, a2), (b0, b1, b2)

	def set_freq(self, wc, Q=None):

		if Q is not None:
			self.Q = Q

		a, b = self._get_coeffs(wc, self.Q)
		self.set_coeffs(a, b)


_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(1 kHz @ 44.1 kHz, Q=0.5)",
	lambda: BiquadLowpass(1./44.1, Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-1., 0.), (-3.0, 0.0), (-3.0, 0.0), (-6.1, -5.9), (-22., -20.), (-44.0, -40.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(100 Hz @ 44.1 kHz, Q=0.5)",
	lambda: BiquadLowpass(100./44100., Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-1., 0.), (-3.0, 0.0), (-6.1, -5.9), (-22., -20.), (-41.0, -40.0), (-61., -60.), (-88., -80.)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(1 kHz @ 44.1 kHz, Q=0.71)",
	lambda: BiquadLowpass(1./44.1, Q=1.0/sqrt(2.0)),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-1., 0.), (-3.0, 0.0), (-3.0, 0.0), (-3.1, -2.9), (-22., -20.), (-44.0, -40.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


class BiquadHighpass(BiquadFilter):
	
	def __init__(self, wc, Q=0.5, verbose=False):
		self.Q = Q
		a, b = self._get_coeffs(wc, self.Q)
		super().__init__(a, b)
		
		if verbose:
			print('Biquad highpass filter: wc=%f, Q=%.2f, A:[1, %f, %f], B:[%f, %f, %f]' % (
				wc, self.Q,
				self.a1, self.a2,
				self.b0, self.b1, self.b2))

	@staticmethod
	def _get_coeffs(wc, Q):
		# Formulae from audio EQ cookbook
		# http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

		w0 = 2.0*pi * wc

		alpha = sin(w0)/(2.0*Q)

		b0 =  (1.0 + cos(w0))/2.0
		b1 = -(1.0 + cos(w0))
		b2 =  (1.0 + cos(w0))/2.0
		a0 =   1.0 + alpha
		a1 =  -2.0*cos(w0)
		a2 =   1.0 - alpha

		return (a0, a1, a2), (b0, b1, b2)

	def set_freq(self, wc, Q=None):

		if Q is not None:
			self.Q = Q

		a, b = self._get_coeffs(wc, self.Q)
		self.set_coeffs(a, b)


_unit_tests.append(FilterUnitTest(
	"BiquadHighpass(1 kHz @ 44.1 kHz, Q=0.5)",
	lambda: BiquadHighpass(1./44.1, Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-82, -80), (-62, -60), (-42, -40), (-22, -20), (-6.1, -5.9), (-3, 0), (-0.5, 0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


_unit_tests.append(FilterUnitTest(
	"BiquadHighpass(100 Hz @ 44.1 kHz, Q=0.5)",
	lambda: BiquadHighpass(100./44100., Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-42, -40), (-22, -20), (-6.1, -5.9), (-3, 0), (-1, 0), (-1, 0), (-0.1, 0.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


_unit_tests.append(FilterUnitTest(
	"BiquadHighpass(100 Hz @ 44.1 kHz, Q=0.71)",
	lambda: BiquadHighpass(100./44100., Q=1.0/sqrt(2.0)),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-42, -40), (-22, -20), (-3.1, -2.9), (-2, 0), (-1, 0), (-1, 0), (-0.1, 0.0)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))


class BiquadBandpass(BiquadFilter):
	def __init__(self, wc, Q=0.5, peak_0dB=False, verbose=False):
		self.Q = Q
		self.peak_0dB = peak_0dB
		a, b = self._get_coeffs(wc, self.Q, self.peak_0dB)
		super().__init__(a, b)

		if verbose:
			print('Biquad bandpass filter: wc=%f, Q=%.2f, A:[1, %f, %f], B:[%f, 0, %f]' % (
				wc, self.Q,
				self.a1, self.a2,
				self.b0, self.b2))

	@staticmethod
	def _get_coeffs(wc, Q, peak_0dB):

		# Formulae from audio EQ cookbook
		# http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt

		w0 = 2.0*pi * wc

		alpha = sin(w0)/(2.0*Q)

		b0 =  alpha
		b1 =  0.0
		b2 = -alpha
		a0 =  1.0 + alpha
		a1 = -2.0*cos(w0)
		a2 =  1.0 - alpha

		if not peak_0dB:
			b0 *= Q
			b2 *= Q

		return (a0, a1, a2), (b0, b1, b2)

	def set_freq(self, wc, Q=None):

		if Q is not None:
			self.Q = Q

		a, b = self._get_coeffs(wc, self.Q, self.peak_0dB)
		self.set_coeffs(a, b)


_unit_tests.append(FilterUnitTest(
	"BiquadBandpass(1 kHz @ 44.1 kHz, Q=0.5)",
	lambda: BiquadBandpass(1./44.1, Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-42, -40), (-32, -30), (-22, -20), (-12, -10), (-6.1, -5.9), (-12, -10), (-22, -20)],
	expected_phase_response_range_degrees=None, #[(), (), (), None],
	deterministic=True,
	linear=True
))



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
			b, a = scipy.signal.butter(self.order, wc*2.0, btype='lowpass', analog=False, output="ba")
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
			b, a = scipy.signal.butter(self.order, wc*2.0, btype='highpass', analog=False, output="ba")
			self.filt = HigherOrderFilter(self.order, a, b)

	def process_sample(self, x):
		return self.filt.process_sample(x)

	def process_vector(self, v):
		return self.filt.process_vector(v)


class LeakyIntegrator(Filter):
	def __init__(self, wc, w_norm=None, verbose=False):
		"""

		:param wc: normalized cutoff frequency (cutoff / sample rate)
		:param w_norm: frequency at which gain will be normalized to 0 dB; only accurate if w_norm >> wc
		:param verbose:
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
		self.z1 = self.alpha*self.z1 + x*self.one_minus_alpha
		return self.gain * self.z1


class CrossoverLpf(CascadedFilters):
	"""Linkwitz-Riley lowpass filter"""

	def __init__(self, wc, order: int=2, verbose=False):
		if order % 2 != 0:
			raise ValueError('Order must be even')
		order //= 2
		if order == 1:
			super().__init__([BasicOnePole(wc, verbose=verbose) for _ in range(2)])
		else:
			super().__init__([ButterworthLowpass(wc, order, verbose=verbose) for _ in range(2)])


class CrossoverHpf(CascadedFilters):
	"""Linkwitz-Riley highpass filter"""

	def __init__(self, wc, order: int=2, verbose=False):
		if order % 2 != 0:
			raise ValueError('Order must be even')
		order //= 2
		if order == 1:
			super().__init__([BasicOnePoleHighpass(wc, verbose=verbose) for _ in range(2)])
		else:
			super().__init__([ButterworthHighpass(wc, order, verbose=verbose) for _ in range(2)])


class _ParallelCrossover(ParallelFilters):
	"""Crossover HPF & LPF in parallel, used for plotting & unit testing"""
	def __init__(self, *args, **kwargs):
		super().__init__([CrossoverLpf(*args, **kwargs), CrossoverHpf(*args, **kwargs)])


def make_crossover_pair(wc, order) -> Tuple[CrossoverLpf, CrossoverHpf]:
	return CrossoverLpf(wc, order), CrossoverHpf(wc, order)


def _run_unit_tests():
	import unit_test
	unit_test.run_unit_tests(_unit_tests)


def main():
	from matplotlib import pyplot as plt
	import argparse
	from freq_response import get_freq_response

	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true', help='Verbose unit tests')
	parser.add_argument('--test', action='store_true', help='Run unit tests')
	parser.add_argument('--basic', action='store_true')
	parser.add_argument('--biquad', action='store_true')
	parser.add_argument('--butter', action='store_true', help='Butterorth filters')
	parser.add_argument('--cross', action='store_true', help='Crossover filters')
	parser.add_argument('--int', action='store_true', help='Integrators')
	args = parser.parse_args()

	if args.test:
		_run_unit_tests()
		return

	one_over_sqrt2 = 1.0/math.sqrt(2.0)  # 0.7071

	default_cutoff = 1000.
	sample_rate = 48000.
	n_samp = int(round(sample_rate / 4.))

	filter_list_full = [
		(BasicOnePole, None),
		(BasicOnePoleHighpass, None),
		(TrapzOnePole, None),
		(TrapzOnePoleHighpass, None),
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
			dict(cutoff=100.0, f_norm=1000.0)]),
		(CrossoverLpf, [dict(order=2)]),
		(CrossoverHpf, [dict(order=2)]),
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
		(TrapzOnePoleHighpass, [
			dict(cutoff=10.0),
			dict(cutoff=100.0),
			dict(cutoff=1000.0)]),
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
			dict(order=1),
			dict(order=2),
			dict(order=4),
			dict(order=8),
			dict(order=12),
			dict(order=12, cascade_sos=False),]),
		(ButterworthHighpass, [
			dict(order=1),
			dict(order=2),
			dict(order=4),
			dict(order=8),
			dict(order=12),
			dict(order=12, cascade_sos=False),]),
	]

	filter_list_cross = [
		([CrossoverLpf, CrossoverHpf], [
			dict(order=2),
			dict(order=4),
			dict(order=6)]),
		(_ParallelCrossover, [
			dict(order=2),
			dict(order=4),
			dict(order=6)])
	]

	filter_list = []
	if args.basic:
		filter_list += filter_list_basic
	
	if args.biquad:
		filter_list += filter_list_biquad

	if args.butter:
		filter_list += filter_list_butterworth

	if args.int:
		filter_list += filter_list_integrator

	if args.cross:
		filter_list += filter_list_cross

	if not filter_list:
		filter_list = filter_list_full

	f = np.array([
		10., 20., 30., 50., 
		100., 200., 300., 500., 700., 800., 900., 950.,
		1000., 1050., 1100., 1200., 1300., 1500., 2000., 3000., 5000.,
		10000., 11000., 13000., 15000., 20000.])

	t = np.linspace(0., n_samp/sample_rate, n_samp)

	for filter_types, extra_args_list in filter_list:

		plt.figure()

		if extra_args_list is None:
			add_legend = False
			extra_args_list = [dict()]
		else:
			add_legend = True

		max_amp_seen = 0.0
		min_amp_seen = 0.0

		if not hasattr(filter_types, "__iter__"):
			filter_types = [filter_types]

		for filter_type in filter_types:

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
				amps, phases, group_delay = get_freq_response(filt, f, sample_rate, n_samp=n_samp, group_delay=True)

				amps = to_dB(amps)

				max_amp_seen = max(max_amp_seen, np.amax(amps))
				min_amp_seen = min(min_amp_seen, np.amin(amps))

				phases_deg = np.rad2deg(phases)
				phases_deg = (phases_deg + 180.) % 360 - 180.

				plt.subplot(411)
				plt.semilogx(f, amps, label=label)

				plt.subplot(412)
				plt.semilogx(f, amps, label=label)

				plt.subplot(413)
				plt.semilogx(f, phases_deg, label=label)

				plt.subplot(414)
				plt.semilogx(f, group_delay, label=label)

		name = ', '.join([type.__name__ for type in filter_types])

		plt.subplot(411)
		plt.title('%s, sample rate %.0f' % (name, sample_rate))
		plt.ylabel('Amplitude (dB)')

		max_amp = math.ceil(max_amp_seen / 6.0) * 6.0
		min_amp = math.floor(min_amp_seen / 6.0) * 6.0

		plt.yticks(np.arange(min_amp, max_amp + 6, 6))
		plt.ylim([max(min_amp, -60.0), max(max_amp, 6.0)])
		plt.grid()
		if add_legend:
			plt.legend()

		plt.subplot(412)
		plt.ylabel('Amplitude (dB)')

		max_amp = math.ceil(max_amp_seen / 3.0) * 3.0
		min_amp = math.floor(min_amp_seen / 3.0) * 3.0

		yticks = np.arange(min_amp, max_amp + 3, 3)
		plt.yticks(yticks)
		plt.ylim([max(min_amp, -6.0), min(max_amp, 6.0)])
		plt.grid()

		plt.subplot(413)
		plt.ylabel('Phase')
		plt.grid()
		plt.yticks([-180, -90, 0, 90, 180])

		plt.subplot(414)
		plt.grid()
		plt.ylabel('Group delay')

		plt.xlabel('Freq')

	plt.show()


if __name__ == "__main__":
	main()
