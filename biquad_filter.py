#!/usr/bin/env python3

from filter_base import FilterBase
from typing import Tuple
from math import pi, sin, cos, sqrt
from filter_unit_test import FilterUnitTest
import numpy as np
import overdrive


# TODO: could have much more efficient process_vector using scipy.signal.lfilter
# would need to deal with state updates with zi and zf


_unit_tests = []


class BiquadFilterBase(FilterBase):
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
		self.b0 = b0 / a0
		self.b1 = b1 / a0
		self.b2 = b2 / a0
		self.a1 = a1 / a0
		self.a2 = a2 / a0

	def process_sample(self, x):

		# Assign these to make the math readable below
		a1 = self.a1; a2 = self.a2
		b0 = self.b0; b1 = self.b1; b2 = self.b2

		x1 = self.x1; x2 = self.x2
		y1 = self.y1; y2 = self.y2

		# DF1
		y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

		# Update state
		self.x2 = self.x1; self.x1 = x
		self.y2 = self.y1; self.y1 = y

		return y


# TODO: could add much more efficient process_vector using scipy.signal.lfilter
# would need to deal with state updates with zi and zf


class BiquadLowpass(BiquadFilterBase):

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

		w0 = 2.0 * pi * wc

		alpha = sin(w0) / (2.0 * Q)

		b0 = (1.0 - cos(w0)) / 2.0
		b1 = 1.0 - cos(w0)
		b2 = (1.0 - cos(w0)) / 2.0
		a0 = 1.0 + alpha
		a1 = -2.0 * cos(w0)
		a2 = 1.0 - alpha

		return (a0, a1, a2), (b0, b1, b2)

	def set_freq(self, wc, Q=None):

		if Q is not None:
			self.Q = Q

		a, b = self._get_coeffs(wc, self.Q)
		self.set_coeffs(a, b)


_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(1 kHz @ 44.1 kHz, Q=0.5)",
	lambda: BiquadLowpass(1. / 44.1, Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-1., 0.), (-3.0, 0.0), (-3.0, 0.0), (-6.1, -5.9), (-22., -20.), (-44.0, -40.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(100 Hz @ 44.1 kHz, Q=0.5)",
	lambda: BiquadLowpass(100. / 44100., Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-1., 0.), (-3.0, 0.0), (-6.1, -5.9), (-22., -20.), (-41.0, -40.0), (-61., -60.), (-88., -80.)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(1 kHz @ 44.1 kHz, Q=0.71)",
	lambda: BiquadLowpass(1. / 44.1, Q=1.0 / sqrt(2.0)),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-1., 0.), (-3.0, 0.0), (-3.0, 0.0), (-3.1, -2.9), (-22., -20.), (-44.0, -40.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))


class BiquadHighpass(BiquadFilterBase):

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

		w0 = 2.0 * pi * wc

		alpha = sin(w0) / (2.0 * Q)

		b0 = (1.0 + cos(w0)) / 2.0
		b1 = -(1.0 + cos(w0))
		b2 = (1.0 + cos(w0)) / 2.0
		a0 = 1.0 + alpha
		a1 = -2.0 * cos(w0)
		a2 = 1.0 - alpha

		return (a0, a1, a2), (b0, b1, b2)

	def set_freq(self, wc, Q=None):

		if Q is not None:
			self.Q = Q

		a, b = self._get_coeffs(wc, self.Q)
		self.set_coeffs(a, b)


_unit_tests.append(FilterUnitTest(
	"BiquadHighpass(1 kHz @ 44.1 kHz, Q=0.5)",
	lambda: BiquadHighpass(1. / 44.1, Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-82, -80), (-62, -60), (-42, -40), (-22, -20), (-6.1, -5.9), (-3, 0), (-0.5, 0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"BiquadHighpass(100 Hz @ 44.1 kHz, Q=0.5)",
	lambda: BiquadHighpass(100. / 44100., Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-42, -40), (-22, -20), (-6.1, -5.9), (-3, 0), (-1, 0), (-1, 0), (-0.1, 0.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"BiquadHighpass(100 Hz @ 44.1 kHz, Q=0.71)",
	lambda: BiquadHighpass(100. / 44100., Q=1.0 / sqrt(2.0)),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-42, -40), (-22, -20), (-3.1, -2.9), (-2, 0), (-1, 0), (-1, 0), (-0.1, 0.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))


class BiquadBandpass(BiquadFilterBase):
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

		w0 = 2.0 * pi * wc

		alpha = sin(w0) / (2.0 * Q)

		b0 = alpha
		b1 = 0.0
		b2 = -alpha
		a0 = 1.0 + alpha
		a1 = -2.0 * cos(w0)
		a2 = 1.0 - alpha

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
	lambda: BiquadBandpass(1. / 44.1, Q=0.5),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-42, -40), (-32, -30), (-22, -20), (-12, -10), (-6.1, -5.9), (-12, -10), (-22, -20)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))


class Rossum92Biquad(BiquadLowpass):
	"""
	2-pole lowpass filter based on Dave Rossum's 1992 paper
	"Making Digital Filters Sound 'Analog'"
	"""

	def __init__(self, wc, Q=0.5, verbose=False):
		super().__init__(wc, Q=Q, verbose=verbose)

	def process_sample(self, x: float) -> float:
		# Assign these to make the math readable below
		a1 = self.a1; a2 = self.a2
		b0 = self.b0; b1 = self.b1; b2 = self.b2

		x1 = self.x1; x2 = self.x2
		y1 = self.y1; y2 = self.y2

		# DF1
		y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

		# Update state
		self.x2 = self.x1; self.x1 = overdrive.tanh(x)
		self.y2 = self.y1; self.y1 = y

		return y

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		y = np.zeros_like(vec)
		for n, x in enumerate(vec):
			y[n] = self.process_sample(x)
		return y


class OverdrivenInputBiquad(BiquadLowpass):
	"""
	2-pole lowpass filter with just the input overdriven
	"""

	def __init__(self, wc, Q=0.5, verbose=False):
		super().__init__(wc, Q=Q, verbose=verbose)

	def process_sample(self, x: float) -> float:
		return super().process_sample(overdrive.tanh(x))

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		return super().process_vector(overdrive.tanh(vec))


def _run_unit_tests():
	import unit_test
	unit_test.run_unit_tests(_unit_tests)


def plot_nonlinear(args):
	import signal_generation
	from matplotlib import pyplot as plt
	import wavfile
	import argparse
	import utils
	from utils import to_pretty_str

	freq = 220.

	freq1, freq2, freq3 = freq + 0.6, freq + 0.291, freq - 0.75

	cutoff_start = 10000.
	cutoff_end = 100.

	sample_rate = 96000
	n_samp = sample_rate * 2

	wc = cutoff_start / sample_rate

	Q = 1.0 / sqrt(2.0)  # 0.7071

	filters = [
		(BiquadLowpass, dict(wc=wc, Q=Q, gain=1.0)),
		(Rossum92Biquad, dict(wc=wc, Q=Q, gain=1.0)),
		(OverdrivenInputBiquad, dict(wc=wc, Q=Q, gain=1.0)),

		(BiquadLowpass, dict(wc=wc, Q=2.0, gain=1.0)),
		(Rossum92Biquad, dict(wc=wc, Q=2.0, gain=1.0)),
		(OverdrivenInputBiquad, dict(wc=wc, Q=2.0, gain=1.0)),

		(Rossum92Biquad, dict(wc=wc, Q=Q, gain=10.0)),
		(OverdrivenInputBiquad, dict(wc=wc, Q=Q, gain=10.0)),
	]

	saws = \
		signal_generation.gen_saw(freq1 / sample_rate, n_samp) + \
		signal_generation.gen_saw(freq2 / sample_rate, n_samp) + \
		signal_generation.gen_saw(freq3 / sample_rate, n_samp)

	saws /= 3.0

	t = signal_generation.sample_time_index(n_samp, sample_rate)

	plt.figure()

	plt.plot(t, saws, label='Input')

	data_out = np.copy(saws) if args.outfile else None

	for constructor, filt_args in filters:

		args_list = ', '.join([
			'%s=%s' % (k, to_pretty_str(v, num_decimals=3))
			for k, v in filt_args.items()])

		name = '%s(%s)' % (constructor.__name__, args_list)

		print('Processing %s' % name)

		gain = filt_args.pop('gain') if 'gain' in filt_args else 1.0

		filt = constructor(**filt_args)

		x = saws * gain
		y = filt.process_freq_sweep(
			x,
			cutoff_start / sample_rate,
			cutoff_end / sample_rate,
			log=True)
		y /= gain

		if data_out is not None:
			data_out = np.append(data_out, y)

		plt.plot(t, y, label=name)

	plt.xlabel('Time (s)')
	plt.legend()

	plt.grid()

	if args.outfile:
		print('Saving %s' % args.outfile)
		data_out = utils.normalize(data_out)
		wavfile.export_wavfile(data_out, sample_rate, args.outfile, allow_overwrite=True)

	plt.show()


def plot_freq_resp(args):
	from plot_filters import plot_filters

	one_over_sqrt2 = 1.0 / sqrt(2.0)  # 0.7071

	default_cutoff = 1000.
	sample_rate = 48000.

	filter_list = [
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

	freqs = np.array([
		10., 20., 30., 50.,
		100., 200., 300., 500., 700., 800., 900., 950.,
		1000., 1050., 1100., 1200., 1300., 1500., 2000., 3000., 5000.,
		10000., 11000., 13000., 15000., 20000.])

	for filter_types, extra_args_list in filter_list:
		plot_filters(filter_types, extra_args_list, freqs, sample_rate, default_cutoff, zoom=True, phase=True, group_delay=True)


def main():
	from matplotlib import pyplot as plt
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('outfile', nargs='?')
	parser.add_argument('-v', '--verbose', action='store_true', help='Verbose unit tests')
	parser.add_argument('--test', action='store_true', help='Run unit tests')
	args = parser.parse_args()

	if args.test:
		_run_unit_tests()
		return

	plot_freq_resp(args)
	plot_nonlinear(args)

	plt.show()


if __name__ == "__main__":
	main()
