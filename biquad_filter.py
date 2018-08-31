#!/usr/bin/env python3

from filter_base import FilterBase, FilterForm
from typing import Tuple
from math import pi, sin, cos, sqrt
from processor_unit_test import FilterUnitTest
import numpy as np
import scipy.signal
import overdrive
import utils


_unit_tests = []


class BiquadFilterBase(FilterBase):
	def __init__(
			self,
			a: Tuple[float, float, float],
			b: Tuple[float, float, float],
			form=FilterForm.D2t):

		# Internal data type
		self.dtype = np.float64

		# D2t by default because it can be vectorized with scipy.lfilter
		self.form = form

		self.set_coeffs(a, b)
		self.reset()

	def reset(self):
		if self.form in [FilterForm.D2, FilterForm.D2t]:
			self.z = np.array([0.0, 0.0], dtype=self.dtype)
		else:
			self.zx = np.array([0.0, 0.0], dtype=self.dtype)
			self.zy = np.array([0.0, 0.0], dtype=self.dtype)

	def set_coeffs(self, a: Tuple[float, float, float], b: Tuple[float, float, float]):
		if len(a) != len(b) != 3:
			raise ValueError('biquad a & b coeff vectors must have length 3')

		if a[0] == 0.0:
			raise ZeroDivisionError('Biquad a0 coeff must not be 0')

		# Normalize so that a[0] == 1
		self.a = np.array(a, dtype=self.dtype) / a[0]
		self.b = np.array(b, dtype=self.dtype) / a[0]

		assert utils.approx_equal_scalar(self.a[0], 1.0)

	def process_sample(self, x):

		_, a1, a2 = self.a
		b0, b1, b2 = self.b

		if self.form == FilterForm.D1:

			x1, x2 = self.zx
			y1, y2 = self.zy

			y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

			# Update state
			self.zx[1] = self.zx[0]
			self.zx[0] = x

			self.zy[1] = self.zy[0]
			self.zy[0] = y

		elif self.form == FilterForm.D1t:

			x1, x2 = self.zx
			y1, y2 = self.zy

			v = x + x1
			y = b0 * v + y1

			self.zx[0] = self.zx[1] - a1*v
			self.zx[1] = -a2 * v

			self.zy[0] = self.zy[1] + b1*v
			self.zy[1] = b2 * v

		elif self.form == FilterForm.D2:

			z1, z2 = self.z

			v = x - a1 * z1 - a2 * z2
			y = b0 * v + b1 * z1 + b2 * z2

			self.z[1] = self.z[0]
			self.z[0] = v

		elif self.form == FilterForm.D2t:

			y = b0*x + self.z[0]

			self.z[0] = self.z[1] + b1 * x - a1 * y
			self.z[1] = b2 * x - a2 * y

		else:
			raise ValueError('Unexpected filter form %s!' % str(self.form.value))

		return y

	def process_vector(self, vec: np.ndarray) -> np.ndarray:

		if self.form == FilterForm.D2t:
			y, self.z = scipy.signal.lfilter(b=self.b, a=self.a, x=vec, zi=self.z)
			assert len(self.z) == 2

		else:
			y = np.zeros_like(vec)
			for n, x in enumerate(vec):
				y[n] = self.process_sample(x)

		return y


class BiquadLowpass(BiquadFilterBase):

	def __init__(self, wc, Q=0.5, verbose=False, form=FilterForm.D2t):
		self.Q = Q
		a, b = self._get_coeffs(wc, self.Q)
		super().__init__(a, b, form=form)

		if verbose:
			print('Biquad lowpass filter: wc=%f, Q=%.2f, A:[1, %f, %f], B:[%f, %f, %f]' % (
				wc, self.Q,
				self.a[1], self.a[2],
				self.b[0], self.b[1], self.b[2]))

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
		super().throw_if_invalid_freq(wc)

		if Q is not None:
			self.Q = Q

		a, b = self._get_coeffs(wc, self.Q)
		self.set_coeffs(a, b)


_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(1 kHz @ 44.1 kHz, Q=0.5, DF1)",
	lambda: BiquadLowpass(1. / 44.1, Q=0.5, form=FilterForm.D1),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-1., 0.), (-3.0, 0.0), (-3.0, 0.0), (-6.1, -5.9), (-22., -20.), (-44.0, -40.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(1 kHz @ 44.1 kHz, Q=0.5, Df2)",
	lambda: BiquadLowpass(1. / 44.1, Q=0.5, form=FilterForm.D2),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-1., 0.), (-3.0, 0.0), (-3.0, 0.0), (-6.1, -5.9), (-22., -20.), (-44.0, -40.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(1 kHz @ 44.1 kHz, Q=0.5, Transposed DF1)",
	lambda: BiquadLowpass(1. / 44.1, Q=0.5, form=FilterForm.D1t),
	freqs_to_test=np.array([10., 31.62, 100., 316.2, 1000., 3162., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-1., 0.), (-3.0, 0.0), (-3.0, 0.0), (-6.1, -5.9), (-22., -20.), (-44.0, -40.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"BiquadLowpass(1 kHz @ 44.1 kHz, Q=0.5, Transposed DF2)",
	lambda: BiquadLowpass(1. / 44.1, Q=0.5, form=FilterForm.D2t),
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

	def __init__(self, wc, Q=0.5, verbose=False, form=FilterForm.D2t):
		self.Q = Q
		a, b = self._get_coeffs(wc, self.Q)
		super().__init__(a, b, form=form)

		if verbose:
			print('Biquad highpass filter: wc=%f, Q=%.2f, A:[1, %f, %f], B:[%f, %f, %f]' % (
				wc, self.Q,
				self.a[1], self.a[2],
				self.b[0], self.b[1], self.b[2]))

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
		super().throw_if_invalid_freq(wc)

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
	def __init__(self, wc, Q=0.5, peak_0dB=False, verbose=False, form=FilterForm.D2t):
		self.Q = Q
		self.peak_0dB = peak_0dB
		a, b = self._get_coeffs(wc, self.Q, self.peak_0dB)
		super().__init__(a, b, form=form)

		if verbose:
			print('Biquad bandpass filter: wc=%f, Q=%.2f, A:[1, %f, %f], B:[%f, 0, %f]' % (
				wc, self.Q,
				self.a[1], self.a[2],
				self.b[0], self.b[2]))

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
		super().throw_if_invalid_freq(wc)

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

	def __init__(self, wc, Q=0.5, verbose=False, form=FilterForm.D1):
		super().__init__(wc, Q=Q, verbose=verbose, form=form)

	def process_sample(self, x: float) -> float:

		if self.form == FilterForm.D1:
			y = super().process_sample(x)
			self.zx[0] = overdrive.tanh(self.zx[0])

		elif self.form == FilterForm.D2:
			y = super().process_sample(x)
			self.z[0] = overdrive.tanh(self.z[0])

		elif self.form == FilterForm.D1t:

			_, a1, a2 = self.a
			b0, b1, b2 = self.b

			x1, x2 = self.zx
			y1, y2 = self.zy

			v = x + x1
			y = b0 * v + y1

			vo = overdrive.tanh(v)

			self.zx[0] = self.zx[1] - a1*vo
			self.zx[1] = -a2*vo

			self.zy[0] = self.zy[1] + b1*v
			self.zy[1] = b2*v

		elif self.form == FilterForm.D2t:

			_, a1, a2 = self.a
			b0, b1, b2 = self.b

			y = b0 * x + self.z[0]

			xo = overdrive.tanh(x)

			self.z[0] = self.z[1] + b1 * xo - a1 * y
			self.z[1] = b2 * xo - a2 * y

		else:
			raise AssertionError('Invalid form: %s' % self.form.value)

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

	def __init__(self, wc, Q=0.5, verbose=False, form=FilterForm.D2t):
		super().__init__(wc, Q=Q, verbose=verbose, form=form)

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
		[
			(BiquadLowpass, dict(wc=wc, Q=Q, gain=1.0)),
			(Rossum92Biquad, dict(wc=wc, Q=Q, gain=1.0, form=FilterForm.D1)),
			(Rossum92Biquad, dict(wc=wc, Q=Q, gain=1.0, form=FilterForm.D2)),
			(Rossum92Biquad, dict(wc=wc, Q=Q, gain=1.0, form=FilterForm.D1t)),
			(Rossum92Biquad, dict(wc=wc, Q=Q, gain=1.0, form=FilterForm.D2t)),
			(OverdrivenInputBiquad, dict(wc=wc, Q=Q, gain=1.0)),
		],

		[
			(BiquadLowpass, dict(wc=wc, Q=2.0, gain=1.0)),
			(Rossum92Biquad, dict(wc=wc, Q=2.0, gain=1.0, form=FilterForm.D1)),
			(Rossum92Biquad, dict(wc=wc, Q=2.0, gain=1.0, form=FilterForm.D2)),
			(Rossum92Biquad, dict(wc=wc, Q=2.0, gain=1.0, form=FilterForm.D1t)),
			(Rossum92Biquad, dict(wc=wc, Q=2.0, gain=1.0, form=FilterForm.D1t)),
			(OverdrivenInputBiquad, dict(wc=wc, Q=2.0, gain=1.0)),
		],

		[
			(Rossum92Biquad, dict(wc=wc, Q=Q, gain=10.0, form=FilterForm.D1)),
			(Rossum92Biquad, dict(wc=wc, Q=Q, gain=10.0, form=FilterForm.D2)),
			(Rossum92Biquad, dict(wc=wc, Q=Q, gain=10.0, form=FilterForm.D1t)),
			(Rossum92Biquad, dict(wc=wc, Q=Q, gain=10.0, form=FilterForm.D2t)),
			(OverdrivenInputBiquad, dict(wc=wc, Q=Q, gain=10.0)),
		],
	]

	saws = \
		signal_generation.gen_saw(freq1 / sample_rate, n_samp) + \
		signal_generation.gen_saw(freq2 / sample_rate, n_samp) + \
		signal_generation.gen_saw(freq3 / sample_rate, n_samp)

	saws /= 3.0

	t = signal_generation.sample_time_index(n_samp, sample_rate)

	data_out = np.copy(saws) if args.outfile else None

	for filter_list in filters:
		plt.figure()
		plt.plot(t, saws, label='Input')

		for constructor, filt_args in filter_list:

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


def plot_freq_resp(args):
	import plot_utils

	one_over_sqrt2 = 1.0 / sqrt(2.0)  # 0.7071

	default_cutoff = 1000.
	sample_rate = 48000.

	wc = default_cutoff / sample_rate

	common_args = dict(wc=wc)

	filter_list = [
		(BiquadLowpass, [
			dict(Q=0.5),
			dict(Q=one_over_sqrt2),
			dict(Q=1.0),
			dict(Q=2.0)]),
		(BiquadLowpass, [
			dict(Q=2.0, form=FilterForm.D1),
			dict(Q=2.0, form=FilterForm.D2),
			dict(Q=2.0, form=FilterForm.D1t),
			dict(Q=2.0, form=FilterForm.D2t),]),
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
		plot_utils.plot_freq_resp(
			filter_types, common_args, extra_args_list,
			freqs, sample_rate,
			freq_args=['wc'],
			zoom=True, phase=True, group_delay=True)


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

	print('Showing plots')
	plt.show()


if __name__ == "__main__":
	main()
