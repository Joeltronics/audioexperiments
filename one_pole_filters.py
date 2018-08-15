#!/usr/bin/env python3

from filter import Filter
from filter_unit_test import FilterUnitTest
import utils

from math import pi, exp, tan, log10
import numpy as np


# TODO: figure out why one pole filter response is way off, and tighten up test requirements
# I don't think it's just measurement error, because Butterworth filters are dead on
# Seems to be an error in calculating coeffs


# TODO: could have much more efficient process_vector using scipy.signal.lfilter
# would need to deal with state updates with zi and zf


_unit_tests = []


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
		self.a1 = exp(-2.0 * pi * wc)
		self.b0 = 1.0 - self.a1

	def process_sample(self, x):
		self.z1 = self.b0 * x + self.a1 * self.z1
		y = self.z1
		return y


_unit_tests.append(FilterUnitTest(
	"BasicOnePole(1 kHz @ 44.1 kHz)",
	lambda: BasicOnePole(1. / 44.1),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"BasicOnePole(100 Hz @ 44.1 kHz)",
	lambda: BasicOnePole(100. / 44100.),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-3.0, 0.0), (-4.0, -2.0), (-21.0, -20.0), (-48.0, -38.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))


class BasicOnePoleHighpass(Filter):
	def __init__(self, wc, verbose=False):
		self.lpf = BasicOnePole(wc)
		if verbose:
			print(
				'Basic one pole highpass filter - underlying LPF: wc=%f, a1=%f, b0=%f' % (wc, self.lpf.a1, self.lpf.b0))

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
	lambda: BasicOnePoleHighpass(1. / 44.1),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-48.0, -38.0), (-24.0, -18.0), (-4.0, -2.0), (-3.0, 0.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))


class TrapzOnePole(Filter):
	"""Trapezoidal-integration one pole filter"""

	def __init__(self, wc, verbose=False):
		self.s = 0.0  # State
		self.g = 0.0  # Integrator gain
		self.m = 0.0  # Pre-calculated value (derived from gain)
		self.set_freq(wc)
		if verbose:
			print('Trapezoid filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi * wc))

	def reset(self):
		self.s = 0.0

	def set_freq(self, wc):
		self.g = tan(pi * wc)
		self.m = 1.0 / (self.g + 1.0)

	def process_sample(self, x):
		# y = g*(x - y) + s
		#   = g*x - g*y + s
		#   = (g*x + s) / (g + 1)
		#   = m * (g*x + s)

		y = self.m * (self.g * x + self.s)
		self.s = 2.0 * y - self.s
		return y


_unit_tests.append(FilterUnitTest(
	"TrapzOnePole(1 kHz @ 44.1 kHz)",
	lambda: TrapzOnePole(1. / 44.1),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-0.1, 0.0), (-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))

_unit_tests.append(FilterUnitTest(
	"TrapzOnePole(100 Hz @ 44.1 kHz)",
	lambda: TrapzOnePole(100. / 44100.),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0), (-48.0, -38.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
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
	lambda: TrapzOnePoleHighpass(1. / 44.1),
	freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
	expected_freq_response_range_dB=[(-48.0, -38.0), (-24.0, -18.0), (-4.0, -2.0), (-3.0, 0.0)],
	expected_phase_response_range_degrees=None,  # [(), (), (), None],
	deterministic=True,
	linear=True
))


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
			print('Leaky integrator: wc=%f, alpha=%f, w_norm=%s, gain=%.2f dB' % (
			wc, self.alpha, str(self.w_norm), utils.to_dB(self.gain)))

	def reset(self):
		self.z1 = 0.0

	def set_freq(self, wc):
		self.alpha = exp(-2.0 * pi * wc)
		self.one_minus_alpha = 1.0 - self.alpha

		# Now calculate gain
		if self.w_norm is None:
			self.gain = 1.0
		else:
			# Just use Bode plot approximation (i.e. 20 dB per decade)
			decades_above_w_norm = log10(self.w_norm / wc)
			self.gain = utils.from_dB(decades_above_w_norm * 20.0)

	def process_sample(self, x):
		self.z1 = self.alpha * self.z1 + x * self.one_minus_alpha
		return self.gain * self.z1


def _run_unit_tests():
	import unit_test
	unit_test.run_unit_tests(_unit_tests)


def main():
	from matplotlib import pyplot as plt
	import argparse
	from plot_filters import plot_filters

	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true', help='Verbose unit tests')
	parser.add_argument('--test', action='store_true', help='Run unit tests')
	args = parser.parse_args()

	if args.test:
		_run_unit_tests()
		return

	default_cutoff = 1000.
	sample_rate = 48000.

	filter_list = [
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
		(LeakyIntegrator, [
			dict(cutoff=10.0, f_norm=100.0),
			dict(cutoff=10.0, f_norm=1000.0),
			dict(cutoff=100.0, f_norm=1000.0),
			dict(cutoff=100.0, f_norm=10000.0),
			dict(cutoff=1000.0, f_norm=10000.0)]),
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


