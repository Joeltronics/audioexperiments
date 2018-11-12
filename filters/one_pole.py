#!/usr/bin/env python3

from .filter_base import FilterBase, FilterForm
from unit_test.processor_unit_test import FilterUnitTest
from utils import utils

from math import pi, exp, tan, log10
import numpy as np
import scipy.signal


# TODO: figure out why one pole filter response is way off, and tighten up test requirements
# I don't think it's just measurement error, because Butterworth filters are dead on
# Seems to be an error in calculating coeffs


_unit_tests = []


class BasicOnePole(FilterBase):
	def __init__(self, wc, verbose=False, form=FilterForm.D1, gain=1.0):

		self.z1 = 0.0
		self.na1 = 0.0  # -a1
		self.b0 = 0.0
		self.gain = gain

		self.form = form

		self.set_freq(wc)
		if verbose:
			print('Basic one pole filter: wc=%f, a1=%f, b0=%f' % (wc, -self.na1, self.b0))

	def reset(self):
		self.z1 = 0.0

	def get_state(self):
		return self.z1

	def set_state(self, s):
		self.z1 = s

	def set_freq(self, wc, gain=None):
		super().throw_if_invalid_freq(wc)

		if gain is not None:
			self.gain = gain

		self.na1 = exp(-2.0 * pi * wc)
		self.b0 = (1.0 - self.na1) * self.gain

		assert self.na1 > 0.0
		assert self.b0 > 0.0

	def process_sample(self, x):

		if self.form == FilterForm.D1:
			y = self.z1 = (self.b0 * x) + (self.na1 * self.z1)

		elif self.form == FilterForm.D2:
			self.z1 = x + (self.na1 * self.z1)
			y = self.b0 * self.z1

		elif self.form == FilterForm.D1t:
			v = self.z1 + x
			self.z1 = (self.na1 * v)
			y = v * self.b0

		elif self.form == FilterForm.D2t:
			y = (self.b0 * x) + self.z1
			self.z1 = (self.na1 * y)

		else:
			raise ValueError('Unexpected filter form %s!' % str(self.form.value))

		return y

	def process_vector(self, vec: np.ndarray) -> np.ndarray:

		if self.form == FilterForm.D1:

			# z1 is state from Direct Form I - Need to convert to transposed DFII and back

			"""
			DF1:  z1_D1  = y
			DF2t: z1_D2t = -a1*y
			
			Together:
			z1_D2t = -a1*y = -a1*z1_D1
			"""

			z1_D2t = self.na1 * self.z1

			y, zf = scipy.signal.lfilter(b=[self.b0], a=[1.0, -self.na1], x=vec, zi=[z1_D2t])
			assert len(zf) == 1
			self.z1 = zf[0] / self.na1

		elif self.form == FilterForm.D2t:
			y, zf = scipy.signal.lfilter(b=[self.b0], a=[1.0, -self.na1], x=vec, zi=[self.z1])

			assert len(zf) == 1
			self.z1 = zf[0]

		else:
			# TODO: use scipy with D2 and D1t as well

			y = np.zeros_like(vec)
			for n, x in enumerate(vec):
				y[n] = self.process_sample(x)

		return y


_unit_tests.append(FilterUnitTest(
		"BasicOnePole(1 kHz @ 44.1 kHz), DF1",
		lambda: BasicOnePole(1. / 44.1, form=FilterForm.D1),
		freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
		expected_freq_response_range_dB=[(-0.1, 0.0), (-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0)],
		expected_phase_response_range_degrees=None,  # [(), (), (), None],
		deterministic=True,
		linear=True
	))

_unit_tests.append(FilterUnitTest(
		"BasicOnePole(1 kHz @ 44.1 kHz), DF2",
		lambda: BasicOnePole(1. / 44.1, form=FilterForm.D2),
		freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
		expected_freq_response_range_dB=[(-0.1, 0.0), (-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0)],
		expected_phase_response_range_degrees=None,  # [(), (), (), None],
		deterministic=True,
		linear=True
	))

_unit_tests.append(FilterUnitTest(
		"BasicOnePole(1 kHz @ 44.1 kHz), Transposed DF1",
		lambda: BasicOnePole(1. / 44.1, form=FilterForm.D1t),
		freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
		expected_freq_response_range_dB=[(-0.1, 0.0), (-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0)],
		expected_phase_response_range_degrees=None,  # [(), (), (), None],
		deterministic=True,
		linear=True
	))

_unit_tests.append(FilterUnitTest(
		"BasicOnePole(1 kHz @ 44.1 kHz), Transposed DF2",
		lambda: BasicOnePole(1. / 44.1, form=FilterForm.D2t),
		freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
		expected_freq_response_range_dB=[(-0.1, 0.0), (-3.0, 0.0), (-4.0, -2.0), (-24.0, -18.0)],
		expected_phase_response_range_degrees=None,  # [(), (), (), None],
		deterministic=True,
		linear=True
	))

_unit_tests.append(FilterUnitTest(
		"BasicOnePole(1 kHz @ 44.1 kHz, 3 dB gain)",
		lambda: BasicOnePole(1. / 44.1, gain=utils.from_dB(3.0)),
		freqs_to_test=np.array([10., 100., 1000., 10000.]) / 44100.,
		expected_freq_response_range_dB=[(2.9, 3.0), (0.0, 3.0), (-1.0, 1.0), (-21.0, -15.0)],
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


class BasicOnePoleHighpass(FilterBase):
	def __init__(self, wc, verbose=False, form=FilterForm.D2t):
		self.lpf = BasicOnePole(wc, form=form)
		if verbose:
			print(
				'Basic one pole highpass filter - underlying LPF: wc=%f, a1=%f, b0=%f' % (wc, -self.lpf.na1, self.lpf.b0))

	def reset(self):
		self.lpf.reset()

	def get_state(self):
		return self.lpf.get_state()

	def set_state(self, s):
		self.lpf.set_state(s)

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


class TrapzOnePole(FilterBase):
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

	def get_state(self):
		return self.s

	def set_state(self, s):
		self.s = s

	def set_freq(self, wc):
		super().throw_if_invalid_freq(wc)
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


class TrapzOnePoleHighpass(FilterBase):
	def __init__(self, wc, verbose=False):
		self.x_prev = 0.0
		self.lpf = TrapzOnePole(wc=wc, verbose=verbose)

	def set_freq(self, wc: float):
		self.lpf.set_freq(wc)

	def reset(self):
		self.x_prev = 0.0
		self.lpf.reset()

	def get_state(self):
		return [self.x_prev, self.lpf.get_state()]

	def set_state(self, s):
		self.x_prev = s[0]
		self.lpf.set_state(s[1])

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


class LeakyIntegrator(FilterBase):
	def __init__(self, wc, w_norm=None, verbose=False):
		"""

		:param wc: normalized cutoff frequency (cutoff / sample rate)
		:param w_norm: frequency at which gain will be normalized to 0 dB; only accurate if w_norm >> wc
		:param verbose:
		"""

		self.w_norm = w_norm
		gain = self._calc_gain(wc)
		self.lpf = BasicOnePole(wc, gain=gain)

		if verbose:
			print('Leaky integrator: wc=%f, a1=%f, b0=%f, gain=%.2f dB' % (wc, -self.lpf.na1, self.lpf.b0, utils.to_dB(gain)))

	def reset(self):
		self.lpf.reset()

	def get_state(self):
		return self.lpf.get_state()

	def set_state(self, s):
		self.lpf.set_state(s)

	def set_freq(self, wc):
		self.lpf.set_freq(wc, gain=self._calc_gain(wc))

	def _calc_gain(self, wc):
		if self.w_norm is None:
			return 1.0
		else:
			# Just use Bode plot approximation (i.e. 20 dB per decade)
			decades_above_w_norm = log10(self.w_norm / wc)
			return utils.from_dB(decades_above_w_norm * 20.0)

	def process_sample(self, x):
		return self.lpf.process_sample(x)

	def process_vector(self, vec: np.ndarray):
		return self.lpf.process_vector(vec)


def test(verbose=False):
	from unit_test import unit_test
	return unit_test.run_unit_tests(_unit_tests, verbose=verbose)


def plot(args):
	from matplotlib import pyplot as plt
	from utils.plot_utils import plot_freq_resp

	sr = 48000.

	filter_list = [

		(BasicOnePole, [
			dict(wc=100.0/sr),
			dict(wc=1000.0/sr),
			dict(wc=10000.0/sr)]),

		(BasicOnePoleHighpass, [
			dict(wc=10.0/sr),
			dict(wc=100.0/sr),
			dict(wc=1000.0/sr)]),

		(TrapzOnePole, [
			dict(wc=100.0/sr),
			dict(wc=1000.0/sr),
			dict(wc=10000.0/sr)]),

		(TrapzOnePoleHighpass, [
			dict(wc=10.0/sr),
			dict(wc=100.0/sr),
			dict(wc=1000.0/sr)]),

		(LeakyIntegrator, [
			dict(wc=10.0/sr, w_norm=100.0/sr),
			dict(wc=10.0/sr, w_norm=1000.0/sr),
			dict(wc=100.0/sr, w_norm=1000.0/sr),
			dict(wc=100.0/sr, w_norm=10000.0/sr),
			dict(wc=1000.0/sr, w_norm=10000.0/sr)]),
	]

	freqs = np.array([
		10., 20., 30., 40., 50., 60., 70., 80., 90.,
		100., 150., 200., 300., 400., 500., 600., 700., 800., 900.,
		1000., 1500., 2000., 3000., 4000., 5000., 6000., 7000., 8000., 9000.,
		10000., 11000., 13000., 15000., 20000.])

	for filter_types, extra_args_list in filter_list:
		plot_freq_resp(
			filter_types, None, extra_args_list,
			freqs, sr,
			freq_args=['wc', 'w_norm'],
			zoom=True, phase=True, group_delay=True)

	plt.show()


def main(args):
	plot(args)
