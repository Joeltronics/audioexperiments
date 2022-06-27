#!/usr/bin/env python3

import argparse
from math import pi, tanh, cosh
import math
from multiprocessing import Pool
import time
from typing import Tuple, Optional, Any

import numpy as np
from matplotlib import pyplot as plt

from delay_reverb.delay_line import FIRDelayLine
from filters.allpass import FractionalDelayAllpass
from filters.filter_base import FilterBase
from filters.filter_audio_test import test_non_resonant_filter
from filters.filter_base import FilterBase
from generation.signal_generation import gen_sine, gen_saw
from solvers.iter_stats import IterStats
import solvers.solvers as solvers
from utils.utils import to_dB, print_timestamped


"""
Other future things to implement in single filter stages:
- OTA output distortion (hard clip, or realistic)
- 2040 style asymmetric distortion
- Nonlinear buffers (e.g. FET, Darlington, maybe crossover distortion)
- Buffers adding DC offset, and imperfect OTA bias (both of which cause CV leakage)
- Slew limiting instead of proper integration (apparently this is one element of the Polivoks sound)
"""


class ZdfOnePoleBase(FilterBase):
	"""Base class for TPT-based one-pole filters"""
	
	"""
	TODO: See if I can roll integration discretization into this class too
	Rk4OnePoleBase does this, and that makes the child classes extremely simple
	"""

	@staticmethod
	def freq_to_gain(wc: float):
		return math.tan(pi * wc)

	def __init__(self, wc, use_newton=False, iter_stats: Optional[IterStats] = None):
		self.g = 0.
		self.m = 0.
		self.set_freq(wc)
		self.s = 0.

		self.use_newton = use_newton
		self.iter_stats = iter_stats

		# These are just for (attempted) improvements of estimator
		self.prev_x = 0.
		self.prev_est = 0.
		self.prev_y = 0.

	def set_freq(self, wc: float) -> None:
		self.throw_if_invalid_freq(wc)
		self.g = self.freq_to_gain(wc)
		self.m = 1. / (self.g + 1.)

	def get_estimate(self, x: float) -> float:

		if True:
			# calculate linear case
			est = self.m * (self.g*x + self.s)

		elif False:
			# integrator state
			est = self.s

		else:
			# Use previous
			if abs(self.prev_x) > 0.01:
				est = (self.prev_y)/(self.prev_x) * x
			else:
				est = self.prev_y - self.prev_x + x

		# TODO: improve linear estimate with knowledge of previous
		#est = self.prev_y - self.prev_est + est

		self.prev_x = x
		self.prev_est = est
		return est

	# TODO: These conditionally-needed overrides confuse pylint/pylance
	# Pass these functions in as constructor arguments instead

	@staticmethod
	def y_func(x: float, y: float, s: float, g: float) -> float:
		""" Function to solve iteratively for filter output, y = f(y) """
		raise NotImplementedError('To be implemented by child class if not using Newton and not overriding process_sample_no_state_update')

	@staticmethod
	def y_zero_func(x: float, y: float, s: float, g: float) -> float:
		""" Function to solve iteratively for filter output, 0 = f(y) """
		raise NotImplementedError('To be implemented by child class if using Newton and not overriding process_sample_no_state_update')

	@staticmethod
	def dy_zero_func(x: float, y: float, s: float, g: float) -> float:
		""" Derivative of y_zero_func, for use with Newton-Raphson """
		raise NotImplementedError('To be implemented by child class if using Newton and not overriding process_sample_no_state_update')

	@staticmethod
	def state_func(x: float, y: float, s: float, g: float) -> Any:
		""" Function to determine next state value """
		return 2.0 * y - s

	def process_sample_no_state_update(self, x: float, estimate=None) -> Tuple[float, Any]:
		"""
		:returns: (output, state)
		"""
		if estimate is None:
			estimate = self.get_estimate(x)
		if self.use_newton:
			y = solvers.solve_nr(
				lambda y: self.y_zero_func(x=x, y=y, s=self.s, g=self.g),
				lambda y: self.dy_zero_func(x=x, y=y, s=self.s, g=self.g),
				estimate=estimate,
				iter_stats=self.iter_stats,
			)
		else:
			y = solvers.solve_fb_iterative(
				lambda y: self.y_func(x=x, y=y, s=self.s, g=self.g),
				estimate=estimate,
				iter_stats=self.iter_stats,
			)
		s = self.state_func(x=x, y=y, s=self.s, g=self.g)

		return y, s

	def process_sample(self, x: float) -> float:
		y, s = self.process_sample_no_state_update(x)
		self.s = s
		self.prev_y = y
		return y

	def process_vector(self, input_sig: np.ndarray) -> np.ndarray:
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n] = self.process_sample(x)
		return y

	# TODO: override process_freq_sweep, see if can improve performance a bit

	def get_state(self) -> Any:
		return self.s

	def get_integrator_state(self) -> float:
		return self.s

	def set_state(self, state: Any):
		self.s = state

	def reset(self):
		self.s = 0.0


class Rk4OnePoleBase(FilterBase):

	# 50 is probably way overkill here (this leads to FIR kernel size 100)
	FIR_LATENCY = 50

	METHOD_IIR = 'iir'
	METHOD_FIR = 'fir'
	METHOD_LERP = 'lerp'
	METHOD_SAWTOOTH_LERP = 'saw_lerp'

	def __init__(self, wc: float, method=METHOD_IIR):

		if method not in [self.METHOD_IIR, self.METHOD_FIR, self.METHOD_LERP, self.METHOD_SAWTOOTH_LERP]:
			raise ValueError('Invalid method')

		self.x_n = 0.0
		self.y_n = 0.0

		self.iir_allpass = (method == self.METHOD_IIR)
		self.lerp = (method == self.METHOD_LERP)
		self.fir_allpass = (method == self.METHOD_FIR)
		self.sawtooth_lerp = (method == self.METHOD_SAWTOOTH_LERP)

		self.fir_delay_time_0 = None
		self.fir_delay_time_05 = None
		self.fir_delay_time_1 = None
		self.half_sample_delay = None

		if self.iir_allpass:
			self.half_sample_delay = FractionalDelayAllpass(0.5)

		elif self.fir_allpass:
			self.fir_delay_time_0 = self.FIR_LATENCY
			self.fir_delay_time_05 = self.fir_delay_time_0 - 0.5
			self.fir_delay_time_1 = self.fir_delay_time_0 - 1
			self.half_sample_delay = FIRDelayLine(
				delay_samples=self.fir_delay_time_05,
				max_delay_samples=self.fir_delay_time_0,
			)

		self.g = None
		self.set_freq(wc)

	def set_freq(self, wc: float):
		self.throw_if_invalid_freq(wc)
		self.g = 2.0 * math.tan(pi * wc)

	def dydt(self, xt: float, yt: float) -> float:
		raise NotImplementedError("To be implemented by the child class!")

	def process_sample(self, x: float) -> float:

		y_n = self.y_n

		if self.fir_allpass:
			x_n = self.half_sample_delay[self.fir_delay_time_0]
			x_n_1 = self.half_sample_delay[self.fir_delay_time_1]
			x_n_05 = self.half_sample_delay.process_sample(x)

		else:
			x_n = self.x_n
			x_n_1 = x

			if self.iir_allpass:
				x_n_05 = self.half_sample_delay.process_sample(x_n)

			elif self.lerp:
				x_n_05 = 0.5 * (x_n + x_n_1)

			elif self.sawtooth_lerp:
				# If we know incoming signal is a sawtooth wave with range +/- 1, we can perfectly interpolate
				delta = x_n_1 - x_n

				if delta < -1.0:
					# Falling edge
					x_n_05 = x_n + delta + 2.0
					if x_n_05 > 1.0:
						x_n_05 -= 2.0
				elif delta > 1.0:
					# Rising edge (i.e. with falling saw)
					x_n_05 = x_n + delta - 2.0
					if x_n_05 < -1.0:
						x_n_05 += 2.0
				else:
					x_n_05 = 0.5 * (x_n + x_n_1)

				#print(f'saw lerp: x[n]={x_n:.3f}, x[n+1]={x_n_1:.3f}, x[n+0.5]={x_n_05:.3f}')

			else:
				raise AssertionError

		# step size is 1 sample, so h = 1

		k1 = self.dydt(x_n, y_n)
		k2 = self.dydt(x_n_05, y_n + (k1 / 2))
		k3 = self.dydt(x_n_05, y_n + (k2 / 2))
		k4 = self.dydt(x_n_1, y_n + k3)

		incr = (k1 + k4) / 6 + (k2 + k3) / 3

		y = y_n + incr

		self.x_n = x
		self.y_n = y
		return y

	def get_state(self) -> Any:
		return self.x_n, self.half_sample_delay.get_state(), self.y_n

	def set_state(self, state: Any) -> None:
		self.x_n, half_sample_state, self.y_n = state
		self.half_sample_delay.set_state(half_sample_state)

	def reset(self) -> None:
		self.half_sample_delay.reset()
		self.x_n = 0.0
		self.y_n = 0.0


class BasicOnePole(FilterBase):
	"""
	Linear 1-pole lowpass filter, forward Euler
	"""

	def __init__(self, wc, verbose=False):
		self.a1 = self.b0 = 0.0
		self.set_freq(wc)

		self.z1 = 0.0
		
		if verbose:
			print('Basic filter: wc=%f, a1=%f, b0=%f' % (wc, self.a1, self.b0))
	
	def set_freq(self, wc: float):
		self.a1 = math.exp(-2.0*pi * wc)
		self.b0 = 1.0 - self.a1

	def process_vector(self, input_sig):
		
		y = np.zeros_like(input_sig)
		
		for n, x in enumerate(input_sig):
			self.z1 = self.b0*x + self.a1*self.z1
			y[n] = self.z1
		
		return y

	def get_state(self) -> float:
		return self.z1

	def set_state(self, state: float):
		self.z1 = state

	def reset(self):
		self.z1 = 0.0


class TrapzOnePole(ZdfOnePoleBase):
	"""
	Linear 1-pole lowpass filter, trapezoidal integration
	"""

	def __init__(self, wc, verbose=False):
		super().__init__(wc)
		if verbose:
			print('Linear trapezoid filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def process_sample_no_state_update(self, x, estimate=None):

		# y = g*(x - y) + s
		# y + g*y = g*x + s
		# y = (g*x + s) / (g + 1)

		# if m = 1/(1+g)
		# y = m * (g*x + s)

		y = self.m * (self.g*x + self.s)
		s = 2.*y - self.s

		return y, s


class OnePoleRk4(Rk4OnePoleBase):
	def dydt(self, xt: float, yt: float) -> float:
		return self.g * (xt - yt)


class TanhInputTrapzOnePole(ZdfOnePoleBase):

	def __init__(self, wc, verbose=False):
		super().__init__(wc)
		if verbose:
			print('Tanh input filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def process_sample_no_state_update(self, x, estimate=None):

		# y = g*(tanh(x) - y) + s
		# y = (g*x + s) / (g + 1)
		# y = m * (g*x + s) for m = 1/(1+g)

		x = tanh(x)
		y = self.m * (self.g*x + self.s)
		s = 2.0*y - self.s

		return y, s


class LadderOnePole(ZdfOnePoleBase):

	def __init__(self, wc, iter_stats: Optional[IterStats] = None, use_newton=True, verbose=False):
		super().__init__(wc, iter_stats=iter_stats, use_newton=use_newton)

		if verbose:
			print('Ladder filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def get_estimate(self, x):
		# Pre-applying tanh to input seems to give slightly better estimate than strictly linear case
		return super().get_estimate(tanh(x))

	@staticmethod
	def y_func(x: float, y: float, s: float, g: float) -> float:
		return g * (tanh(x) - tanh(y)) + s

	@staticmethod
	def y_zero_func(x: float, y: float, s: float, g: float) -> float:
		return y + g * (tanh(y) - tanh(x)) - s

	@staticmethod
	def dy_zero_func(x: float, y: float, s: float, g: float) -> float:
		return g * pow(cosh(y), -2.0) + 1.0


class LadderOnePoleRk4(Rk4OnePoleBase):
	def dydt(self, xt: float, yt: float) -> float:
		return self.g * (tanh(xt) - tanh(yt))


class IdealOtaOnePole(ZdfOnePoleBase):
	def __init__(self, wc, iter_stats: Optional[IterStats] = None, use_newton=True, verbose=False):
		super().__init__(wc, iter_stats=iter_stats, use_newton=use_newton)

		if verbose:
			print('Ideal OTA filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	@staticmethod
	def y_func(x: float, y: float, s: float, g: float) -> float:
		return g * tanh(x - y) + s

	@staticmethod
	def y_zero_func(x: float, y: float, s: float, g: float) -> float:

		# y = g*tanh(x - y) + s
		# 0 = y + g*tanh(y - x) - s

		return y + g * tanh(y - x) - s

	@staticmethod
	def dy_zero_func(x: float, y: float, s: float, g: float) -> float:
		# 0 = y + g*tanh(y-x) - s
		# d/dy = g*(sech(x-y))^2 + 1
		#      = g*(cosh(x-y))^-2 + 1
		return g * pow(cosh(x - y), -2.0) + 1.0


class IdealOtaOnePoleRk4(Rk4OnePoleBase):
	def dydt(self, xt: float, yt: float) -> float:
		return self.g * tanh(xt - yt)


class IdealOtaOnePoleNegative(ZdfOnePoleBase):

	def __init__(self, wc, iter_stats: Optional[IterStats] = None, use_newton=True, verbose=False):
		super().__init__(wc, iter_stats=iter_stats, use_newton=use_newton)

		if verbose:
			print('Ideal OTA negative filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def get_estimate(self, x):
		return super().get_estimate(-x)

	@staticmethod
	def y_func(x: float, y: float, s: float, g: float) -> float:
		return g * tanh(-x - y) + s

	@staticmethod
	def y_zero_func(x: float, y: float, s: float, g: float) -> float:
		return y + g * tanh(x + y) - s

	@staticmethod
	def dy_zero_func(x: float, y: float, s: float, g: float) -> float:
		# 0 = y + g*tanh(y-x) - s
		# d/dy = g*(sech(x-y))^2 + 1
		#      = g*(cosh(x-y))^-2 + 1
		return g * pow(cosh(x + y), -2.0) + 1


########## Utilities & main/plot ##########


def do_fft(x, n_fft, window=False):

	if window:
		y = np.fft.fft(x * np.hamming(len(x)), n=n_fft)
	else:
		y = np.fft.fft(x, n=n_fft)

	f = np.fft.fftfreq(n_fft, 1.0)

	# Only take first half
	y = y[0:len(y)//2]
	f = f[0:len(f)//2]

	y = to_dB(np.abs(y))

	return y, f


def find_3dB_freq(freqs, Y):

	# Not the most "pythonic" way of doing this, but it works

	y_plus_3 = Y + 3.0

	prev_f = freqs[0]
	prev_y = y_plus_3[0]

	for f, y in zip(freqs, y_plus_3):

		if (y <= 0 and prev_y > 0) or (y >= 0 and prev_y < 0):
			zero_x_loc = (0.0 - prev_y) / (y - prev_y)
			assert(zero_x_loc >= 0.0 and zero_x_loc <= 1.0)
			return (f * zero_x_loc) + (prev_f * (1.0 - zero_x_loc))

		prev_y = y
		prev_f = f

	return 0


def impulse_response(fc=0.003, n_samp=4096, n_fft=None):

	if n_fft is None:
		n_fft = n_samp

	x = np.zeros(n_samp)
	x[0] = 1.0

	filt_lin        = BasicOnePole(fc)
	filt_tanh_input = TrapzOnePole(fc)

	y_lin        = filt_lin.process(x)
	y_tanh_input = filt_tanh_input.process(x)

	fft_y_lin,        f = do_fft(y_lin, n_fft=n_fft, window=False)
	fft_y_tanh_input, _ = do_fft(y_tanh_input, n_fft=n_fft, window=False)

	fc1 = find_3dB_freq(f, fft_y_lin)
	fc2 = find_3dB_freq(f, fft_y_tanh_input)

	print('Ideal fc = %.4f' % fc)
	print('Basic fc = %.4f (error %.2f%%)' % (fc1, abs(fc1-fc)/fc * 100.0))
	print('Trapz fc = %.4f (error %.2f%%)' % (fc2, abs(fc2-fc)/fc * 100.0))

	plt.figure()

	plt.subplot(211)

	plt.semilogx(f, fft_y_lin, f, fft_y_tanh_input)
	plt.legend(['Basic', 'Trapezoid'], loc=3)
	plt.title('fc = %f' % fc)
	plt.grid()
	plt.ylim(-12, 3)

	plt.subplot(212)

	plt.semilogx(f, fft_y_lin, f, fft_y_tanh_input)
	plt.grid()


def find_amp_phase(y, x):

	y_amp = math.sqrt(np.sum(np.square(y)))
	x_amp = math.sqrt(np.sum(np.square(x)))

	#print('y_amp = %f, x_amp = %f' % (y_amp, x_amp))

	amp = y_amp / x_amp

	ph = 0 # TODO

	return amp, ph


def freq_sweep(fc=0.003, n_samp=4096, n_sweep=None):

	if n_sweep is None:
		n_sweep = n_samp

	f = np.linspace(0.0, 0.49, n_sweep)
	fft_y_lin = np.zeros_like(f)
	fft_y_tanh_input = np.zeros_like(f)
	ph1 = np.zeros_like(f)
	ph2 = np.zeros_like(f)

	filt_lin = BasicOnePole(fc)
	filt_tanh_input = TrapzOnePole(fc)

	for n, sin_freq in enumerate(f):

		if sin_freq == 0:
			x = np.ones(n_samp)
		else:
			x = gen_sine(sin_freq, n_samp)

		filt_lin.z1 = 0.0
		filt_tanh_input.s = 0.0

		y_lin = filt_lin.process(x)
		y_tanh_input = filt_tanh_input.process(x)

		fft_y_lin[n], ph1[n] = find_amp_phase(y_lin, x)
		fft_y_tanh_input[n], ph2[n] = find_amp_phase(y_tanh_input, x)

	fft_y_lin = 20*np.log10(fft_y_lin)
	fft_y_tanh_input = 20*np.log10(fft_y_tanh_input)

	plt.figure()

	plt.subplot(211)

	plt.semilogx(f, fft_y_lin, f, fft_y_tanh_input)
	plt.legend(['Basic','Trapezoid'], loc=3)
	plt.title('fc = %f' % fc)
	plt.grid()
	plt.ylim(-12, 3)

	plt.subplot(212)

	plt.semilogx(f, ph1, f, ph2)
	plt.grid()


def plot_nonlin_filters(fc=0.1, f_saw=0.01, gain=2.0, n_samp=2048, method='iteration', stats=False):

	# TODO: also plot some oversampled cases

	METHOD_ITER = 'iteration'
	METHOD_NEWTON = 'newton'
	METHOD_RK4_LIN = 'rk4_lin'
	METHOD_RK4_NONLIN = 'rk4_nonlin'

	if method not in [METHOD_ITER, METHOD_NEWTON, METHOD_RK4_LIN, METHOD_RK4_NONLIN]:
		raise ValueError(f'Invalid method: {method}')

	method_str = {
		METHOD_ITER: 'Simple iteration',
		METHOD_NEWTON: 'Newton-Raphson',
		METHOD_RK4_LIN: 'RK4',
		METHOD_RK4_NONLIN: 'RK4',
	}[method]

	iterative = method in [METHOD_ITER, METHOD_NEWTON]

	fig, (time_plot, freq_plot) = plt.subplots(2, 1)

	MAX_LATENCY = Rk4OnePoleBase.FIR_LATENCY

	t = np.arange(n_samp)
	x_full = gen_saw(f_saw, n_samp + MAX_LATENCY, start_phase=0.5) * gain * 0.5
	x = x_full[:n_samp]

	fft_x, f = do_fft(x, n_fft=n_samp, window=True)

	time_plot.plot(t, x, label='Input')
	peaks_f, peaks_y = find_fft_peaks(f, fft_x)
	line, = freq_plot.semilogx(f, fft_x, label='Input')
	freq_plot.semilogx(peaks_f, peaks_y, '.', color=line.get_color())

	stats_ladder = IterStats('Ladder, ' + method_str) if (iterative and stats) else None
	stats_ota = IterStats('Ideal OTA, ' + method_str) if (iterative and stats) else None
	stats_ota_neg = IterStats('Ideal OTA Negative, ' + method_str) if (iterative and stats) else None

	if iterative:
		fig.suptitle(method_str)
		use_newton = (method == METHOD_NEWTON)
		filters = [
			dict(filt=TrapzOnePole(fc), name='Linear', iter_stats=None),
			#dict(filt=TanhInputTrapzOnePole(fc), name='tanh input', iter_stats=None),
			dict(filt=LadderOnePole(fc, use_newton=use_newton, iter_stats=stats_ladder), name='Ladder'),
			dict(filt=IdealOtaOnePole(fc, use_newton=use_newton, iter_stats=stats_ota), name='OTA'),
			dict(filt=IdealOtaOnePoleNegative(fc, use_newton=use_newton, iter_stats=stats_ota_neg), name='-OTA', negate=True),
		]
	elif method == METHOD_RK4_LIN:
		fig.suptitle('RK4, linear filters')
		filters = [
			dict(filt=TrapzOnePole(fc), name='Linear (TPT)', iter_stats=None),
			dict(filt=OnePoleRk4(fc, method='saw_lerp'), name='Linear RK4, perfect interpolation'),
			dict(filt=OnePoleRk4(fc, method='lerp'), name='Linear RK4 lerp'),
			dict(filt=OnePoleRk4(fc, method='iir'), name='Linear RK4 IIR Allpass'),
			dict(filt=OnePoleRk4(fc, method='fir'), name='Linear RK4 FIR (latency compensated)', latency=Rk4OnePoleBase.FIR_LATENCY),
		]
	elif method == METHOD_RK4_NONLIN:
		fig.suptitle('RK4, nonlinear filters')
		filters = [
			dict(filt=LadderOnePole(fc, use_newton=True), name='Ladder (TPT)'),
			dict(filt=LadderOnePoleRk4(fc), name='Ladder RK4 IIR Allpass'),

			dict(filt=IdealOtaOnePole(fc, use_newton=True), name='OTA (TPT)'),
			dict(filt=IdealOtaOnePoleRk4(fc), name='OTA RK4 IIR Allpass'),
		]
	else:
		raise AssertionError

	for filter in filters:
		filt = filter['filt']
		name = filter['name']

		negate = filter['negate'] if 'negate' in filter else False
		latency = filter['latency'] if 'latency' in filter else 0

		assert latency <= MAX_LATENCY

		if latency:
			y = filt.process_vector(x_full[:n_samp + latency])
			y = y[-n_samp:]
		else:
			y = filt.process_vector(x)

		if negate:
			y = -y

		fft_y, f = do_fft(y, n_fft=n_samp, window=True)
		time_plot.plot(t, y, label=name)
		peaks_f, peaks_y = find_fft_peaks(f, fft_y)
		line, = freq_plot.semilogx(f, fft_y, label=name)
		freq_plot.semilogx(peaks_f, peaks_y, '.', color=line.get_color())

	time_plot.legend()
	time_plot.set_xlim([0, 256])
	time_plot.grid()

	freq_plot.grid()
	
	for stats in [stats_ladder, stats_ota, stats_ota_neg]:
		if stats is not None:
			stats.output()


def plot_impulse_response(fc=0.003, n_samp=4096, n_fft=None):
	if n_fft is None:
		n_fft = n_samp

	x = np.zeros(n_samp)
	x[0] = 1.0

	filt1 = BasicOnePole(fc)
	filt2 = TrapzOnePole(fc)

	y1 = filt1.process_vector(x)
	y2 = filt2.process_vector(x)

	Y1, f = do_fft(y1, n_fft=n_fft, window=False)
	Y2, _ = do_fft(y2, n_fft=n_fft, window=False)

	Y1 = to_dB(Y1)
	Y2 = to_dB(Y2)

	fc1 = find_3dB_freq(f, Y1)
	fc2 = find_3dB_freq(f, Y2)

	print('Ideal fc = %.4f' % fc)
	print('Basic fc = %.4f (error %.2f%%)' % (fc1, abs(fc1 - fc) / fc * 100.0))
	print('Trapz fc = %.4f (error %.2f%%)' % (fc2, abs(fc2 - fc) / fc * 100.0))

	plt.figure()

	plt.subplot(211)

	plt.semilogx(f, Y1, f, Y2)
	plt.legend(['Basic', 'Trapezoid'], loc=3)
	plt.title('fc = %f' % fc)
	plt.grid()
	plt.ylim(-12, 3)

	plt.subplot(212)

	plt.semilogx(f, Y1, f, Y2)
	plt.grid()


def find_fft_peaks(f: np.ndarray, y: np.ndarray, peak_thresh=20.0) -> Tuple[list, list]:

	if len(f) != len(y):
		raise ValueError(f'Must have same length ({len(f)} != {len(y)})')

	peaks_f = []
	peaks_y = []

	prev_peak_val = None

	for idx in range(1, len(y) - 1):

		val = y[idx]

		is_local_maximum = (val >= y[idx - 1]) and (val > y[idx + 1])

		if not is_local_maximum:
			continue

		# It's a local maximum, but is it a peak?
		# e.g. window side lobes are local maxima, but we don't want those

		if (prev_peak_val is None) or (peak_thresh is None):
			is_peak = True
		else:
			is_peak = val > (prev_peak_val - peak_thresh)

		if is_peak:
			peaks_f.append(f[idx])
			peaks_y.append(val)
			prev_peak_val = val

	return peaks_f, peaks_y


def plot_step(fc: float, n_samp_per_level: int):

	levels = [
		0, 0.1, 0.2, 0.3,
		0, 0.25, 0.5, 0.75,
		0, 0.5, 1.0, 1.5,
		0, 1.0, 2.0, 3.0,
		0, 2.0, 4.0, 6.0,
		0, 4.0, 8.0, 12.0,
	]

	x = []
	for level in levels:
		x += [float(level)] * n_samp_per_level

	x = np.array(x)

	linear_filter = TrapzOnePole(fc)
	y_linear = linear_filter.process_vector(x)

	filter_specs = [
		#dict(filt=TanhInputTrapzOnePole(fc), name='tanh input'),
		dict(filt=LadderOnePole(fc), name='Ladder'),
		dict(filt=IdealOtaOnePole(fc), name='OTA'),
		#dict(filt=IdealOtaOnePoleNegative(fc), name='-OTA', negate=True),
	]

	fig, [ax_step, ax_slope, ax_err] = plt.subplots(3, 1)
	fig.suptitle('Step responses')

	#ax_step.plot(x, '-', label='Input')
	#next(ax_slope._get_lines.prop_cycler)
	#next(ax_err._get_lines.prop_cycler)

	ax_step.plot(y_linear, '-', label='Linear')
	ax_slope.plot(np.diff(y_linear), label='Linear')
	next(ax_err._get_lines.prop_cycler)

	for filter_spec in filter_specs:
		filt = filter_spec['filt']
		y = filt.process_vector(x)

		if 'negate' in filter_spec and filter_spec['negate']:
			y = -y

		delta = np.diff(y)

		err = y_linear - y

		ax_step.plot(y, label=filter_spec['name'])
		ax_err.plot(err, label=filter_spec['name'])
		ax_slope.plot(delta, label=filter_spec['name'])

	ax_step.grid()
	ax_step.legend()

	ax_slope.grid()
	ax_slope.legend()
	ax_slope.set_ylabel('Slope (sample delta)')

	ax_err.grid()
	ax_err.legend()
	ax_err.set_ylabel('Error from linear')


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--no-pool', action='store_false', dest='use_pool')
	parser.add_argument('--lm13700', action='store_true', help='Include LM13700 filters')
	parser.add_argument('--stats', action='store_true', help='Show iteration stats')
	return parser


def plot(args=None):

	#plot_impulse_response(fc=0.001, n_samp=32768)
	#freq_sweep(fc=0.001, n_samp=2048)

	for method in ['iteration', 'newton', 'rk4_lin', 'rk4_nonlin']:
		plot_nonlin_filters(fc=0.1, f_saw=0.01, gain=(2.0 if (method == 'rk4_lin') else 4.0), n_samp=2048, method=method, stats=args.stats)

	plot_step(fc=0.1, n_samp_per_level=250)

	plt.show()


def main(args=None):
	from filters.zdf.lm13700 import Lm13700OnePolePositive, Lm13700OnePoleInverting

	start = time.monotonic()
	print_timestamped('Starting onepole audio test')

	filters = [
		dict(constructor=TrapzOnePole, name='Linear'),
		dict(constructor=OnePoleRk4, name='Linear RK4'),
		dict(constructor=TanhInputTrapzOnePole, name='tanh input'),
		dict(constructor=LadderOnePole, name='Ladder'),
		dict(constructor=LadderOnePoleRk4, name='Ladder RK4'),
		dict(constructor=IdealOtaOnePole, name='OTA'),
		dict(constructor=IdealOtaOnePoleRk4, name='OTA RK4'),
	]

	if args.lm13700:
		filters += [
			dict(constructor=Lm13700OnePolePositive, name='LM13700 noninverting'),
			dict(constructor=Lm13700OnePoleInverting, name='LM13700 inverting'),
		]

	def test_filter(filter_spec, pool):
		name = filter_spec['name']

		filename = 'onepole_%s.wav' % name.lower().replace(' ', '_')

		test_non_resonant_filter(
			filter_constructor=filter_spec['constructor'],
			filename=filename,
			sample_rate_out=48000,
			oversampling=4,
			name=name,
			pool=pool,
		)

	if args.use_pool:
		# TODO: this still isn't the most efficient, as each filter will wait for all of its own jobs to complete
		with Pool() as p:
			for filter_spec in filters:
				test_filter(filter_spec, pool=p)
	else:
		for filter_spec in filters:
			test_filter(filter_spec, pool=None)

	duration = time.monotonic() - start

	if duration > 60.0:
		duration_str = '%i:%i:%.3f' % (
			int(duration / 60.0),
			int(duration % 60.0),
			(duration % 1.0)
		)
	else:
		duration_str = '%.3f seconds' % duration

	print_timestamped(f'Total duration: {duration_str}')
