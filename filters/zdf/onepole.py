#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from math import pi, tanh, pow
import math
from typing import Tuple

from generation.signal_generation import gen_sine, gen_saw
from solvers.iter_stats import IterStats
import solvers.solvers as solvers
from utils.utils import to_dB


#g_default_eps = 1e-7 # -140 dB
#g_default_eps = 1e-6 # -120 dB
g_default_eps = 1e-5 # -100 dB
#g_default_eps = 1e-4 # -80 dB


"""
Other future things to implement in single filter stage:
- Using negative LP input
- With nonlinear buffers (e.g. BJT, FET, Darlington - also crossover distortion?)
"""

"""
Test cases to determine best algorithms/approximations/estimates:

Sawtooth

Combination of sines
	One paper I found used 110 Hz + 155 Hz, which seems good (IM is at 75/200, HD2 at 220/310)

Variety of gain levels

Variety of input frequencies

Variety cutoff frequencies

Instant transitions vs bandlimited

Square waves
	good case because they have fast transitions and are always at one end or the
	other (in heavy distortion region), yet the distortion wouldn't affect the
	wave if it weren't for the lowpass filtering

Different stages
1st stage might have different optimal parameters from 4th stage

Different resonance levels, including self-osc

Audio-rate FM
"""


class ZdfOnePoleBase:

	@staticmethod
	def freq_to_gain(wc: float):
		return math.tan(pi * wc)

	def __init__(self, wc):
		self.g = 0.
		self.m = 0.
		self.set_freq(wc)
		self.s = 0.

		# These are just for (attempted) improvements of estimator
		self.prev_x = 0.
		self.prev_ext = 0.
		self.prev_y = 0.

	def set_freq(self, wc: float) -> None:
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

		# Improve linear estimate with knowledge of previous
		#est = self.prev_y - self.prev_ext + est

		self.prev_x = x
		self.prev_ext = est
		return est

	def process_tick_no_state_update(self, x: float) -> Tuple[float, float]:
		raise NotImplementedError('To be implemented by child class!')

	def process_tick(self, x: float) -> float:
		y, s = self.process_tick_no_state_update(x)
		self.s = s
		self.prev_y = y
		return y

	def process_buf(self, input_sig: np.ndarray) -> np.ndarray:
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n] = self.process_tick(x)
		return y


class BasicOnePole:
	"""
	Linear 1-pole lowpass filter, forward Euler
	"""

	def __init__(self, wc):
		self.a1 = math.exp(-2.0*pi * wc)
		self.b0 = 1.0 - self.a1
		self.z1 = 0.0
		
		print('Basic filter: wc=%f, a1=%f, b0=%f' % (wc, self.a1, self.b0))
	
	def process_buf(self, input_sig):
		
		y = np.zeros_like(input_sig)
		
		for n, x in enumerate(input_sig):
			self.z1 = self.b0*x + self.a1*self.z1
			y[n] = self.z1
		
		return y


class TrapzOnePole(ZdfOnePoleBase):
	"""
	Linear 1-pole lowpass filter, trapezoidal integration
	"""

	def __init__(self, wc, verbose=False):
		super().__init__(wc)
		if verbose:
			print('Linear trapezoid filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def process_tick_no_state_update(self, x):

		# y = g*(x - y) + s
		# y + g*y = g*x + s
		# y = (g*x + s) / (g + 1)

		# if m = 1/(1+g)
		# y = m * (g*x + s)

		y = self.m * (self.g*x + self.s)
		s = 2.*y - self.s

		return y, s


class TanhInputTrapzOnePole(ZdfOnePoleBase):

	def __init__(self, wc, verbose=False):
		super().__init__(wc)
		if verbose:
			print('Tanh input filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def process_tick_no_state_update(self, x):

		# y = g*(tanh(x) - y) + s
		# y = (g*x + s) / (g + 1)
		# y = m * (g*x + s) for m = 1/(1+g)

		x = tanh(x)
		y = self.m * (self.g*x + self.s)
		s = 2.0*y - self.s

		return y, s


class LadderOnePole(ZdfOnePoleBase):

	def __init__(self, wc, iter_stats=None, use_newton=True, verbose=False):
		super().__init__(wc)
		self.iter_stats = IterStats('1P Ladder') if iter_stats is None else iter_stats
		self.use_newton = use_newton

		if verbose:
			print('Ladder filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def get_estimate(self, x):
		# Pre-applying tanh to input seems to give slightly better estimate than strictly linear case
		return super().get_estimate(tanh(x))

	def process_tick_no_state_update(self, x):

		# y = g*(tanh(x) - tanh(y)) + s
		# y + g*(tanh(y) - tanh(x)) - s = 0

		def f(y):
			return y + self.g*(tanh(y) - tanh(x)) - self.s

		est = self.get_estimate(x)

		if self.use_newton:

			# 0 = y + g*(tanh(y) - tanh(x)) - s
			# d/dy = -g*(sech(y))^2 - 1
			#      = -g*(cosh(y))^-2 - 1

			def df(y):
				return self.g * math.pow(math.cosh(y), -2.0) + 1

			y = solvers.solve_nr_legacy(f, df, est, iter_stats=self.iter_stats)

		else:

			# y = g*(tanh(x) - tanh(y)) + s
			# y + g*tanh(y) = g*tanh(x) + s

			# y + g*tanh(y) = rhs
			# y + g*tanh(y) - rhs = 0

			y = solvers.solve_iterative_legacy(f, estimate=est, iter_stats=self.iter_stats)

		s = 2.0*y - self.s

		return y, s


class OtaOnePole(ZdfOnePoleBase):

	def __init__(self, wc, iter_stats=None, use_newton=True, verbose=False):
		super().__init__(wc)
		self.iter_stats = IterStats('1P OTA') if iter_stats is None else iter_stats
		self.use_newton = use_newton

		if verbose:
			print('OTA filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def process_tick_no_state_update(self, x):

		# y = g*tanh(x - y) + s
		# No good way to separate this
		# Can rearrange a little bit though:
		# y + g*tanh(y - x) - s = 0

		def f(y):
			return y + self.g*tanh(y - x) - self.s

		est = self.get_estimate(x)

		if self.use_newton:

			# Solve for
			# 0 = y + g*tanh(y-x) - s
			# d/dy = g*(sech(x-y))^2 + 1
			#      = g*(cosh(x-y))^-2 + 1

			def df(y):
				return self.g * math.pow(math.cosh(x - y), -2.0) + 1

			y = solvers.solve_nr_legacy(f, df, est, iter_stats=self.iter_stats)

		else:
			y = solvers.solve_iterative_legacy(f, estimate=est, iter_stats=self.iter_stats)

		s = 2.0*y - self.s

		return y, s


class OtaOnePoleNegative(ZdfOnePoleBase):

	def __init__(self, wc, iter_stats=None, use_newton=True, verbose=False):
		super().__init__(wc)
		self.iter_stats = IterStats('1P OTA Negative') if iter_stats is None else iter_stats
		self.use_newton = use_newton

		if verbose:
			print('OTA negative filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def get_estimate(self, x):

		# Calculate linear case
		# FIXME: this doesn't work
		#return (self.s - self.g*x) / (1.0 - self.g)
		#return self.m * (self.s - self.g*x)

		# integrator state
		return self.s

	def process_tick_no_state_update(self, x):

		# y = g*tanh(-x - y) + s

		est = self.get_estimate(x)

		def f(y):
			return y + self.g*tanh(x + y) - self.s

		if self.use_newton:

			# Solve for
			# 0 = y + g*tanh(y-x) - s
			# d/dy = g*(sech(x-y))^2 + 1
			#      = g*(cosh(x-y))^-2 + 1

			def df(y):
				return self.g * math.pow(math.cosh(x + y), -2.0) + 1

			y = solvers.solve_nr_legacy(f, df, est, iter_stats=self.iter_stats)

		else:
			y = solvers.solve_iterative_legacy(f, estimate=est, iter_stats=self.iter_stats)

		s = 2.0*y - self.s
		return y, s


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
	plt.legend(['Basic','Trapezoid'], loc=3)
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


def nonlin_filter(fc=0.1, f_saw=0.01, gain=2.0, n_samp=2048, use_newton=True):

	method_str = 'Newton-Raphson' if use_newton else 'Simple iteration'

	fig, (time_plot, freq_plot) = plt.subplots(2, 1)

	fig.suptitle(method_str)

	t = np.arange(n_samp)
	x = gen_saw(f_saw, n_samp) * gain * 0.5

	fft_x, f = do_fft(x, n_fft=n_samp, window=True)

	time_plot.plot(t, x, label='Input')
	freq_plot.semilogx(f, fft_x, label='Input')

	stats_ladder = IterStats('Ladder, ' + method_str)
	stats_ota = IterStats('OTA, ' + method_str)
	stats_ota_neg = IterStats('OTA Negative, ' + method_str)

	filters = [
		dict(filt=TrapzOnePole(fc), name='Linear', iter_stats=None),
		#dict(filt=TanhInputTrapzOnePole(fc), name='tanh input', iter_stats=None),
		dict(filt=LadderOnePole(fc, use_newton=use_newton, iter_stats=stats_ladder), name='Ladder'),
		dict(filt=OtaOnePole(fc, use_newton=use_newton, iter_stats=stats_ota), name='OTA'),
		dict(filt=OtaOnePoleNegative(fc, use_newton=use_newton, iter_stats=stats_ota_neg), name='-OTA', negate=True),
	]

	for filter in filters:
		filt = filter['filt']
		name = filter['name']

		negate = filter['negate'] if 'negate' in filter else False

		y = filt.process_buf(x)

		if negate:
			y = -y

		fft_y, f = do_fft(y, n_fft=n_samp, window=True)
		time_plot.plot(t, y, label=name)
		freq_plot.semilogx(f, fft_y, label=name)

	time_plot.legend()
	time_plot.set_xlim([0, 256])
	time_plot.grid()

	freq_plot.grid()
	
	for stats in [stats_ladder, stats_ota, stats_ota_neg]:
		stats.output()


def plot_impulse_response(fc=0.003, n_samp=4096, n_fft=None):
	if n_fft is None:
		n_fft = n_samp

	x = np.zeros(n_samp)
	x[0] = 1.0

	filt1 = BasicOnePole(fc)
	filt2 = TrapzOnePole(fc)

	y1 = filt1.process_buf(x)
	y2 = filt2.process_buf(x)

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


def plot(args=None):

	#plot_impulse_response(fc=0.001, n_samp=32768)
	#freq_sweep(fc=0.001, n_samp=2048)

	for use_newton in [False, True]:
		nonlin_filter(fc=0.1, f_saw=0.01, gain=4.0, n_samp=2048, use_newton=use_newton)

	plt.show()


def main(args=None):
	plot(args)
