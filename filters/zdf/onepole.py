#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from math import pi
import math
from iter_stats import IterStats
from collections import Counter
import solvers

g_max_n_iter = 20

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

def main():
	
	# Equivalent frequency at fc = 44.1 kHz:
	#fc = 0.3 # 13 kHz
	#fc = 0.1 # 4.4 kHz
	#fc = 0.03 # 1.3 kHz
	#fc = 0.01 # 441 Hz
	#fc = 0.003 # 132 Hz
	fc = 0.001 # 4.4 Hz
	
	#impulse_response(fc=fc, n_samp=32768)
	#freq_sweep(fc=fc, n_samp=2048)
	
	for use_newton in [False, True]:
		nonlin_filter(fc=0.1, f_saw=0.01, gain=4.0, n_samp=2048, use_newton=use_newton)
	
	plt.show()


class BasicOnePole:

	def __init__(self, wc):
		self.a1 = math.exp(-2.0*pi * wc)
		self.b0 = 1.0 - self.a1
		self.z1 = 0.0
		
		print('Basic filter: wc=%f, a1=%f, b0=%f' % (wc, self.a1, self.b0))
	
	def process(self, input_sig):
		
		y = np.zeros_like(input_sig)
		
		for n, x in enumerate(input_sig):
			self.z1 = self.b0*x + self.a1*self.z1
			y[n] = self.z1
		
		return y


class TrapzOnePole:
	
	def __init__(self, wc):

		pi_wc = pi*wc
		self.g = math.tan(pi_wc)
		self.s = 0.0
		self.multiplier = 1.0 / (self.g + 1.0)

		print('Trapezoid filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi_wc))
	
	def process(self, input_sig):
		
		y = np.zeros_like(input_sig)
		
		for n, x in enumerate(input_sig):
			
			# y = g*(x - y) + s
			# y + g*y = g*x + s
			# y = (g*x + s) / (g + 1)
			
			# if m = 1/(1+g)
			# y = m * (g*x + s)
			
			yn = self.multiplier * (self.g*x + self.s)
			
			self.s = 2.0*yn - self.s
			y[n] = yn
		
		return y


class TanhInputTrapzOnePole:
	
	def __init__(self, wc):
		
		pi_wc = pi*wc
		self.g = math.tan(pi_wc)
		self.s = 0.0
		self.multiplier = 1.0 / (self.g + 1.0)
		
		print('Trapezoid filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi_wc))
	
	def process(self, input_sig):
		
		y = np.zeros_like(input_sig)
		tanh_input_sig = np.tanh(input_sig)
		
		for n, x in enumerate(tanh_input_sig):
			
			# y = g*(x - y) + s
			# y + g*y = g*x + s
			# y = (g*x + s) / (g + 1)
			
			# if m = 1/(1+g)
			# y = m * (g*x + s)
			
			yn = self.multiplier * (self.g*x + self.s)
			
			self.s = 2.0*yn - self.s
			y[n] = yn
		
		return y


class LadderOnePole:
	
	def __init__(self, wc, use_newton=True, stats=None):
		
		self.use_newton = use_newton
		self.stats = stats

		pi_wc = pi*wc
		self.g = math.tan(pi_wc)
		self.s = 0.0
		self.multiplier = 1.0 / (self.g + 1.0)

		print('Ladder filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi_wc))
	
	def process(self, input_sig):
		
		y = np.zeros_like(input_sig)
		
		tanh_input_sig = np.tanh(input_sig)
		
		for n, tanh_x in enumerate(tanh_input_sig):

			# y = g*(tanh(x) - tanh(y)) + s
			# y + g*(tanh(y) - tanh(x)) - s = 0
			f  = lambda y: y + self.g*(math.tanh(y) - tanh_x) - self.s
			
			# Initial estimate: calculate linear case
			est = self.multiplier * (self.g*tanh_x + self.s)
			
			# Initial estimate: integrator state
			#est = self.s
			
			if self.use_newton:
				
				# 0 = y + g*(tanh(y) - tanh(x)) - s
				# d/dy = -g*(sech(y))^2 - 1
				#      = -g*(cosh(y))^-2 - 1
			
				df = lambda y: self.g * math.pow(math.cosh(y), -2.0) + 1
				
				yn = solvers.newton_raphson(f, df, est, stats=self.stats)
		
			else:
				
				# y = g*(tanh(x) - tanh(y)) + s
				# y + g*tanh(y) = g*tanh(x) + s
				
				# y + g*tanh(y) = rhs
				# y + g*tanh(y) - rhs = 0
				
				yn = solvers.iterative(f, initial_estimate=est, stats=self.stats)
			
			self.s = 2.0*yn - self.s
			y[n] = yn
		
		return y


class OtaOnePole:
	
	def __init__(self, wc, use_newton=True, stats=None):

		self.use_newton = use_newton
		self.stats = stats

		pi_wc = pi*wc
		self.g = math.tan(pi_wc)
		self.s = 0.0
		self.multiplier = 1.0 / (self.g + 1.0)

		print('OTA filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi_wc))
	
	def process(self, input_sig):
		
		y = np.zeros_like(input_sig)
		
		for n, x in enumerate(input_sig):

			# y = g*tanh(x - y) + s
			
			# No good way to separate this
			# Can rearrange a little bit though
			
			# y = g*tanh(x - y) + s
			# y - g*tanh(x - y) = s
			# y + g*tanh(y - x) = s
			# y + g*tanh(y - x) - s = 0
			
			# Initial estimate: calculate linear case
			est = self.multiplier * (self.g*x + self.s)
			
			# Initial estimate: integrator state
			#est = self.s
			
			f  = lambda y: y + self.g*math.tanh(y - x) - self.s
			
			if self.use_newton:
				
				# Solve for
				# 0 = y + g*tanh(y-x) - s
				# d/dy = g*(sech(x-y))^2 + 1
				#      = g*(cosh(x-y))^-2 + 1
			
				df = lambda y: self.g * math.pow(math.cosh(x - y), -2.0) + 1
				
				yn = solvers.newton_raphson(f, df, est, stats=self.stats)
			
			else:
			
				yn = solvers.iterative(f, initial_estimate=est, stats=self.stats)
				
			self.s = 2.0*yn - self.s
			y[n] = yn
		
		return y

class OtaOnePoleNegative:
	
	def __init__(self, wc, use_newton=True, stats=None):
		
		self.use_newton = use_newton
		self.stats = stats

		pi_wc = pi*wc
		self.g = math.tan(pi_wc)
		self.s = 0.0
		self.multiplier = 1.0 / (1.0 - self.g)

		print('OTA negative filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi_wc))
	
	def process(self, input_sig):
		
		y = np.zeros_like(input_sig)
		
		for n, x in enumerate(input_sig):

			# y = g*tanh(x - y) + s
			
			# No good way to separate this
			# Can rearrange a little bit though
			
			# y = g*tanh(-(x + y)) + s
			# y - g*tanh(-(x + y)) = s
			# y + g*tanh(x + y) = s
			# y + g*tanh(x + y) - s = 0
			
			
			# Linear:
			# y = g*(-x - y) + s
			# y = -gx + gy + s
			# y - gy = -gx + s
			# y(1 - g) = -gx + s
			# y = (-gx + s) / (1 - g)
			
			# Initial estimate: calculate linear case
			# FIXME: this doesn't work
			#est = (self.s - self.g*x) / (1.0 - self.g)
			#est = self.multiplier * (self.s - self.g*x)
			
			# Initial estimate: integrator state
			est = self.s
			
			f  = lambda y: y + self.g*math.tanh(x + y) - self.s
			
			if self.use_newton:
				
				# Solve for
				# 0 = y + g*tanh(y-x) - s
				# d/dy = g*(sech(x-y))^2 + 1
				#      = g*(cosh(x-y))^-2 + 1
			
				df = lambda y: self.g * math.pow(math.cosh(x + y), -2.0) + 1
				
				yn = solvers.newton_raphson(f, df, est, stats=self.stats)
			
			else:
			
				yn = solvers.iterative(f, initial_estimate=est, stats=self.stats)
				
			self.s = 2.0*yn - self.s
			y[n] = yn
		
		return y


def gen_phase(freq, n_samp, startPhase=0.0):
	
	if (freq <= 0.0) or (freq >= 0.5):
		print("Warning: freq out of range %f" % freq)

	# This could be vectorized, but that's hard to do without internet access right now ;)
	ph = np.zeros(n_samp)
	phase = startPhase
	for n in range(n_samp):
		phase += freq
		ph[n] = phase
	
	ph = np.mod(ph, 1.0)
	
	return ph


def gen_saw(freq, len):
	return gen_phase(freq, len) - 0.5


def gen_sine(freq, len):
	"""
	y = np.zeros(len)
	prev = 0.0
	for n in range(len):
		y[n] = prev
		prev += freq
	"""
	y = gen_phase(freq, len)
	y *= 2.0 * pi
	return np.sin(y)


def do_fft(x, n_fft, window=False):
	
	if window:
		y = np.fft.fft(x * np.hamming(len(x)), n=n_fft)
	else:
		y = np.fft.fft(x, n=n_fft)
	
	f = np.fft.fftfreq(n_fft, 1.0)
	
	# Only take first half
	y = y[0:len(y)//2]
	f = f[0:len(f)//2]
	
	return y, f


def to_dB(x):
	return 20.0*np.log10(np.abs(x))


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
	
	filt1 = BasicOnePole(fc)
	filt2 = TrapzOnePole(fc)
	
	y1 = filt1.process(x)
	y2 = filt2.process(x)
	
	Y1, f = do_fft(y1, n_fft=n_fft, window=False)
	Y2, _ = do_fft(y2, n_fft=n_fft, window=False)
	
	Y1 = to_dB(Y1)
	Y2 = to_dB(Y2)
	
	fc1 = find_3dB_freq(f, Y1)
	fc2 = find_3dB_freq(f, Y2)
	
	print('Ideal fc = %.4f' % fc)
	print('Basic fc = %.4f (error %.2f%%)' % (fc1, abs(fc1-fc)/fc * 100.0))
	print('Trapz fc = %.4f (error %.2f%%)' % (fc2, abs(fc2-fc)/fc * 100.0))
	
	plt.figure()
	
	plt.subplot(211)
	
	plt.semilogx(f, Y1, f, Y2)
	plt.legend(['Basic','Trapezoid'], loc=3)
	plt.title('fc = %f' % fc)
	plt.grid()
	plt.ylim(-12, 3)
	
	plt.subplot(212)
	
	plt.semilogx(f, Y1, f, Y2)
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
	Y1 = np.zeros_like(f)
	Y2 = np.zeros_like(f)
	ph1 = np.zeros_like(f)
	ph2 = np.zeros_like(f)
	
	filt1 = BasicOnePole(fc)
	filt2 = TrapzOnePole(fc)
	
	for n, sinFreq in enumerate(f):
		
		if sinFreq == 0:
			x = np.ones(n_samp)
		else:
			x = gen_sine(sinFreq, n_samp)
		
		filt1.z1 = 0.0
		filt2.s = 0.0
		
		y1 = filt1.process(x)
		y2 = filt2.process(x)
		
		Y1[n], ph1[n] = find_amp_phase(y1, x)
		Y2[n], ph2[n] = find_amp_phase(y2, x)
	
	Y1 = 20*np.log10(Y1)
	Y2 = 20*np.log10(Y2)
	
	plt.figure()
	
	plt.subplot(211)
	
	plt.semilogx(f, Y1, f, Y2)
	plt.legend(['Basic','Trapezoid'], loc=3)
	plt.title('fc = %f' % fc)
	plt.grid()
	plt.ylim(-12, 3)
	
	plt.subplot(212)
	
	plt.semilogx(f, ph1, f, ph2)
	plt.grid()


def nonlin_filter(fc=0.1, f_saw=0.01, gain=2.0, n_samp=2048, use_newton=True):

	method_str = 'Newton-Raphson' if use_newton else 'Simple iteration'

	stats_ladder = IterStats('Ladder, ' + method_str)
	stats_ota = IterStats('OTA, ' + method_str)
	stats_ota_neg = IterStats('OTA Negative, ' + method_str)

	filt1 = TrapzOnePole(fc)
	filt2 = TanhInputTrapzOnePole(fc)
	filt3 = LadderOnePole(fc, use_newton=use_newton, stats=stats_ladder)
	filt4 = OtaOnePole(fc, use_newton=use_newton, stats=stats_ota)
	filt5 = OtaOnePoleNegative(fc, use_newton=use_newton, stats=stats_ota_neg)
	
	x = gen_saw(f_saw, n_samp) * gain
	
	y1 = filt1.process(x)
	y2 = filt2.process(x)
	y3 = filt3.process(x)
	y4 = filt4.process(x)
	y5 = filt5.process(x)
	
	y5 = -y5
	
	t = np.arange(n_samp)
	
	X, f  = do_fft(x, n_fft=n_samp, window=True)
	Y1, _ = do_fft(y1, n_fft=n_samp, window=True)
	Y2, _ = do_fft(y2, n_fft=n_samp, window=True)
	Y3, _ = do_fft(y3, n_fft=n_samp, window=True)
	Y4, _ = do_fft(y4, n_fft=n_samp, window=True)
	Y5, _ = do_fft(y5, n_fft=n_samp, window=True)
	
	X = to_dB(X)
	Y1 = to_dB(Y1)
	Y2 = to_dB(Y2)
	Y3 = to_dB(Y3)
	Y4 = to_dB(Y4)
	Y5 = to_dB(Y5)
	
	fig = plt.figure()
	fig.suptitle(method_str)
	
	plt.subplot(211)
	plt.plot(t, x, t, y1, t, y3, t, y4)
	plt.legend(['Input','Linear','Ladder','OTA'])
	#plt.plot(t, x, t, y1, t, y2, t, y3, t, y4, t, y5)
	#plt.legend(['Input','Linear','tanh input','Ladder','OTA', '-OTA'])
	plt.xlim([0, 256])
	plt.grid()
	
	plt.subplot(212)
	plt.semilogx(f, X, f, Y1, f, Y3, f, Y4)
	#plt.semilogx(f, X, f, Y1, f, Y2, f, Y3, f, Y4, f, Y5)
	plt.grid()
	
	for stats in [stats_ladder, stats_ota, stats_ota_neg]:
		stats.output()


if __name__ == "__main__":
	main()