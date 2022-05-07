#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from math import pi
import math

import filters.zdf.filters as filters
from generation.signal_generation import gen_sine, gen_saw
import solvers.solvers as solvers
from utils.utils import to_dB

solvers.legacy_max_num_iter = 20
solvers.legacy_eps = 1e-5

"""
Other future things to implement in single filter stage:
- With nonlinear buffers (e.g. BJT, FET, Darlington - also crossover distortion?)
- Asymmetry
- CV leakage
"""

"""
Test cases to determine best algorithms/approximations/estimates:

Sawtooth

Combination of sines
  One paper I found used 110 Hz + 155 Hz, which seems good (IM is at 75/200, HD2 at 220/310)

Variety of gain levels

Variety of input frequencies

Variety cutoff frequencies

Instant transisions vs bandlimited

Square waves
  good case because they have fast transitions and are always at one end or the
  other (in heavy distortion region), yet the distortion wouldn't affect the
  wave if it weren't for the lowpass filtering

Different stages
1st stage might have different optimal parameters from 4th stage

Different resonance levels, including self-osc

Audio-rate FM
"""


########## main ##########	


def plot(args=None):
	
	# Equivalent frequency at fs = 44.1 kHz:
	#fc = 0.3 # 13 kHz
	#fc = 0.1 # 4.4 kHz
	#fc = 0.03 # 1.3 kHz
	#fc = 0.01 # 441 Hz
	#fc = 0.003 # 132 Hz
	#fc = 0.001 # 44 Hz
	
	#plot_impulse_response(fc=fc, n_samp=32768)
	#freq_sweep(fc=fc, n_samp=2048)
	
	plot_nonlin_filter(fc=0.1, f_saw=0.01, gain=4.0, n_samp=2048)
	
	plot_lin_4pole(fc=0.1, f_saw=0.01, res=1.5, n_samp=2048)

	plt.show()


def main(args=None):
	plot(args)


########## Utility functions ##########	


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


def find_3dB_freq(freqs, Y):
	
	# Not the most "pythonic" way of doing this, but it works
	
	y_plus_3 = Y + 3.0
	
	prev_f = freqs[0]
	prev_y = y_plus_3[0]
	
	for f, y in zip(freqs, y_plus_3):
		
		if (y <= 0 and prev_y > 0) or (y >= 0 and prev_y < 0):
			zerox_loc = (0.0 - prev_y) / (y - prev_y)
			assert(zerox_loc >= 0.0 and zerox_loc <= 1.0)
			return (f * zerox_loc) + (prev_f * (1.0 - zerox_loc))
		
		prev_y = y
		prev_f = f

	return 0


########## Processing ##########	


def plot_impulse_response(fc=0.003, n_samp=4096, n_fft=None):
	
	if n_fft is None:
		n_fft = n_samp
	
	x = np.zeros(n_samp)
	x[0] = 1.0
	
	filt1 = filters.BasicOnePole(fc)
	filt2 = filters.TrapzOnePole(fc)
	
	y1 = filt1.process_buf(x)
	y2 = filt2.process_buf(x)
	
	Y1, f = do_fft(y1, n_fft=n_fft, window=False)
	Y2, _ = do_fft(y2, n_fft=n_fft, window=False)
	
	Y1 = to_dB(np.abs(Y1))
	Y2 = to_dB(np.abs(Y2))
	
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
	
	yAmp = math.sqrt(np.sum(np.square(y)))
	xAmp = math.sqrt(np.sum(np.square(x)))
	
	#print('yAmp = %f, xAmp = %f' % (yAmp, xAmp))
	
	amp = yAmp / xAmp
	
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
	
	filt1 = filters.BasicOnePole(fc)
	filt2 = filters.TrapzOnePole(fc)
	
	for n, sin_freq in enumerate(f):
		
		if sin_freq == 0:
			x = np.ones(n_samp)
		else:
			x = gen_sine(sin_freq, n_samp)
		
		filt1.z1 = 0.0
		filt2.s = 0.0
		
		y1 = filt1.process_buf(x)
		y2 = filt2.process_buf(x)
		
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


def plot_nonlin_filter(fc=0.1, f_saw=0.01, res=1.5, gain=2.0, n_samp=2048):
	
	filts = [
		{'name': '1P Linear', 'filt': filters.TrapzOnePole(fc), 'invert': False},
		{'name': '1P Tanh', 'filt': filters.TanhInputTrapzOnePole(fc), 'invert': False},
		{'name': '1P Ladder', 'filt': filters.LadderOnePole(fc), 'invert': False},
		{'name': '1P Ota', 'filt': filters.OtaOnePole(fc), 'invert': False},
		{'name': '1P Ota Negative', 'filt': filters.OtaOnePoleNegative(fc), 'invert': True},
		{'name': '4P Linear', 'filt': filters.LinearCascadeFilter(fc, res), 'invert': False},
		{'name': '4P Ladder', 'filt': filters.LadderFilter(fc, res), 'invert': False},
		{'name': '4P Ota', 'filt': filters.OtaFilter(fc, res), 'invert': False},
		{'name': '4P Ota Negative', 'filt': filters.OtaNegFilter(fc, res), 'invert': False},
		{'name': 'Basic SVF', 'filt': filters.BasicSvf(fc, res), 'invert': False},
		{'name': 'SVF Linear', 'filt': filters.SvfLinear(fc, res), 'invert': False},
		{'name': 'SVF Nonlin', 'filt': filters.NonlinSvf(fc, res, res_limit=None), 'invert': False},
		{'name': 'SVF Nonlin, res limit', 'filt': filters.NonlinSvf(fc, res, res_limit='hard'), 'invert': False},
		{'name': 'SVF Nonlin, tanh res', 'filt': filters.NonlinSvf(fc, res, res_limit=None, fb_nonlin=True), 'invert': False},
	]

	# FIXME: 'SVF Nonlin, tanh res' blows up if gain is higher

	x = gen_saw(f_saw, n_samp) * gain * 0.5
	
	X, f = do_fft(x, n_fft=n_samp, window=True)
	X = to_dB(np.abs(X))
	
	for filt in filts:
		
		y = filt['filt'].process_buf(x)
		
		if filt['invert']:
			y = -y
		
		Y, _ = do_fft(y, n_fft=n_samp, window=True)
		Y = to_dB(np.abs(Y))
		
		filt['y'] = y
		filt['Y'] = Y
	
	t = np.arange(n_samp)
	
	##### Plot filter responses #####

	def plot_filters(filter_idxs_to_plot):
		fig = plt.figure()
		fig.suptitle('f_in=%g, fc=%g, res=%g, gain=%g' % (f_saw, fc, res, gain))
		
		plt.subplot(2, 1, 1)
		
		plt.plot(t, x, '.-')
		legend = ['Input']
		
		for n in filter_idxs_to_plot:
			plt.plot(t, filts[n]['y'], '.-')
			legend += [filts[n]['name']]
		
		plt.legend(legend)
		
		plt.xlim([0, 256])
		plt.grid()
		
		plt.subplot(2, 1, 2)
		
		plt.semilogx(f, X)
		for n in filter_idxs_to_plot:
			plt.semilogx(f, filts[n]['Y'])
		
		plt.grid()

		#plt.subplot(3, 1, 3)
		#for n in filter_idxs_to_plot:
		#	plt.semilogx(f, filts[n]['Y'] - X)

		#plt.grid()
	
	# 1-pole
	#plot_filters([0, 2, 3])
	#plot_filters(range(4))
	
	# 4-pole
	plot_filters([5, 6, 7])
	#plot_filters([5, 6, 7, 8])

	# SVF
	plot_filters([9, 10, 11, 12, 13])


def plot_lin_4pole(fc=0.1, f_saw=0.01, res=0, n_samp=2048):
	
	filt_iterative = filters.LinearCascadeFilterIterative(fc)
	filt_solved = filters.LinearCascadeFilter(fc)
	
	filt_iterative.res = res
	filt_solved.res = res
	
	x = gen_saw(f_saw, n_samp) * 0.5
	
	y_iterative = filt_iterative.process_buf(x)
	y_solved    = filt_solved.process_buf(x)

	y_diff = y_iterative - y_solved
	
	amp_x, f         = do_fft(x, n_fft=n_samp, window=True)
	amp_iterative, _ = do_fft(y_iterative, n_fft=n_samp, window=True)
	amp_solved, _    = do_fft(y_solved, n_fft=n_samp, window=True)
	
	amp_x         = to_dB(np.abs(amp_x))
	amp_iterative = to_dB(np.abs(amp_iterative))
	amp_solved    = to_dB(np.abs(amp_solved))
	
	t = np.arange(n_samp)
	
	fig = plt.figure()
	fig.suptitle('f_in=%g, fc=%g, res=%g' % (f_saw, fc, res))
	
	plt.subplot(3, 1, 1)
	plt.plot(t, x, '.-', t, y_iterative, '.-', t, y_solved, '.-')
	plt.legend(['Input','Linear iterative','Linear solved'])
	plt.xlim([0, 256])
	plt.grid()
	
	plt.subplot(3, 1, 2)
	plt.semilogy(t, np.abs(y_diff), 'r.-')
	plt.xlim([0, 256])
	plt.grid()
	plt.ylabel('Diff')

	plt.subplot(3, 1, 3)
	plt.semilogx(f, amp_x, f, amp_iterative, f, amp_solved)
	plt.grid()

	print('Max difference between iterative & solved: %f' % np.max(np.abs(y_diff)))

	
if __name__ == "__main__":
	main()
