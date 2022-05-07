#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from math import pi, tanh, pow
import math
from collections import Counter

from iter_stats import IterStats
import solvers
import filters

solvers.max_n_iter = 20
solvers.eps = 1e-5

"""
#eps = 1e-7 # -140 dB
#eps = 1e-6 # -120 dB
eps = 1e-5 # -100 dB
#eps = 1e-4 # -80 dB
"""

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

def main():
	
	# Equivalent frequency at fs = 44.1 kHz:
	#fc = 0.3 # 13 kHz
	#fc = 0.1 # 4.4 kHz
	#fc = 0.03 # 1.3 kHz
	#fc = 0.01 # 441 Hz
	#fc = 0.003 # 132 Hz
	#fc = 0.001 # 44 Hz
	
	#plot_impulse_response(fc=fc, n_samp=32768)
	#freq_sweep(fc=fc, n_samp=2048)
	
	plot_nonlin_filter(fc=0.1, fSaw=0.01, gain=4.0, n_samp=2048)
	
	plot_lin_4pole(fc=0.1, fSaw=0.01, res=1.5, n_samp=2048)

	plt.show()

"""
stats_1PLadder = IterStats('1P Ladder')
stats_1POta = IterStats('1P OTA')
stats_1POtaNeg = IterStats('1P OTA Negative')

stats_LinearOuter = IterStats('Linear outer loop')

stats_LadderPoles = [IterStats('Ladder pole %i' % (i+1)) for i in range(4)]
stats_LadderOuter = IterStats('Ladder outer loop')

stats_OtaPoles = [IterStats('OTA pole %i' % (i+1)) for i in range(4)]
stats_OtaOuter = IterStats('OTA outer loop')

stats_OtaNegPoles = [IterStats('OTA neg pole %i' % (i+1)) for i in range(4)]
stats_OtaNegOuter = IterStats('OTA neg outer loop')
"""

########## Utility functions ##########	

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
	
	for n, sinFreq in enumerate(f):
		
		if sinFreq == 0:
			x = np.ones(n_samp)
		else:
			x = gen_sine(sinFreq, n_samp)
		
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


def plot_nonlin_filter(fc=0.1, fSaw=0.01, res=1.5, gain=2.0, n_samp=2048):
	
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
	
	x = gen_saw(fSaw, n_samp) * gain
	
	X, f = do_fft(x, n_fft=n_samp, window=True)
	X = to_dB(X)
	
	for filt in filts:
		
		y = filt['filt'].process_buf(x)
		
		if filt['invert']:
			y = -y
		
		Y, _ = do_fft(y, n_fft=n_samp, window=True)
		Y = to_dB(Y)
		
		filt['y'] = y
		filt['Y'] = Y
	
	t = np.arange(n_samp)
	
	##### Plot filter responses #####

	def plot_filters(filtsToPlot):
		fig = plt.figure()
		fig.suptitle('f_in=%g, fc=%g, res=%g, gain=%g' % (fSaw, fc, res, gain))
		
		plt.subplot(2, 1, 1)
		
		plt.plot(t, x, '.-')
		legend = ['Input']
		
		for n in filtsToPlot:
			plt.plot(t, filts[n]['y'], '.-')
			legend += [filts[n]['name']]
		
		plt.legend(legend)
		
		plt.xlim([0, 256])
		plt.grid()
		
		plt.subplot(2, 1, 2)
		
		plt.semilogx(f, X)
		for n in filtsToPlot:
			plt.semilogx(f, filts[n]['Y'])
		
		plt.grid()

		#plt.subplot(3, 1, 3)
		#for n in filtsToPlot:
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

	##### Print/plot convergence stats #####
	
	if False:
		plt.figure()
		
		stats_1PLadder.output(bNewFig=False)
		stats_1POta.output(bNewFig=False)
		stats_1POtaNeg.output(bNewFig=False)
		
		plt.legend(['Ladder','OTA','OTA neg'])
		plt.title('1-Pole Convergence')
		plt.xlabel('# iterations')
		plt.grid()
	
	#stats_LinearOuter.output(bNewFig=True)

	def PlotCascadeStats(filt, name):

		plt.figure()

		plt.subplot(211)

		filt.plot_outer.output(bNewFig=False)

		plt.title(name + ' Convergence')
		plt.grid()

		plt.subplot(212)

		for pole in filt.poles:
			pole.stats.output(bNewFig=False)

		plt.legend(['P1', 'P1', 'P3', 'P4'])
		plt.xlabel('# iterations')
		plt.grid()

		plt.figure()
		plt.subplot(121)

		filt.plot_outer.output(bPrint=False, bPlotIter=False, bPlotEst=True, bNewFig=False)

		plt.title(name + ' estimate')
		plt.xlabel('Estimate')
		plt.grid()

		plt.subplot(122)

		for pole in filt.poles:
			pole.stats.output(bPrint=False, bPlotIter=False, bPlotEst=True, bNewFig=False)

		plt.title(name + ' pole estimates')
		plt.legend(['P1', 'P1', 'P3', 'P4'])
		plt.xlabel('Estimate')
		plt.ylabel('Final')
		plt.grid()

		plt.figure()
		plt.subplot(121)

		filt.plot_outer.output(bPrint=False, bPlotIter=False, bPlotEst=False, bPlotErr=True, bNewFig=False)

		plt.title(name + ' estimate error')
		plt.ylabel('Estimate Error')
		plt.grid()
		plt.xlim(0, 255)

		plt.subplot(122)

		for pole in filt.poles:
			pole.stats.output(bPrint=False, bPlotIter=False, bPlotEst=False, bPlotErr=True, bNewFig=False)

		plt.title(name + ' pole estimate errors')
		plt.legend(['P1', 'P1', 'P3', 'P4'])
		plt.grid()
		plt.xlim(0, 255)

	#PlotCascadeStats(filts[6]['filt'], 'Ladder')
	#PlotCascadeStats(filts[7]['filt'], 'OTA')
	#PlotCascadeStats(filts[8]['filt'], 'OTA neg')


def plot_lin_4pole(fc=0.1, fSaw=0.01, res=0, n_samp=2048):
	
	filt_iterative = filters.LinearCascadeFilterIterative(fc)
	filt_solved = filters.LinearCascadeFilter(fc)
	
	filt_iterative.res = res
	filt_solved.res = res
	
	x = gen_saw(fSaw, n_samp)
	
	y_iterative = filt_iterative.process_buf(x)
	y_solved    = filt_solved.process_buf(x)

	y_diff = y_iterative - y_solved
	
	amp_x, f         = do_fft(x, n_fft=n_samp, window=True)
	amp_iterative, _ = do_fft(y_iterative, n_fft=n_samp, window=True)
	amp_solved, _    = do_fft(y_solved, n_fft=n_samp, window=True)
	
	amp_x         = to_dB(amp_x)
	amp_iterative = to_dB(amp_iterative)
	amp_solved    = to_dB(amp_solved)
	
	t = np.arange(n_samp)
	
	fig = plt.figure()
	fig.suptitle('f_in=%g, fc=%g, res=%g' % (fSaw, fc, res))
	
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
