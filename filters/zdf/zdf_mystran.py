#!/usr/bin/env python3

"""
Ladder filter ported from Teemu Voipo's C code
http://www.kvraudio.com/forum/viewtopic.php?f=33&t=349859
"""

# Original C header:
""""
//// LICENSE TERMS: Copyright 2012 Teemu Voipio
//
// You can use this however you like for pretty much any purpose,
// as long as you don't claim you wrote it. There is no warranty.
//
// Distribution of substantial portions of this code in source form
// must include this copyright notice and list of conditions.
""" 

from math import pi, cos, sin, atan2, sqrt, floor, ceil, tan, tanh
import numpy as np
import matplotlib.pyplot as plt


# Vectorized
def tanhXdX(x):
	a = np.square(x)
	numer = (a+105.0)*a + 945.0
	denom = (15.0*a + 420.0)*a + 945.0
	return numer / denom

	
class TransistorLadder:
	
	def __init__(self):
		self.zi = 0
		self.s = np.zeros(4)
		
		self.recover_gain = True
	
	def process(self, input, cutoff, resonance):
		
		# Prewarp
		f = tan(pi * cutoff)
		
		r = (40.0 / 9.0) * resonance
		
		n_samp = len(input)
		
		out = np.zeros(n_samp)
		
		for n, samp in enumerate(input):
			
			# Input with half delay, for nonlinearities
			ih = 0.5 * (samp + self.zi)
			self.zi = samp
			
			# Evaluate the nonlinear gains
			t0 = tanhXdX(ih - r * self.s[3])
			t1 = tanhXdX(self.s[0])
			t2 = tanhXdX(self.s[1])
			t3 = tanhXdX(self.s[2])
			t4 = tanhXdX(self.s[3])
			
			# g# the denominators for solutiosn of individual stages
			g0 = 1 / (1 + f*t1)
			g1 = 1 / (1 + f*t2)
			g2 = 1 / (1 + f*t3)
			g3 = 1 / (1 + f*t4)
			
			# f# are just factored out of the feedback solution
			f3 = f*t3*g3
			f2 = f*t2*g2*f3
			f1 = f*t1*g1*f2
			f0 = f*t0*g0*f1
			
			# solve feedback
			y3 = g3*self.s[3] + f3*g2*self.s[2] + f2*g1*self.s[1] + f1*g0*self.s[0] + f0*samp
			y3 /= (1 + r*f0)
			
			# Then solve the ramining outputs (with the nonlinear gains here)
			xx = t0*(samp - r*y3)
			y0 = t1*g0*(self.s[0]+ f*xx)
			y1 = t2*g1*(self.s[1] + f*y0)
			y2 = t3*g2*(self.s[2] + f*y1)
			
			# Update state
			self.s[0] += 2*f * (xx - y0)
			self.s[1] += 2*f * (y0 - y1)
			self.s[2] += 2*f * (y1 - y2)
			self.s[3] += 2*f * (y2 - t4*y3)
			
			out[n] = y3
		
		if self.recover_gain:
			out *= (1 + r)
		
		return out


def gen_phase(freq, n_samp, start_phase=0.0):
	
	if (freq <= 0.0) or (freq >= 0.5):
		print("Warning: freq out of range %f" % freq)

	# This could be vectorized, but that's hard to do without internet access right now ;)
	ph = np.zeros(n_samp)
	phase = start_phase
	for n in range(n_samp):
		phase += freq
		ph[n] = phase
	
	ph = np.mod(ph, 1.0)
	
	return ph


def atodb(Y, norm=True):
	Y = 20*np.log10(abs(Y))
	if norm:
		Y -= np.max(Y)
	return Y


def fft(x, window=True):
	if window:
		return np.fft.fft(x * np.hamming(len(x)))
	else:
		return np.fft.fft(x)


def plot_wave(n_samp=2048):
	tl = TransistorLadder()
	
	gain = 2.0
	
	freq = 0.01
	fc = 0.1
	resonance = 0.25
	
	ph = gen_phase(freq, n_samp)
	
	x = ph - 0.5
	y = tl.process(x*gain, fc, resonance)
	
	t = np.arange(n_samp)
	
	X = atodb(fft(x, window=True), norm=False)
	Y = atodb(fft(y, window=True), norm=False)
	f = np.arange(n_samp) / n_samp
	
	plt.figure()
	
	plt.subplot(211)
	plt.plot(t, x, t, y)
	plt.grid()
	
	plt.subplot(212)
	#plt.plot(f, Y)
	plt.semilogx(f[0:n_samp//2], X[0:n_samp//2], f[0:n_samp//2], Y[0:n_samp//2])
	plt.grid()
	#plt.xlim([0, 0.5])


def plot_step(n_samp=512, fc=0.05, resonance=0.25, amplitude=0.1, n_iter=4):
	n = np.arange(-n_samp//2,n_samp//2)
	x = np.concatenate((np.zeros(n_samp//2), np.zeros(n_samp//2) + 1.0))
	
	tl = TransistorLadder()
	
	y_lad = tl.process(x*amplitude, fc, resonance)/amplitude
	
	plt.figure()

	plt.plot(n, x, 'r-', n, y_lad, 'b-')
	plt.title('Step response')
	plt.legend(['Input', 'Ladder'], loc='upper left')
	plt.grid()
	if(resonance < 0.9):
		plt.ylim([-0.25, 2.0])
	plt.xlim([-n_samp/2, n_samp/2])

	plt.grid()


def freq_sweep(n_freq=128, n_samp=1024, fc=0.1, resonance=0.25, amplitude=0.1, n_iter=4):
	
	freqs = 0.5 * np.arange(n_freq) / n_freq
	#freqs = np.logspace(np.log10(0.001), np.log10(0.25), n_freq)
	
	Y_lad = np.zeros(n_freq)
	Y_ota = np.zeros(n_freq)
	
	for n, f in enumerate(freqs):
		
		x = (amplitude * np.sin(2*pi*gen_phase(f, n_samp))) if (f != 0) else (amplitude + np.zeros(n_samp))
		
		tl = TransistorLadder()
		
		y_lad = tl.process(x, fc, resonance)
		
		x_rms = sqrt(np.mean(np.square(x)))
		y_lad_rms = sqrt(np.mean(np.square(y_lad)))
		
		Y_lad[n] = y_lad_rms/x_rms if (x_rms != 0.0) else 1.0
	
	Y_lad = atodb(Y_lad, norm=False)
	
	plt.figure()
	plt.semilogx(freqs, Y_lad, '.-')
	plt.ylim([-40,20])
	plt.grid()
	plt.title("fc=%.2f, Res=%.2f" % (fc, resonance))


def plot(args=None):
	
	n_iter = 8
	resonance=0.8
	amp=0.1
	
	plot_wave()
	plot_step(n_iter=n_iter, resonance=resonance, amplitude=amp)
	#freq_sweep(n_iter=n_iter, resonance=resonance, amplitude=amp)
	
	plt.show()


def main(args=None):
	plot()


if __name__ == "__main__":
	main()
