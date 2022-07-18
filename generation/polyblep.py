#!/usr/bin/env python3


"""
PolyBLEP synthesis
Based on "Antialiasing Oscillators in Subtractive Synthesis" by Välimäki & Huovilainen, 2007
"""

import numpy as np
import matplotlib.pyplot as plt
from generation.signal_generation import gen_phase
from utils import utils


@np.vectorize
def polyblep(t):
	"""
	PolyBLEP - polynomial bandlimited step
	"""

	# For optimized implementations, could assume input is bounded [-1.0,1.0]
	if t > 1.0:
		return 1.0
	elif t < -1.0:
		return 0.0
	elif t > 0.0:
		return -0.5*(t*t) + t + 0.5
	else:
		return  0.5*(t*t) + t + 0.5


@np.vectorize
def diff_polyblep(t):
	"""
	Difference between naive step and PolyBLEP
	"""

	# For optimized implementations, could assume input is bounded [-1.0,1.0]
	if (t > 1.0) or (t < -1.0):
		return 0.0
	if t > 0.0:
		return -0.5*(t*t) + t - 0.5
	else:
		return  0.5*(t*t) + t + 0.5


@np.vectorize
def integral_polyblep(t, C=1.0):
	if t < -1.0:
		return 0.5*t + C
	elif t > 1.0:
		return -0.5*t + C
	elif t > 0.0:
		return  1/6*(t*t*t) - 0.5*(t*t) - 1/6 + C
	else:
		return -1/6*(t*t*t) - 0.5*(t*t) - 1/6 + C


@np.vectorize
def integral_diff_polyblep(t):
	if (t > 1.0) or (t < -1.0):
		return 0.0
	elif t > 0.0:
		return -1/6*(t*t*t) + 0.5 * (t*t) - 0.5*t + 1/6
	else:
		return  1/6*(t*t*t) + 0.5 * (t*t) + 0.5*t + 1/6


def saw_polyblep(freq, n_samp, start_phase=0.0, size=1.0, advanced_poly=False):
	"""
	Advanced PolyBLEP:
	Scales PolyBLEP amount according to phase
	Only applies to saw
	Effect is negligible except at very high frequencies or lots of oversampling
	Probably not worth it
	"""
	y = np.zeros(n_samp)
	# pbs = np.zeros(n_samp)
	phase_vec = gen_phase(freq, n_samp, start_phase)

	f_over = freq * size

	for n, phase in enumerate(phase_vec):

		pb = 0.0
		if phase < f_over:
			pb = -diff_polyblep(phase / f_over)
			if advanced_poly:
				pb *= (1.0 - phase)

		elif phase > (1.0 - f_over):
			pb = -diff_polyblep((phase - 1.0) / f_over)
			if advanced_poly:
				pb *= phase

		y[n] = phase - 0.5 + pb
		# pbs[n] = pb

	y *= 2.0

	return y, phase_vec


def square_polyblep(freq, n_samp, start_phase=0.0, size=1.0):
	y = np.zeros(n_samp)
	phase_vec = gen_phase(freq, n_samp, start_phase)

	p_freq = 0.5 * freq * size

	for n, ph in enumerate(phase_vec):

		# This could be optimized, but is written for clarity over performance

		if ph < p_freq:
			# 2nd half of high-low transition
			t = utils.scale(ph, (0.0, p_freq), (0.0, 1.0))
			p = polyblep(t)
			yn = utils.scale(p, (0.0, 1.0), (1.0, -1.0))

		elif ph < (0.5 - p_freq):
			# Low
			yn = -1.0

		elif ph < 0.5:
			# 1st half of low-high transition
			t = utils.scale(ph, (0.5 - p_freq, 0.5), (-1.0, 0.0))
			p = polyblep(t)
			yn = utils.scale(p, (0.0, 1.0), (-1.0, 1.0))

		elif ph < 0.5 + p_freq:
			# 2nd half of low-high transition
			t = utils.scale(ph, (0.5, 0.5 + p_freq), (0.0, 1.0))
			p = polyblep(t)
			yn = utils.scale(p, (0.0, 1.0), (-1.0, 1.0))

		elif ph < 1.0 - p_freq:
			# High, no polyblep
			yn = 1.0

		elif ph < 1.0:
			# 1st half of high-low transition
			t = utils.scale(ph, (1.0 - p_freq, 1.0), (-1.0, 0.0))
			p = polyblep(t)
			yn = utils.scale(p, (0.0, 1.0), (1.0, -1.0))

		else:
			raise AssertionError('Invalid phase')

		y[n] = yn

	return y


def plot_polyblep(n_pts=201):
	x = np.linspace(-2.0, 2.0, n_pts)
	d = np.zeros(n_pts)
	y = np.zeros(n_pts)
	i = np.zeros(n_pts)
	id = np.zeros(n_pts)

	d = diff_polyblep(x)
	y = polyblep(x)
	i = integral_polyblep(x)
	id = integral_diff_polyblep(x)

	# id = 1.0 - 0.5*np.abs(xx) - i

	plt.figure()

	plt.plot(x, y, label='Step')
	plt.plot(x, d, label='DiffStep')
	plt.plot(x, i, label='Integral')
	plt.plot(x, id, label='IntegralDiff')

	plt.title('PolyBLEPs')
	plt.legend()
	plt.grid()


def _fft(vals):
	return np.abs(np.fft.fft(vals * np.hamming(len(vals))))


def _to_dB_norm(vals):
	vals = utils.to_dB(vals)
	vals -= np.amax(vals)
	return vals


def plot_full(saw=True, plot_advanced=False):
	fs = 44100
	f = 440
	n_samp = 1024 * 4
	oversamp = 2
	#t = np.linspace(0.0, 0.01, fs)
	t = np.arange(n_samp)
	w = f / fs

	if saw:
		y, ph = saw_polyblep(w, n_samp, size=oversamp)
		if plot_advanced:
			y_adv, _ = saw_polyblep(w, n_samp, size=oversamp, advanced_poly=True)
		
		naive = ph * 2.0 - 1.0
	else:
		ph = gen_phase(w, n_samp)
		y = square_polyblep(w, n_samp, size=oversamp)
		naive = (ph >= 0.5) * 2.0 - 1.0

	polyblep_mag = _to_dB_norm(_fft(y))

	if plot_advanced:
		adv_mag = _to_dB_norm(_fft(y_adv))

	naive_mag = _to_dB_norm(_fft(naive))

	f = np.arange(n_samp) / n_samp

	plt.figure()

	plt.subplot(211)
	if plot_advanced:
		plt.plot(t, y, '.-', label=f'PolyBLEP, size={oversamp:.0f}')
		plt.plot(t, y_adv, '.-', label='Advanced PolyBLEP')
	else:
		plt.plot(t, naive, '.-', label='Naive')
		plt.plot(t, y, '.-', label=f'PolyBLEP, size={oversamp:.0f}')

	plt.legend()
	plt.title('Naive vs PolyBLEP')
	plt.grid()
	plt.xlim([0, 512])
	plt.ylim([-1.1, 1.1])

	plt.subplot(212)

	f_non_alias = np.arange(0.5 * n_samp / oversamp) / n_samp
	f_alias = np.arange(0.5 * n_samp / oversamp, n_samp) / n_samp
	Y_non_alias = polyblep_mag[0:(n_samp // (2*oversamp))]
	Y_alias = polyblep_mag[(n_samp // (2*oversamp)):n_samp]

	print(len(f_non_alias), len(Y_non_alias), len(f_alias), len(Y_alias))
	print(np.min(f_non_alias), np.max(f_non_alias), np.min(f_alias), np.max(f_alias))

	plt.plot(f, naive_mag, label='Naive')
	plt.plot(f_non_alias, Y_non_alias, label='PolyBLEP')
	plt.plot(f_alias, Y_alias, label='Aliased')
	plt.legend()
	plt.xlim([0, 0.5])
	plt.xticks([1 / 32.0, 1 / 16.0, 1 / 8.0, 1 / 4.0, 1 / 2.0])

	# plt.plot(f, naive_mag, '.-', f, polyblep_mag, '.-', f, adv_mag, '.-')
	# plt.legend(['Naive','PolyBLEP','Advanced'])
	# plt.xlim([0, 1/4.0])
	# plt.xticks([1/32.0, 1/16.0, 1/8.0, 1/4.0])

	# plt.plot(f, polyblep_mag, '.', f, adv_mag, '.')
	# plt.legend(['PolyBLEP','Advanced'])
	# plt.xlim([0, 0.5])
	# plt.xticks([1/32.0, 1/16.0, 1/8.0, 1/4.0, 1/2.0])

	plt.grid()
	plt.ylim([-80, 0])


def plot_cycle(fft_size=512, plot_phase=False):
	oversamp = 4

	actual_len = fft_size // oversamp

	incr = 2.0 / fft_size

	x = np.arange(0, fft_size)
	x2 = np.arange(0, fft_size, oversamp)
	y = x / fft_size + 0.5 + (incr / 2.0)
	y = np.mod(y, 1.0) * 2.0 - 1.0

	y2 = y[x2]

	zero_x_loc = None
	prev = -1.0
	for n, val in enumerate(y):
		if val < prev:
			zero_x_loc = n
			# TODO: value between samples
			break
		prev = val

	print("Num samples: %i" % len(y))
	print("Zero crossing loc: %s" % str(zero_x_loc))
	print("Range: %f, %f" % (np.min(y), np.max(y)))

	fft_y = np.fft.fft(y)
	theta = np.angle(fft_y)
	amp = _to_dB_norm(fft_y)

	plt.figure()

	plt.subplot(311 if plot_phase else 211)
	plt.plot(x, y, '-', x2, y2, '.')
	plt.xlim([0, fft_size-1])
	plt.ylim([-1.1, 1.1])
	plt.grid()
	plt.title('Single cycle')

	plt.subplot(312 if plot_phase else 212)
	plt.plot(x, amp, '.')
	plt.xlim([0, fft_size//2-1])
	plt.ylim([-100, 0])
	plt.xlabel('Harmonic')
	plt.ylabel('Magnitude (dB)')
	plt.grid()

	if plot_phase:
		plt.subplot(313)
		plt.plot(x, theta, '.')
		plt.xlim([0, fft_size//2-1])
		plt.ylabel('Phase')
		plt.grid()


def tri_from_saw_square_polyblep(extra_plot=False):
	fs = 44100
	f = 440
	n_samp = 1024 * 4
	oversamp = 2
	# t = np.linspace(0.0, 0.01, fs)
	t = np.arange(n_samp)
	w = f / fs

	y_saw, ph = saw_polyblep(w, n_samp, size=oversamp)
	# y_squ = square_polyblep(w, n_samp)
	y_squ = np.array([(val >= 0.5) * 2 - 1 for val in ph])  # naive square
	y = y_saw * y_squ

	naive = ph * 2.0 - 1.0

	polyblep_mag = _to_dB_norm(_fft(y))
	naive_mag = _to_dB_norm(_fft(naive))

	f = np.arange(n_samp) / n_samp

	plt.figure()

	if extra_plot:
		plt.subplot(211)
	# plt.plot(t, naive, '.-', t, y, '.-')
	# plt.legend(['Naive','PolyBLEP, oversamp=%.0f' % oversamp])
	plt.plot(t, y_squ, '.-')
	plt.plot(t, y_saw, '.-')
	plt.plot(t, y, '.-')
	plt.grid()
	plt.xlim([0, 512])
	plt.ylim([-1.1, 1.1])
	plt.title('Triangle made from saw & square')

	if extra_plot:
		plt.subplot(212)

		f_non_alias = np.arange(n_samp//(2*oversamp)) / n_samp
		f_alias = np.arange(n_samp//(2*oversamp), n_samp) / n_samp
		Y_non_alias = polyblep_mag[0:n_samp//(2*oversamp)]
		Y_alias = polyblep_mag[n_samp//(2*oversamp):n_samp]

		print(len(f_non_alias), len(Y_non_alias), len(f_alias), len(Y_alias))
		print(np.min(f_non_alias), np.max(f_non_alias), np.min(f_alias), np.max(f_alias))

		plt.plot(f, naive_mag, f_non_alias, label='Naive')
		plt.plot(Y_non_alias, label='PolyBLEP')
		plt.plot(f_alias, Y_alias, label='Aliased')
		plt.legend()
		plt.xlim([0, 0.5])
		plt.xticks([1/32.0, 1/16.0, 1/8.0, 1/4.0, 1/2.0])

		plt.grid()
		plt.ylim([-80, 0])


def plot(args):
	plot_polyblep()
	plot_full(saw=True)
	plot_full(saw=False)
	plot_cycle()
	tri_from_saw_square_polyblep()

	plt.show()

def main(args):
	plot(args)
