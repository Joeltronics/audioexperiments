#!/usr/bin/env python3

from utils import plot_utils
from utils import utils
from generation.signal_generation import sample_time_index
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal
from typing import Optional


def _sec_to_str(val_sec):
	if val_sec < 1:
		return '%.1f ms' % (val_sec * 1000.)
	else:
		return '%.3f sec' % val_sec


def _average(vals: np.ndarray, wc) -> np.ndarray:
	b, a = scipy.signal.butter(1, wc * 2)
	y = scipy.signal.filtfilt(b, a, vals)
	return y


def echo_density(ir: np.ndarray, sample_rate=48000) -> np.ndarray:
	# Based on Abel & Huang's "A Simple, Robust Measure of Reverberation Echo Density", AES 2006
	# https://ccrma.stanford.edu/courses/318/mini-courses/rooms/mus318_Abel_Lecture/echo%20density.pdf
	# However, this implementation uses a one-pole IIR filter for averaging instead of a sliding window

	# density = pct of samples with abs value > standard deviation
	# (stdev = RMS when mean = 0)
	rms = np.sqrt(_average(np.square(ir), wc=20. / sample_rate))
	return _average(np.abs(ir) > rms, wc=20./sample_rate)


def analyze_reverb_ir(ir: np.ndarray, sample_rate=48000, title: Optional[str]=None, normalize=True) -> None:

	def avg(vals):
		return _average(vals, wc=20./sample_rate)

	if normalize:
		ir = utils.normalize(ir)

	n_samp = len(ir)
	time_s = n_samp / sample_rate
	t = sample_time_index(n_samp, sample_rate)

	abs_ir = np.abs(ir)

	dc = avg(ir)
	peak = avg(abs_ir)
	rms = np.sqrt(avg(np.square(ir)))

	# Density

	density = echo_density(ir, sample_rate)
	crest_factor = peak / rms

	# Predelay: find first peak of |ir|

	deriv_abs_ir = np.gradient(abs_ir, t)
	predelay_samples = np.argmax(deriv_abs_ir < 0) - 1
	predelay_sec = predelay_samples / sample_rate

	# Highest peak

	abs_peak = np.amax(abs_ir)
	peak_samples = np.argmax(abs_ir)
	peak_sec = peak_samples / sample_rate

	# dB amplitude

	rms_dB = utils.to_dB(rms, min_dB=-200.)
	peak_dB = utils.to_dB(peak, min_dB=-200.)
	max_dB = np.nanmax(peak_dB)

	# RT60

	rt60_dB = max_dB - 60
	rt60_samples = np.argmax(peak_dB[peak_samples:] < rt60_dB) + peak_samples
	rt60_sec = rt60_samples / sample_rate

	# Plots

	fig = plt.figure()

	plt.subplot(4, 1, 1)

	plt.axvline(predelay_sec, color='r')
	plt.axvline(peak_sec, color='r')

	plt.plot(t, ir, label='Impulse Response')
	plt.plot(t, peak, label='Peak Magnitude')
	plt.plot(t, rms, label='RMS Magnitude (i.e. std. dev.)')
	plt.plot(t, dc, label='Mean (i.e. DC offset)')

	plt.text(predelay_sec, abs_peak, 'Predelay: %s' % _sec_to_str(predelay_sec))
	plt.text(peak_sec, abs_peak*1.1, 'Peak: %s' % _sec_to_str(peak_sec))

	plt.legend()
	plt.grid()
	plt.xlim([0., time_s])
	plt.ylim([-1.1 * abs_peak, 1.2 * abs_peak])

	ax = plt.subplot(4, 1, 2)

	plt.plot(t, density, label='Density')
	plt.plot(t, crest_factor, label='Crest factor')

	plt.legend()
	plt.grid()
	plt.xlim([0., time_s])
	ymin, ymax = ax.get_ylim()
	plt.ylim([min(ymin, 0.), max(ymax, 1.)])

	plt.subplot(4, 1, 3)

	plt.axhline(max_dB, color='r')
	plt.axhline(rt60_dB, color='r')
	plt.axvline(rt60_sec, color='r')

	plt.plot(t, peak_dB, label='Peak Magnitude')

	plt.text(0, max_dB, 'Peak: %.1f dB' % max_dB)
	plt.text(rt60_sec, rt60_dB, 'RT60: %s' % _sec_to_str(rt60_sec))

	plt.legend()
	plt.grid()
	plt.xlim([0., time_s])
	plt.ylim([max_dB - 120, max_dB + 6])
	plt.ylabel('dB')

	plt.subplot(4, 1, 4)
	plot_utils.plot_spectrogram(ir, sample_rate, log=True)
	plt.xlim([0., time_s])
	plt.xlabel('Time (s)')
	plt.ylabel('Freq (Hz)')
	plt.title('Spectrogram')

	if title:
		fig.suptitle(title)

	plt.show()
