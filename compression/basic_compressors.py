#!/usr/bin/env python3

import numpy as np
import math

from utils import utils
from processor import ProcessorBase
from filters.peak_filters import BidirectionalOnePoleFilter


class FeedforwardCompressor(ProcessorBase):
	def __init__(self, ratio: float, attack: float, release: float, threshold_dB=0., knee_dB=0., rms=False, dB=True):
		"""
		:param ratio: Compression ratio
			Typically > 1 (e.g. for 4:1 ratio, set to 4)
			Must not be 0
			For infinity:1, set to None
			Negative ratios (overcompression) are supported
		:param attack: attack time, in samples (i.e. time in seconds * sample rate)
		:param release: release time, in samples (i.e. time in seconds * sample rate)
		:param threshold_dB: Threshold, in dBFS
		:param knee_dB: Total knee width, in dB (e.g. for knee width 6, knee is +- 3 dB)
		:param rms: if true, will use RMS detector instead of peak detector
		:param dB: if False, this will be a naive "quick and dirty" compressor that doesn't do dB conversion
		"""

		if ratio == 0:
			raise ValueError('Ratio must not be zero')

		if knee_dB and not dB:
			raise NotImplementedError('Soft knee not supported in linear mode')

		self.thresh_dB = threshold_dB
		self.thresh_lin = utils.from_dB(threshold_dB)
		self.filt = BidirectionalOnePoleFilter(attack, release)

		self.amp_calc = self._amp_calc_rms if rms else self._amp_calc_peak
		self.gain_calc = self._calc_gain_dB if dB else self._calc_gain_lin

		self.inv_ratio = 1.0 / ratio if ratio is not None else 0.
		self.one_minus_inv_ratio = 1.0 - self.inv_ratio

		self.half_knee_dB = 0.5 * knee_dB

		if knee_dB:
			# Precomputed for faster knee computation later
			self.one_minus_inv_ratio_over_2_knee = (1.0 - self.inv_ratio) / (2 * knee_dB)

	def reset(self):
		self.filt.reset()

	def _soft_knee_reduction_dB(self, amp):
		# Returns output gain

		# Full calculation:
		# (1 - 1/ratio) * np.square(amp + knee_dB/2) / (2 * knee_dB)

		# With precomputed values
		return self.one_minus_inv_ratio_over_2_knee * np.square(amp + self.half_knee_dB)

	def _amp_calc_peak(self, x):
		return self.filt.process_sample(abs(x))

	def _amp_calc_rms(self, x):
		return math.sqrt(self.filt.process_sample(x ** 2.0))

	def _calc_gain_dB(self, amp):

		if amp == 0.0:
			return 1.0

		amp = utils.to_dB(amp)
		over = amp - self.thresh_dB

		if -self.half_knee_dB < over < self.half_knee_dB:
			gain_reduction_dB = self._soft_knee_reduction_dB(amp)
			return utils.from_dB(-gain_reduction_dB)

		elif over > 0.:
			gain_reduction_dB = over * self.one_minus_inv_ratio
			return utils.from_dB(-gain_reduction_dB)

		else:
			return 1.0

	def _calc_gain_lin(self, amp):

		if amp == 0.0:
			return 1.0

		over = amp - self.thresh_lin
		if over <= 0.:
			return 1.0

		desired_amp = (over * self.inv_ratio) + self.thresh_lin

		return desired_amp / amp

	def process_sample(self, sample: float):
		amp = self.amp_calc(sample)
		return sample * self.gain_calc(amp)

	def process_sample_debug(self, x: float):
		amp = self.amp_calc(x)
		gain = self.gain_calc(amp)
		y = x * gain
		return y, dict(amp=amp, gain=gain)


class FeedbackCompressor(ProcessorBase):
	def __init__(self, ratio: float, attack: float, release: float, threshold_dB=0.0, rms=False, dB=True):
		"""
		:param ratio: Must be > 0, typically > 1; infinity not supported
		:param attack: attack time, in samples
		:param release: release time, in samples
		:param threshold_dB: Threshold, in dBFS
		:param rms: if true, will use RMS detector instead of peak detector
		:param dB: if False, this will be a naive "quick and dirty" compressor that doesn't do dB conversion
		"""

		# TODO: optional soft knee

		if ratio <= 0:
			raise ValueError('Ratio must be positive')

		if attack < 1.0 or release < 1.0:
			raise ValueError('Attack and release must be at least 1 sample')

		# Scale attack & release by ratio
		# Figured this out empirically - it seems to be the same phenomenon as the Miller effect, but haven't sat down
		# to do the math to figure out the details
		# (ratio = 1 + gain, and Miller effect scales by gain - 1)
		attack *= ratio
		release *= ratio

		self.gain = ratio - 1
		self.gain_dB = utils.from_dB(self.gain)

		self.gain_calc = self._gain_calc_dB if dB else self._gain_calc_lin
		self.amp_calc = self._amp_calc_rms if rms else self._amp_calc_peak

		self.thresh_dB = threshold_dB
		self.thresh_lin = utils.from_dB(threshold_dB)

		self.filt = BidirectionalOnePoleFilter(attack, release)
		self.gain_z1 = 0.

	def reset(self):
		self.filt.reset()
		self.gain_z1 = 0.

	def _amp_calc_peak(self, x):
		return self.filt.process_sample(abs(x))

	def _amp_calc_rms(self, x):
		return math.sqrt(self.filt.process_sample(x ** 2.0))

	def _gain_calc_dB(self, amp):

		if amp == 0.0:
			return 1.0

		over = utils.to_dB(amp) - self.thresh_dB

		over = max(over, 0.)  # clip amp to 0 (hard knee - TODO: soft knee)
		over *= -self.gain
		return utils.from_dB(over)

	def _gain_calc_lin(self, amp):

		if amp == 0.0:
			return 1.0

		amp_fb = amp - self.thresh_lin

		amp_fb = max(amp_fb, 0.)  # clip amp to 0 (hard knee - TODO: soft knee)

		amp_fb *= self.gain
		amp_fb += self.thresh_lin
		return 1.0 / amp_fb

	def process_sample(self, sample: float):

		# 1 sample delay in gain
		# TODO: experiment with optional ZDF mode
		# (self.filt would need to support state saving & loading)
		# probably not necessary except for really fast attack like 1176 (need to try it to find out!)

		y = sample * self.gain_z1
		amp = self.amp_calc(y)
		self.gain_z1 = self.gain_calc(amp)
		return y

	def process_sample_debug(self, x: float):
		y = x * self.gain_z1
		amp = self.amp_calc(y)
		debug_dict = dict(amp=amp, gain=self.gain_z1)
		self.gain_z1 = self.gain_calc(amp)

		return y, debug_dict


def test(verbose=False):
	pass


def _plot_audio(ratio, knee_dB, attack_ms=30., release_ms=100., freq=1000, sample_rate=48000, n_samp=48000):
	from matplotlib import pyplot as plt
	from generation.signal_generation import gen_sine, sample_time_index

	#attack_ms = 5.

	attack = attack_ms / 1000. * sample_rate
	release = release_ms / 1000. * sample_rate

	x = gen_sine(freq / sample_rate, n_samp)

	ampls = [0.5, 1.0, 2.0, 4.0, 0.5, 0.5, 0.5, 0.5]
	n_samp_per_section = n_samp // len(ampls)
	ampl = np.concatenate(tuple([np.ones(n_samp_per_section) * a for a in ampls]))
	x *= ampl

	y_ff = FeedforwardCompressor(ratio=ratio, attack=attack, release=release, knee_dB=knee_dB).process_vector_debug(x)
	y_fflin = FeedforwardCompressor(ratio=ratio, attack=attack, release=release, dB=False).process_vector_debug(x)
	y_fb = FeedbackCompressor(ratio=ratio, attack=attack, release=release).process_vector_debug(x)
	y_fblin = FeedbackCompressor(ratio=ratio, attack=attack, release=release, dB=False).process_vector_debug(x)

	ampl_ff = FeedforwardCompressor(ratio=ratio, attack=attack, release=release, knee_dB=knee_dB).process_vector_debug(ampl)
	ampl_fflin = FeedforwardCompressor(ratio=ratio, attack=attack, release=release, dB=False).process_vector_debug(ampl)
	ampl_fb = FeedbackCompressor(ratio=ratio, attack=attack, release=release).process_vector_debug(ampl)
	ampl_fblin = FeedbackCompressor(ratio=ratio, attack=attack, release=release, dB=False).process_vector_debug(ampl)

	t = sample_time_index(n_samp, sample_rate)

	plt.figure()

	plt.subplot(4, 1, 1)
	plt.plot(t, x, label='Input')
	plt.ylabel('Input')
	plt.grid()

	plt.subplot(4, 1, 2)
	plt.plot(t, y_ff[0], label='FF')
	plt.ylabel('FF')
	plt.grid()

	plt.subplot(4, 1, 3)
	plt.plot(t, y_fb[0], label='FB')
	plt.ylabel('FB')
	plt.grid()

	plt.subplot(4, 1, 4)
	plt.plot(t, y_fblin[0], label='FB lin')
	plt.ylabel('FB lin')
	plt.grid()

	plt.figure()

	plt.plot(t, ampl, label='Input')
	plt.plot(t, ampl_ff[0], label='FF')
	plt.plot(t, ampl_fb[0], label='FB')
	plt.plot(t, ampl_fflin[0], label='FF lin')
	plt.plot(t, ampl_fblin[0], label='FB lin')
	plt.legend()
	plt.grid()

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.plot(t, x, label='Input')
	plt.plot(t, y_ff[0], label='FB')
	plt.plot(t, y_ff[1]['amp'], label='Amp')
	plt.plot(t, y_ff[1]['gain'], label='Gain')
	plt.title('Feedforward')
	plt.legend()
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.plot(t, ampl, label='Input')
	plt.plot(t, ampl_ff[0], label='FB')
	plt.plot(t, ampl_ff[1]['amp'], label='Amp')
	plt.plot(t, ampl_ff[1]['gain'], label='Gain')
	plt.legend()
	plt.grid()

	plt.figure()

	plt.subplot(2, 1, 1)
	plt.plot(t, x, label='Input')
	plt.plot(t, y_fb[0], label='FB')
	plt.plot(t, y_fb[1]['amp'], label='Amp')
	plt.plot(t, y_fb[1]['gain'], label='Gain')
	plt.title('Feedback')
	plt.legend()
	plt.grid()

	plt.subplot(2, 1, 2)
	plt.plot(t, ampl, label='Input')
	plt.plot(t, ampl_fb[0], label='FB')
	plt.plot(t, ampl_fb[1]['amp'], label='Amp')
	plt.plot(t, ampl_fb[1]['gain'], label='Gain')
	plt.legend()
	plt.grid()



def _trace_comp_response(comp: ProcessorBase, amplitudes, min_samples=1, max_samples_per_ampl=10000, eps=0.01):

	if min_samples < 1:
		raise ValueError('min_samples must be >= 1')

	traced = np.zeros_like(amplitudes)

	for n, amp in enumerate(amplitudes):

		y_prev = 0.
		this_eps = min(eps, eps * amp)

		for _ in range(min_samples):
			y_prev = comp.process_sample(amp)

		for _ in range(max_samples_per_ampl):
			y = comp.process_sample(amp)
			if utils.approx_equal(y, y_prev, eps=this_eps):
				break
			y_prev = y

		else:
			raise Exception('Failed to converge in %u iterations! (amp %f)' % (max_samples_per_ampl, amp))

		traced[n] = y

	return traced


def _plot_trace(ratio, knee_dB):
	from matplotlib import pyplot as plt

	amp_trace = np.logspace(np.log10(0.1), np.log10(1000.), 1000)
	y_ff_amp = _trace_comp_response(FeedforwardCompressor(ratio=ratio, attack=1, release=10, knee_dB=knee_dB), amp_trace, min_samples=1)
	y_fflin_amp = _trace_comp_response(FeedforwardCompressor(ratio=ratio, attack=1, release=10, dB=False), amp_trace, min_samples=1)
	y_fb_amp = _trace_comp_response(FeedbackCompressor(ratio=ratio, attack=10, release=10), amp_trace, min_samples=100)
	y_fblin_amp = _trace_comp_response(FeedbackCompressor(ratio=ratio, attack=10, release=10, dB=False), amp_trace, min_samples=100)

	plt.figure()

	plt.subplot(2, 2, 1)
	plt.plot(amp_trace, y_ff_amp, label='FF')
	plt.plot(amp_trace, y_fb_amp, label='FB')
	plt.grid()
	plt.legend()
	plt.ylabel('linear')
	plt.title('Linear, Ratio = %s' % utils.to_pretty_str(ratio))

	plt.subplot(2, 2, 3)
	plt.plot(amp_trace, y_ff_amp, label='FF')
	plt.plot(amp_trace, y_fb_amp, label='FB')
	plt.grid()
	plt.xlabel('linear')
	plt.ylabel('linear')
	plt.xlim([0, 10])
	plt.ylim([0, 4])

	plt.subplot(2, 2, 2)
	amp_trace_dB = utils.to_dB(amp_trace)
	y_ff_amp_dB = utils.to_dB(y_ff_amp)
	y_fb_amp_dB = utils.to_dB(y_fb_amp)
	plt.plot(amp_trace_dB, y_ff_amp_dB, label='FF')
	plt.plot(amp_trace_dB, y_fb_amp_dB, label='FB')
	plt.plot()
	plt.grid()
	plt.ylabel('dB')
	plt.title('dB, Ratio = %s' % utils.to_pretty_str(ratio))

	plt.subplot(2, 2, 4)
	plt.plot(amp_trace_dB, y_ff_amp_dB, label='FF')
	plt.plot(amp_trace_dB, y_fb_amp_dB, label='FB')
	plt.plot()
	plt.grid()
	plt.xlabel('dB')
	plt.ylabel('dB')
	plt.xlim([-6, 12])
	plt.ylim([-6, 6])

	plt.figure()

	plt.subplot(2, 2, 1)
	plt.plot(amp_trace, y_fflin_amp, label='FF linear')
	plt.plot(amp_trace, y_fblin_amp, label='FB linear')
	plt.grid()
	plt.legend()
	plt.ylabel('linear')
	plt.title('Linear, Ratio = %s' % utils.to_pretty_str(ratio))

	plt.subplot(2, 2, 3)
	plt.plot(amp_trace, y_fflin_amp, label='FF linear')
	plt.plot(amp_trace, y_fblin_amp, label='FB linear')
	plt.grid()
	plt.xlabel('linear')
	plt.ylabel('linear')
	plt.xlim([0, 10])
	plt.ylim([0, 4])

	plt.subplot(2, 2, 2)
	y_fflin_amp_dB = utils.to_dB(y_fflin_amp)
	y_fblin_amp_dB = utils.to_dB(y_fblin_amp)
	plt.plot(amp_trace_dB, y_fflin_amp_dB, label='FF linear')
	plt.plot(amp_trace_dB, y_fblin_amp_dB, label='FB linear')
	plt.plot()
	plt.grid()
	plt.ylabel('dB')
	plt.title('dB, Ratio = %s' % utils.to_pretty_str(ratio))

	plt.subplot(2, 2, 4)
	plt.plot(amp_trace_dB, y_fflin_amp_dB, label='FF linear')
	plt.plot(amp_trace_dB, y_fblin_amp_dB, label='FB linear')
	plt.plot()
	plt.grid()
	plt.xlabel('dB')
	plt.ylabel('dB')
	plt.xlim([-6, 12])
	plt.ylim([-6, 6])


def plot(args):
	from matplotlib import pyplot as plt

	ratio = 4.0
	knee_dB = 6.0

	_plot_audio(ratio, knee_dB)

	_plot_trace(ratio, knee_dB)

	plt.show()


def main(args):
	plot(args)
