#!/usr/bin/env python3

"""
Delay with variable rate (e.g. clock rate or tape speed) but fixed length (e.g. stages/samples/distance)
Equivalent to behavior of BBD, PT2399, or most tape delays
"""

import argparse
from math import floor, ceil
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal
from typing import Optional, Tuple, List

from delay_reverb.delay_line import DelayLine
from generation.signal_generation import gen_sine, gen_saw
from generation.polyblep import polyblep
from processor import ProcessorBase
from utils.utils import lerp, reverse_lerp, scale, to_dB, index_of, parabolic_interp_find_peak


def do_fft(x, n_fft: Optional[int]=None, sample_rate=1.0, window=False):
	
	if n_fft is None:
		n_fft = len(x)

	if window:
		y = np.fft.fft(x * np.hamming(len(x)), n=n_fft)
	else:
		y = np.fft.fft(x, n=n_fft)
	
	f = np.fft.fftfreq(n_fft, sample_rate)
	
	y = y[0:len(y)//2]
	f = f[0:len(f)//2]
	
	y = to_dB(np.abs(y))

	return y, f


class VariableRateDelayLine(ProcessorBase):
	"""
	Fixed length but variable rate delay line

	No built-in anti-aliasing/anti-imaging filters - would typically expect to use external filters, possibly also
	with oversampling

	Can also set 0 stages to use as a lo-fi "bitcrusher" style resampler (i.e. zero-order hold)
	"""
	def __init__(
			self,
			num_stages: int,
			clock_freq: float,
			ramp_output=True,
			polyblep_size: Optional[float]=None,
			linblep=False,
			):
		"""
		:param num_stages: number of delay line stages, or 0 for zero-order-hold bitcurhser
		:param clock_freq: clock frequency, <= 1
		:param ramp_output:
			If False, will use zero-order hold, for behavior akin to BBD without reconstruction filter. Will alias
				badly - not just around the clock rate (which may actually be desirable for analog emulation), but also
				around the operating sample rate (which is undesired).
			If True, will linearly interpolate output samples, for more tape-like behavior. Also helps limit unwanted
				aliasing, so also recommended for BBD emulation when using a good external reconstruction filter.
				Increases delay by 1/2 clock relative to step case non-ramp case.
		:param polyblep_size: Size of polyblep step to use (ignored if ramp_output).
			If num_stages = 0, adds delay of (polyblep_size/2) samples.
		:param linblep: If True, will use linear bandlimited step (ignored if ramp_output or polyblep_size)

		:note:
			Average delay without ramp is ((num_stages + 0.5) / clock_freq) samples
			Average delay with ramp is ((num_stages + 1) / clock_freq) samples
		"""

		self.clock_freq = None

		self.num_stages = num_stages
		self.ramp_output = ramp_output
		self.half_polyblep_size = 0.5 * polyblep_size if polyblep_size else None
		self.linblep = linblep

		self.clock_phase = None
		self.x_prev = None
		self.y_prev = None
		self.y_curr = None
		self.delay_line = DelayLine(num_stages) if self.num_stages > 0 else None

		self.set_clock_freq(clock_freq)
		self.reset()

	def set_clock_freq(self, clock_freq: float):
		# TODO: handle this case better (it will alias, but could still at least attempt to handle it)
		if clock_freq > 1.0:
			raise ValueError('Clock rate too high!')
		self.clock_freq = clock_freq

	def reset(self) -> None:
		self.clock_phase = 0.0
		self.x_prev = 0.0
		self.y_prev = 0.0
		self.y_curr = 0.0

		if self.delay_line is not None:
			self.delay_line.reset()

	def get_state(self):
		return (
			self.clock_phase,
			self.x_prev,
			self.y_prev,
			self.y_curr,
			(self.delay_line.get_state() if self.delay_line is not None else None),
		)

	def set_state(self, state):
		self.clock_phase, self.x_prev, self.y_prev, self.y_curr, delay_line_state = state
		if self.delay_line is not None:
			self.delay_line.set_state(delay_line_state)

	def _process_sample(self, x: float, debug: bool) -> Tuple[float, Optional[dict]]:

		clock_phase_0 = self.clock_phase
		clock_phase_1 = clock_phase_0 + self.clock_freq

		debug_info = None
		if debug:
			debug_info = dict(
				clock_phase=None,
				x_interp_t=None,
				x_interp_val=None,
			)

		x_interp_t = None

		if clock_phase_1 >= 2.0:
			# Shouldn't be possible due to clock_freq < 1 check above (TODO: handle this case)
			raise AssertionError

		elif clock_phase_1 >= 1.0:

			x_interp_t = reverse_lerp((clock_phase_0, clock_phase_1), 1.0)

			# TODO: optional higher-order interpolation
			x_interp_val = lerp((self.x_prev, x), x_interp_t)

			self.y_prev = self.y_curr
			if self.delay_line is None:
				self.y_curr = x_interp_val
			else:
				self.y_curr = self.delay_line.peek_front()
				self.delay_line.push_back(x_interp_val)

			if debug_info is not None:
				debug_info['x_interp_t'] = x_interp_t
				debug_info['x_interp_val'] = x_interp_val

			clock_phase_1 = clock_phase_1 % 1.0

		if self.ramp_output:
			y = lerp((self.y_prev, self.y_curr), clock_phase_1)

		elif self.half_polyblep_size is not None:
			p_freq = self.clock_freq * self.half_polyblep_size
			if self.delay_line is not None:
				if clock_phase_1 > 1.0 - p_freq:
					# Right before edge
					p_idx = scale(clock_phase_1, (1.0 - p_freq, 1.0), (-1.0, 0.0))
					p = polyblep(p_idx)
					y = lerp((self.y_curr, self.delay_line.peek_front()), p)
				elif clock_phase_1 < p_freq:
					# Right after edge
					p_idx = scale(clock_phase_1, (0.0, p_freq), (0.0, 1.0))
					p = polyblep(p_idx)
					y = lerp((self.y_prev, self.y_curr), p)
				else:
					y = self.y_curr
			else:
				if clock_phase_1 < 2.0 * p_freq:
					# Transition
					p_idx = scale(clock_phase_1, (0.0, 2.0*p_freq), (-1.0, 1.0))
					p = polyblep(p_idx)
					y = lerp((self.y_prev, self.y_curr), p)
				else:
					# Not a transition
					y = self.y_curr

		elif self.linblep and x_interp_t is not None:
			"""
			Interpolation is backwards from what you might expect:
			* If x_interp_t near 0, transition was near last sample, so we want to be mostly the new sample
			* If x_interp_t near 1, transition was near current sample, so we want to be mostly the previous
			"""
			y = lerp((self.y_curr, self.y_prev), x_interp_t)

		else:
			y = self.y_curr

		self.clock_phase = clock_phase_1
		self.x_prev = x

		if debug_info is not None:
			debug_info['clock_phase'] = clock_phase_1

		return y, debug_info

	def process_sample(self, sample: float) -> float:
		y, _ = self._process_sample(sample, debug=False)
		return y

	def process_sample_debug(self, sample: float) -> Tuple[float, dict]:
		return self._process_sample(sample, debug=True)


def _interp_clock_phase(clock_phase: np.ndarray) -> Tuple[List[float], List[float]]:
	"""
	Interpolate clock phase to get precise zero crossing locations for plotting
	"""

	clock_phase_t = []
	clock_phase_y = []
	prev_phase = 0.0

	for idx, phase in enumerate(clock_phase):

		if idx == 0:
			clock_phase_t.append(idx)
			clock_phase_y.append(phase)

		if phase < prev_phase:
			phase_pre_wrap = phase + 1.0
			t_zero_x = reverse_lerp((prev_phase, phase_pre_wrap), 1.0)
			clock_phase_t.append(idx - 1 + t_zero_x)
			clock_phase_t.append(idx - 1 + t_zero_x)
			clock_phase_y.append(1.0)
			clock_phase_y.append(0.0)

		if idx == (len(clock_phase) - 1):
			clock_phase_t.append(idx)
			clock_phase_y.append(phase)

		prev_phase = phase
	
	return clock_phase_t, clock_phase_y


def _do_plot(
		num_stages: int,
		freq=0.0126,
		clock_freq=0.11,
		num_samples_plot=128,
		num_samples=4096,
		plot_clock_phase=False,
		verbose=False,
		**kwargs
		):

	t_offset = int(ceil(num_stages / clock_freq))
	plot_t_range = [t_offset, t_offset + num_samples_plot]
	if plot_t_range[1] > num_samples:
		raise ValueError('Must set num_samples higher at this combination of num_stages & clock_freq')

	x = gen_sine(freq, n_samp=num_samples)
	t = np.arange(num_samples)

	delay = VariableRateDelayLine(num_stages=num_stages, clock_freq=clock_freq, **kwargs)
	y, debug_info = delay.process_vector_debug(x)

	b, a = scipy.signal.butter(N=8, Wn=0.5*clock_freq)
	y_reconstructed = scipy.signal.filtfilt(b, a, y)

	clock_phase = debug_info['clock_phase']
	x_interp_t = debug_info['x_interp_t']
	x_interp_val = debug_info['x_interp_val']

	ramp = delay.ramp_output
	polyblep_size = kwargs['polyblep_size'] if 'polyblep_size' in kwargs else None
	linblep = kwargs['linblep'] if 'linblep' in kwargs else None

	ax_c = None
	if verbose:
		fig, (ax_t, ax_f, ax_c) = plt.subplots(3, 1)
	else:
		fig, (ax_t, ax_f) = plt.subplots(2, 1)

	if num_stages and ramp:
		title = f'Tape-ish delay ({num_stages} samples)'
	elif num_stages:
		title = f'BBD style delay ({num_stages} stages)'
	elif ramp:
		title = 'Ramp'
	else:
		title = 'Zero-order hold'

	title_short = title

	title += f', clock rate {clock_freq:g}'

	title_extra = ''

	if not ramp:
		if polyblep_size:
			title_extra += f', polyblep {polyblep_size}'
		elif linblep:
			title_extra += ', linblep'

	title += title_extra
	title_short += title_extra

	fig.suptitle(title)

	if verbose:
		if ramp:
			expected_delay_samples = (num_stages + 1) / clock_freq
		else:
			expected_delay_samples = (num_stages + 0.5) / clock_freq

		xa = x[t_offset:]
		ya = y_reconstructed[t_offset:]
		xcorr = scipy.signal.correlate(ya, xa)
		lags = scipy.signal.correlation_lags(len(ya), len(xa))

		find_peak_samples_range = (
			int(floor(num_stages / clock_freq)),
			int(floor((num_stages + 1.5) / clock_freq))
		)
		
		find_peak_lag_indices = tuple(index_of(val, lags) for val in find_peak_samples_range)

		peak_idxs, _ = scipy.signal.find_peaks(xcorr[find_peak_lag_indices[0]:find_peak_lag_indices[1]])

		if not peak_idxs:
			raise AssertionError('Failed to find peaks')

		if len(peak_idxs) > 1:
			print(len(peak_idxs))
			print(peak_idxs)
			# TODO: better handling of multiple peaks case
			raise AssertionError('Found multiple peaks')

		peak_xcorr_idx = peak_idxs[0] + find_peak_lag_indices[0]

		# Interpolate to find peak in between samples (parabolic, or sine fit)
		px, peak_xcorr = parabolic_interp_find_peak((xcorr[peak_xcorr_idx - 1], xcorr[peak_xcorr_idx], xcorr[peak_xcorr_idx + 1]))

		actual_delay_samples = scale(px, (-1.0, 1.0), (lags[peak_xcorr_idx - 1], lags[peak_xcorr_idx + 1]))

		expected_delay_clocks = expected_delay_samples * clock_freq
		actual_delay_clocks = actual_delay_samples * clock_freq

		extra_delay_samples = actual_delay_samples - expected_delay_samples

		print('%-40s %9.2f %9.2f %9.1f %9.1f %9.2f %9.2f %9.2f' % (
			title_short,
			clock_freq,
			1.0 / clock_freq,
			expected_delay_clocks,
			expected_delay_samples,
			actual_delay_clocks,
			actual_delay_samples,
			extra_delay_samples
		))

		assert ax_c is not None

		line, = ax_c.plot(lags, xcorr, label='XCorr(yr, x)')
		ax_c.plot(actual_delay_samples, peak_xcorr, '.', color=line.get_color(), label='XCorr peak')
		ax_c.axvline(expected_delay_samples, color='darkgreen', label='Expected delay')
		ax_c.axvline(1/clock_freq, color='orange', label='1 clock period')

		if expected_delay_samples:
			ax_c.set_xlim([0, max(2*expected_delay_samples, 2/clock_freq)])
		else:
			ax_c.set_xlim([-2/clock_freq, 2/clock_freq])

		ax_c.set_xlabel(r'$\Delta$t (samples)')
		ax_c.set_ylabel('Cross-Correlation')
		ax_c.grid()
		ax_c.legend(loc='lower right')

	ax_f.axvline(freq, label='X frequency')
	ax_f.axvline(clock_freq, label='Clock', color='orange')

	ax_t.plot(t, x, '.-', label='X')
	
	fft_x, f = do_fft(x, window=True)
	ax_f.plot(f, fft_x, label='X')

	mask = np.isfinite(x_interp_t)
	ax_t.plot(t[mask] + x_interp_t[mask] - 1, x_interp_val[mask], '.', label='X samples')

	line, = ax_t.plot(t, y, '.-', label='Y')

	fft_y, f = do_fft(y, window=True)
	ax_f.plot(f, fft_y, label='Y', color=line.get_color(), zorder=-2)

	line, = ax_t.plot(t, y_reconstructed, label='Y filtered')
	fft_y_r, f = do_fft(y_reconstructed, window=True)
	ax_f.plot(f, fft_y_r, label='Y filtered', color=line.get_color(), zorder=-1)

	if plot_clock_phase:
		clock_phase_t, clock_phase_y = _interp_clock_phase(clock_phase)
		ax_t.plot(clock_phase_t, clock_phase_y, label='Clock phase', zorder=-1)

	ax_t.set_xlabel('t (samples)')
	ax_t.grid()
	ax_t.legend()
	ax_t.set_xlim(plot_t_range)

	ax_f.set_xlabel('f')
	ax_f.set_ylabel('FFT (dB)')
	ax_f.grid()
	ax_f.legend()


def _plot_clock_rate_step(num_stages=32, freq=0.0126, clock_freq=(0.13, 0.21), num_samples=2048, **kwargs):

	x = gen_sine(freq, n_samp=num_samples)
	t = np.arange(num_samples)

	delay = VariableRateDelayLine(num_stages=num_stages, clock_freq=clock_freq[0], **kwargs)
	y1, debug_info_1 = delay.process_vector_debug(x[:num_samples//2])
	delay.set_clock_freq(clock_freq[1])
	y2, debug_info_2 = delay.process_vector_debug(x[num_samples//2:])

	y = np.concatenate((y1, y2))
	clock_phase = np.concatenate((debug_info_1['clock_phase'], debug_info_2['clock_phase']))
	x_interp_t = np.concatenate((debug_info_1['x_interp_t'], debug_info_2['x_interp_t']))
	x_interp_val = np.concatenate((debug_info_1['x_interp_val'], debug_info_2['x_interp_val']))

	fig, (ax_t, ax_f) = plt.subplots(2, 1)
	fig.suptitle(f'BBD style delay ({num_stages} stages), clock rate change {clock_freq[0]:g} -> {clock_freq[1]:g}')

	ax_f.axvline(freq, label='X frequency')
	ax_f.axvline(clock_freq[0], label='Clock', color='orange')
	ax_f.axvline(clock_freq[1], color='orange')

	ax_t.plot(t, x, '-', label='X')
	
	fft_x, f = do_fft(x, window=True)
	ax_f.plot(f, fft_x, label='X')

	mask = np.isfinite(x_interp_t)
	ax_t.plot(t[mask] + x_interp_t[mask] - 1, x_interp_val[mask], '.', label='X samples')

	line, = ax_t.plot(t, y, '-', label='Y')

	fft_y, f = do_fft(y, window=True)
	ax_f.plot(f, fft_y, label='Y', color=line.get_color(), zorder=-1)

	clock_phase_t, clock_phase_y = _interp_clock_phase(clock_phase)
	ax_t.plot(clock_phase_t, clock_phase_y, label='Clock phase', zorder=-1)

	ax_t.set_xlabel('t (samples)')
	ax_t.grid()
	ax_t.legend()

	ax_f.set_xlabel('f')
	ax_f.set_ylabel('FFT (dB)')
	ax_f.grid()
	ax_f.legend()


def _plot_chorus(num_stages=256, freq=0.0126, clock_freq=(0.05, 0.15), chorus_freq=0.01, num_samples=8192, saw=False):

	x = gen_saw(freq, n_samp=num_samples) if saw else gen_sine(freq, n_samp=num_samples)
	t = np.arange(num_samples)

	clock_avg = 0.5 * (clock_freq[1] + clock_freq[0])
	clock_mod = 0.5 * (clock_freq[1] - clock_freq[0])

	clock_freq = clock_mod * gen_sine(chorus_freq, n_samp=num_samples) + clock_avg

	y = np.zeros(num_samples)
	clock_phase = np.zeros(num_samples)
	x_interp_t = np.zeros(num_samples)
	x_interp_val = np.zeros(num_samples)

	delay = VariableRateDelayLine(num_stages=num_stages, clock_freq=clock_freq[0], ramp_output=True)

	for n in range(num_samples):
		xn = x[n]
		yn, debug_info = delay.process_sample_debug(xn)
		y[n] = yn
		clock_phase[n] = debug_info['clock_phase']
		x_interp_t[n] = debug_info['x_interp_t']
		x_interp_val[n] = debug_info['x_interp_val']

	fig, (ax_t, ax_f) = plt.subplots(2, 1)
	fig.suptitle(f'BBD style chorus ({num_stages} stages), average clock {clock_avg:g}')

	ax_f.axvline(freq, label='X frequency')
	ax_f.axvline(clock_freq[0], label='Clock', color='orange')

	ax_t.plot(t, x, '-', label='X')
	
	fft_x, f = do_fft(x, window=True)
	ax_f.plot(f, fft_x, label='X')

	mask = np.isfinite(x_interp_t)
	ax_t.plot(t[mask] + x_interp_t[mask] - 1, x_interp_val[mask], '.', label='X samples')

	line, = ax_t.plot(t, y, '-', label='Y')

	fft_y, f = do_fft(y, window=True)
	ax_f.plot(f, fft_y, label='Y', color=line.get_color(), zorder=-1)

	clock_phase_t, clock_phase_y = _interp_clock_phase(clock_phase)
	ax_t.plot(clock_phase_t, clock_phase_y, label='Clock phase', zorder=-1)

	ax_t.set_xlabel('t (samples)')
	ax_t.grid()
	ax_t.legend()

	ax_f.set_xlabel('f')
	ax_f.set_ylabel('FFT (dB)')
	ax_f.grid()
	ax_f.legend()


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--clock', action='store_true', dest='plot_clock_phase', help='Plot clock phase')
	return parser


def plot(args, verbose=False):

	if verbose:
		print()
		print('%-40s  %18s  %18s  %18s  %8s' % (
			'', 'Clock'.center(18), 'Expected Delay'.center(18), 'Actual Delay'.center(18), 'Diff'.center(8)
		))
		print('%40s %9s %9s %9s %9s %9s %9s %9s' % (
			'', 'Freq', 'Period', 'Clocks', 'Samples', 'Clocks', 'Samples', 'Samples'
		))
		print()

	_do_plot(num_stages=0, plot_clock_phase=args.plot_clock_phase, ramp_output=False, verbose=verbose)
	_do_plot(num_stages=0, plot_clock_phase=args.plot_clock_phase, ramp_output=False, linblep=True, verbose=verbose)
	_do_plot(num_stages=1, plot_clock_phase=args.plot_clock_phase, polyblep_size=1, ramp_output=False, verbose=verbose)
	_do_plot(num_stages=1, plot_clock_phase=args.plot_clock_phase, polyblep_size=2, ramp_output=False, verbose=verbose)
	_do_plot(num_stages=1, plot_clock_phase=args.plot_clock_phase, polyblep_size=4, ramp_output=False, verbose=verbose)
	_do_plot(num_stages=0, plot_clock_phase=args.plot_clock_phase, polyblep_size=1, ramp_output=False, verbose=verbose)
	_do_plot(num_stages=0, plot_clock_phase=args.plot_clock_phase, polyblep_size=2, ramp_output=False, verbose=verbose)
	_do_plot(num_stages=0, plot_clock_phase=args.plot_clock_phase, polyblep_size=4, ramp_output=False, verbose=verbose)
	_do_plot(num_stages=16, plot_clock_phase=args.plot_clock_phase, ramp_output=False, verbose=verbose)
	_do_plot(num_stages=0, plot_clock_phase=args.plot_clock_phase, ramp_output=True, verbose=verbose)
	_do_plot(num_stages=16, plot_clock_phase=args.plot_clock_phase, ramp_output=True, verbose=verbose)
	_plot_clock_rate_step(ramp_output=False)
	_plot_clock_rate_step(ramp_output=True)
	_plot_chorus(saw=False)
	_plot_chorus(saw=True)
	plt.show()


def main(args):
	plot(args, verbose=True)
