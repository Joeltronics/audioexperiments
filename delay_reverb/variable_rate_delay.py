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
from utils.utils import lerp, reverse_lerp, scale, to_dB, index_of, parabolic_interp_find_peak, quadratic_coeffs


def do_fft(x, n_fft: Optional[int]=None, sample_rate=1.0, window=False):
	
	if n_fft is None:
		n_fft = len(x)

	if window:
		y = np.fft.fft(x * np.hamming(len(x)), n=n_fft, norm='forward')
	else:
		y = np.fft.fft(x, n=n_fft, norm='forward')

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
			ramp_output: Optional[bool] = None,
			quadratic_ramp = False,
			polyblep_size: Optional[float] = None,
			linblep = False,
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
			Defaults to False if num_stages == 0, True otherwise
		:param quadratic_ramp: If using ramp_output, ramp will be quadratic instead of linear if this is set
		:param polyblep_size: Size of polyblep step to use (ignored if ramp_output); will be clipped to <= 1/clock_freq.
			If num_stages = 0, adds delay of min(polyblep_size/2, 0.5/clock_freq) samples.
		:param linblep: If True, will use linear bandlimited step (ignored if ramp_output or polyblep_size)
			Adds 0.5 samples extra delay

		:note:
			Average delay without ramp is ((max(num_stages, 1) - 0.5) / clock_freq) samples
			Average delay with ramp is (num_stages / clock_freq) samples
		"""

		if ramp_output and (num_stages == 0):
			raise ValueError('num_stages is minimum 1 with ramp_output')
		if ramp_output is None:
			ramp_output = num_stages > 0

		self.num_stages = num_stages
		self.ramp_output = ramp_output
		self.quadratic_ramp = quadratic_ramp and ramp_output
		self.half_polyblep_size = 0.5 * polyblep_size if (polyblep_size and not ramp_output) else None
		self.linblep = linblep and not (polyblep_size or ramp_output)
		self.naive = not (ramp_output or polyblep_size or linblep)

		assert(1 == sum([bool(val) for val in [
			self.naive, self.ramp_output, self.half_polyblep_size, self.linblep
		]]))

		self.zero_delay_polyblep = self.half_polyblep_size and self.num_stages > 1

		self.delay_line = DelayLine(num_stages - 1) if self.num_stages > 1 else None

		self.clock_freq = None
		self.quadratic_coeffs = None
		self.clock_phase = None
		self.x1 = None
		self.y1 = None
		self.y0 = None

		self.set_clock_freq(clock_freq)
		self.reset()

	def get_name(self) -> str:

		plural = "s" if self.num_stages > 1 else ""

		if self.num_stages == 0:
			title = 'Zero-order hold downsampler'
		elif self.ramp_output:
			title = f'Tape-ish delay ({self.num_stages} stage{plural})'
		else:
			title = f'BBD style delay ({self.num_stages} stage{plural})'

		if not self.ramp_output:
			if self.half_polyblep_size:
				title += f', polyblep size {2*self.half_polyblep_size:g}'
			elif self.linblep:
				title += ', linblep'
			else:
				title += ', naive step'
		elif self.quadratic_ramp:
			title += ', quadratic'

		return title

	def set_clock_freq(self, clock_freq: float):
		self.clock_freq = clock_freq

	def reset(self) -> None:
		self.clock_phase = 0.0
		self.x1 = 0.0
		self.y1 = 0.0
		self.y0 = 0.0

		if self.quadratic_ramp:
			self.quadratic_coeffs = (0.0, 0.0, 0.0)

		if self.delay_line is not None:
			self.delay_line.reset()

	def get_state(self):
		return (
			self.clock_phase,
			self.x1,
			self.y1,
			self.y0,
			self.quadratic_coeffs,
			(self.delay_line.get_state() if self.delay_line is not None else None),
		)

	def set_state(self, state):
		self.clock_phase, self.x1, self.y1, self.y0, self.quadratic_coeffs, delay_line_state = state
		if self.delay_line is not None:
			self.delay_line.set_state(delay_line_state)

	def _process_sample(self, x: float, debug: bool) -> Tuple[float, Optional[dict]]:

		clock_phase_0 = self.clock_phase
		clock_phase_1_unwrapped = clock_phase_0 + self.clock_freq
		clock_phase_1_wrapped = clock_phase_1_unwrapped % 1.0

		debug_info = None
		if debug:
			debug_info = dict(
				clock_phase=None,
				x_interp_t=None,
				x_interp_val=None,
				num_zero_crossings=0,
			)

		y = None

		if clock_phase_1_unwrapped >= 2.0:
			# Multiple clock zero crossings in this sample

			# If not naive, then y = weighted average of all samples seen
			# (equivalent to linblep, but for more than 2 samples)
			# If self.naive, this will be ignored later
			y = 0.0

			x_interp_t_prev = 0.0
			zero_crossing = 1
			while zero_crossing < clock_phase_1_unwrapped:
				
				x_interp_t = reverse_lerp((clock_phase_0, clock_phase_1_unwrapped), float(zero_crossing))
				# TODO: optional higher-order interpolation
				x_interp_val = lerp((self.x1, x), x_interp_t)

				y += self.y1 * (x_interp_t - x_interp_t_prev)

				y2 = self.y1
				self.y1 = self.y0
				if self.delay_line is None:
					self.y0 = x_interp_val
				else:
					self.y0 = self.delay_line.peek_front()
					self.delay_line.push_back(x_interp_val)

				if self.quadratic_ramp:
					self.quadratic_coeffs = quadratic_coeffs(y2, self.y1, self.y0)

				if zero_crossing == 1 and debug_info is not None:
					# Give debug info for 1st zero crossing only
					debug_info['x_interp_t'] = x_interp_t
					debug_info['x_interp_val'] = x_interp_val

				x_interp_t_prev = x_interp_t
				zero_crossing += 1

			y += self.y0 * (1.0 - x_interp_t_prev)

			if debug_info is not None:
				debug_info['num_zero_crossings'] = (zero_crossing - 1)

		elif clock_phase_1_unwrapped >= 1.0:

			x_interp_t = reverse_lerp((clock_phase_0, clock_phase_1_unwrapped), 1.0)

			# TODO: optional higher-order interpolation
			x_interp_val = lerp((self.x1, x), x_interp_t)

			y2 = self.y1
			self.y1 = self.y0
			if self.delay_line is None:
				self.y0 = x_interp_val
			else:
				self.y0 = self.delay_line.peek_front()
				self.delay_line.push_back(x_interp_val)

			if self.quadratic_ramp:
				self.quadratic_coeffs = quadratic_coeffs(y2, self.y1, self.y0)

			if self.linblep:
				"""
				Interpolation is backwards from what you might expect:
				* If x_interp_t near 0, transition was near last sample, so we want to be mostly the new sample
				* If x_interp_t near 1, transition was near current sample, so we want to be mostly the previous
				"""
				y = lerp((self.y0, self.y1), x_interp_t)

			if debug_info is not None:
				debug_info['x_interp_t'] = x_interp_t
				debug_info['x_interp_val'] = x_interp_val
				debug_info['num_zero_crossings'] = 1

		if self.naive:
			y = self.y0

		elif y is not None:
			pass

		elif self.quadratic_ramp:
			assert self.quadratic_coeffs is not None
			a, b, c = self.quadratic_coeffs
			y = a * (clock_phase_1_wrapped ** 2) + b * clock_phase_1_wrapped + c

		elif self.ramp_output:
			y = lerp((self.y1, self.y0), clock_phase_1_wrapped)

		elif self.half_polyblep_size:
			p_freq = self.clock_freq * self.half_polyblep_size
			# There are problems if polyblep size is too large for clock freq, so limit max p_freq to 0.5
			p_freq = min(p_freq, 0.5)

			if self.zero_delay_polyblep:
				assert self.delay_line is not None
				if clock_phase_1_wrapped > 1.0 - p_freq:
					# Right before edge
					p_idx = scale(clock_phase_1_wrapped, (1.0 - p_freq, 1.0), (-1.0, 0.0))
					p = polyblep(p_idx)
					y = lerp((self.y0, self.delay_line.peek_front()), p)
				elif clock_phase_1_wrapped < p_freq:
					# Right after edge
					p_idx = scale(clock_phase_1_wrapped, (0.0, p_freq), (0.0, 1.0))
					p = polyblep(p_idx)
					y = lerp((self.y1, self.y0), p)
				else:
					y = self.y0
			else:
				if clock_phase_1_wrapped < 2.0 * p_freq:
					# Transition
					p_idx = scale(clock_phase_1_wrapped, (0.0, 2.0*p_freq), (-1.0, 1.0))
					p = polyblep(p_idx)
					y = lerp((self.y1, self.y0), p)
				else:
					# Not a transition
					y = self.y0

		else:
			y = self.y0

		self.clock_phase = clock_phase_1_wrapped
		self.x1 = x

		if debug_info is not None:
			# TODO: make this unwrapped (will cause problems in some debug graphs though; make them handle this)
			debug_info['clock_phase'] = clock_phase_1_wrapped

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
		plot_xcorr=False,
		**kwargs
		):

	t_offset = int(ceil((num_stages + 1) / clock_freq)) if num_stages > 0 else 0
	plot_t_range = [t_offset, t_offset + num_samples_plot]
	if plot_t_range[1] > num_samples:
		raise ValueError('Must set num_samples higher at this combination of num_stages & clock_freq')

	x = gen_sine(freq, n_samp=num_samples)
	t = np.arange(num_samples)

	delay = VariableRateDelayLine(num_stages=num_stages, clock_freq=clock_freq, **kwargs)
	y, debug_info = delay.process_vector_debug(x)

	b, a = scipy.signal.butter(N=8, Wn=4.0 * freq)
	y_reconstructed = scipy.signal.filtfilt(b, a, y)

	clock_phase = debug_info['clock_phase']
	x_interp_t = debug_info['x_interp_t']
	x_interp_val = debug_info['x_interp_val']

	ramp = delay.ramp_output

	ax_c = None
	if verbose and plot_xcorr:
		fig, (ax_t, ax_f, ax_c) = plt.subplots(3, 1)
	else:
		fig, (ax_t, ax_f) = plt.subplots(2, 1)

	title_short = delay.get_name()
	title = f'{title_short}, clock rate {clock_freq:g}'

	fig.suptitle(title)

	ax_f.axvline(freq, label='X frequency', color='darkgreen')
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
	ax_f.set_ylim([-120, 0])
	ax_f.grid()
	ax_f.legend()
	
	if not verbose:
		return

	if num_stages == 0:
		expected_extra_sample_delay = 0.0
		if delay.linblep:
			expected_extra_sample_delay = 0.5
		elif delay.half_polyblep_size:
			expected_extra_sample_delay = min(delay.half_polyblep_size, 0.5/clock_freq)
		expected_delay = (0.5, expected_extra_sample_delay)
	elif ramp:
		#expected_delay = (num_stages, 0.5 if linblep else 0)
		expected_delay = (num_stages, 0)
		#expected_delay_total_samples = num_stages / clock_freq
	else:
		expected_extra_sample_delay = 0.0
		if delay.linblep:
			expected_extra_sample_delay = 0.5
		elif delay.half_polyblep_size:
			expected_extra_sample_delay = min(delay.half_polyblep_size, 0.5/clock_freq)
		#expected_delay = (num_stages - 0.5, 0.5 if linblep else 0)
		expected_delay = (num_stages - 0.5, expected_extra_sample_delay)
		#expected_delay_total_samples = (num_stages - 0.5) / clock_freq

	expected_delay_total_samples = (expected_delay[0]) / clock_freq + expected_delay[1]

	xa = x[t_offset:]
	ya = y_reconstructed[t_offset:]
	xcorr = scipy.signal.correlate(ya, xa)
	lags = scipy.signal.correlation_lags(len(ya), len(xa))

	find_peak_samples_range = [
		int(floor((num_stages - 1) / clock_freq)) - 1,
		int(floor((num_stages + 1.5) / clock_freq))
	]
	
	if (num_stages == 0) and (delay.half_polyblep_size is not None):
		find_peak_samples_range[1] += int(ceil(delay.half_polyblep_size))

	find_peak_lag_indices = [index_of(val, lags) for val in find_peak_samples_range]

	peak_idxs, _ = scipy.signal.find_peaks(xcorr[find_peak_lag_indices[0]:])

	if len(peak_idxs) == 0:
		raise AssertionError(f'Failed to find peaks in range {find_peak_lag_indices[0]}:{len(find_peak_lag_indices)}')

	peak_xcorr_idx = peak_idxs[0] + find_peak_lag_indices[0]

	# Interpolate to find peak in between samples (parabolic - TODO: try sine fit)
	px, peak_xcorr = parabolic_interp_find_peak((xcorr[peak_xcorr_idx - 1], xcorr[peak_xcorr_idx], xcorr[peak_xcorr_idx + 1]))

	actual_delay_total_samples = scale(px, (-1.0, 1.0), (lags[peak_xcorr_idx - 1], lags[peak_xcorr_idx + 1]))

	actual_delay_clocks = actual_delay_total_samples * clock_freq

	actual_delay_clocks = 0.5 * int(floor(actual_delay_clocks * 2.0))
	actual_delay_extra_samples = actual_delay_total_samples - (actual_delay_clocks / clock_freq)
	actual_delay = (actual_delay_clocks, actual_delay_extra_samples)

	extra_delay_samples = actual_delay_total_samples - expected_delay_total_samples

	print('%-60s %6i %9.2f %9.2f %7.1f c + %5.1f s = %6.2f s %6.1f c + %5.2f s = %6.2f s %6.2f s' % (
		title_short,
		num_stages,
		clock_freq,
		1.0 / clock_freq,
		expected_delay[0],
		expected_delay[1],
		expected_delay_total_samples,
		actual_delay[0],
		actual_delay[1],
		actual_delay_total_samples,
		extra_delay_samples
	))

	if ax_c is None:
		return

	line, = ax_c.plot(lags, xcorr, label='XCorr(yr, x)')
	ax_c.plot(actual_delay_total_samples, peak_xcorr, '.', color=line.get_color(), label='XCorr peak')
	ax_c.axvline(expected_delay_total_samples, color='darkgreen', label='Expected delay')
	ax_c.axvline(1/clock_freq, color='orange', label='1 clock period')

	if expected_delay_total_samples:
		ax_c.set_xlim([0, max(actual_delay_total_samples + 1, 2 * expected_delay_total_samples)])
	else:
		ax_c.set_xlim([-2/clock_freq, 2/clock_freq])

	ax_c.set_xlabel(r'$\Delta$t (samples)')
	ax_c.set_ylabel('Cross-Correlation')
	ax_c.grid()
	ax_c.legend(loc='lower right')


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

	fig.suptitle(f'{delay.get_name()}, clock rate change {clock_freq[0]:g} -> {clock_freq[1]:g}')

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
	ax_f.set_ylim([-120, 0])
	ax_f.grid()
	ax_f.legend()


def _plot_clock_rate_ramp(num_stages=32, freq=0.0126, clock_freq_range=(0.1, 2.0), num_samples=2048, **kwargs):

	x = gen_sine(freq, n_samp=num_samples)
	t = np.arange(num_samples)

	clock_rate = np.linspace(clock_freq_range[0], clock_freq_range[1], num_samples, endpoint=True)

	delay = VariableRateDelayLine(num_stages=num_stages, clock_freq=clock_rate[0], **kwargs)
	
	y = np.zeros_like(x)
	debug_info = dict()
	for n, (xn, cn) in enumerate(zip(x, clock_rate)):

		delay.set_clock_freq(cn)
		y[n], debug_out_this_sample = delay.process_sample_debug(xn)

		if n == 0:
			for key, value in debug_out_this_sample.items():
				debug_info[key] = np.zeros_like(x)

		for key, value in debug_out_this_sample.items():
			debug_info[key][n] = value

	clock_phase = debug_info['clock_phase']
	x_interp_t = debug_info['x_interp_t']
	x_interp_val = debug_info['x_interp_val']

	b, a = scipy.signal.butter(N=8, Wn=4.0 * freq)
	y_reconstructed = scipy.signal.filtfilt(b, a, y)

	fig, (ax_t, ax_c) = plt.subplots(2, 1, sharex=True)
	fig.suptitle(f'{delay.get_name()}, clock rate ramp {clock_freq_range[0]:g} -> {clock_freq_range[1]:g}')

	ax_t.plot(t, x, '-', label='X')

	mask = np.isfinite(x_interp_t)
	#ax_t.plot(t[mask] + x_interp_t[mask] - 1, x_interp_val[mask], '.', label='X samples')

	ax_t.plot(t, y, '-', label='Y', zorder=-1)
	ax_t.plot(t, y_reconstructed, '-', label='Y reconstructed', zorder=-2)

	clock_phase_t, clock_phase_y = _interp_clock_phase(clock_phase)
	ax_c.plot(clock_phase_t, clock_phase_y, label='Clock phase', zorder=-2)

	ax_c.plot(t, clock_rate, label='Clock rate', zorder=-1)

	ax_t.set_xlabel('t (samples)')
	ax_t.grid()
	ax_t.legend()

	ax_c.set_xlabel('t (samples)')
	ax_c.grid()
	ax_c.legend()


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
	ax_f.set_ylim([-120, 0])
	ax_f.grid()
	ax_f.legend()


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)
	
	parser.add_argument('--clock', action='store_true', dest='plot_clock_phase', help='Add clock phase to plot')

	parser.add_argument('--basic', action='store_true', help='Plot basic examples (automatically set if no other plot args given)')
	parser.add_argument('--quadratic', action='store_true', help='Plot quadratic vs linear interpolation')
	parser.add_argument('--polyblep', action='store_true', help='Plot polybleps')
	parser.add_argument('--fast', action='store_true', dest='fast_clock', help='Plot high clock rates')
	parser.add_argument('--step', action='store_true', dest='clock_step', help='Plot clock rate step')
	parser.add_argument('--ramp', action='store_true', dest='clock_ramp', help='Plot clock rate ramp')
	parser.add_argument('--chorus', action='store_true', help='Plot chorus')

	return parser


def plot(args, verbose=False):

	if not any([args.fast_clock, args.clock_step, args.clock_ramp, args.chorus, args.quadratic, args.polyblep]):
		args.basic = True

	if verbose and any([args.basic, args.quadratic, args.fast_clock, args.polyblep]):

		print()
		print('%-60s %6s %18s  %27s  %27s  %8s' % (			'', '', 'Clock'.center(18), 'Expected Delay'.center(27), 'Actual Delay'.center(27), 'Diff'.center(8)
		))
		print('%60s %6s %9s %9s %9s %9s %9s %9s %9s %9s %9s' % (
			'', 'Stages', 'Freq', 'Period', 'Clocks', '+ Samples', '= Samples', 'Clocks', '+ Samples', '= Samples', 'Samples'
		))
		print()

	# Fixed clock rate examples

	common_kwargs = dict(
		plot_clock_phase=args.plot_clock_phase,
		verbose=verbose,
	)

	if args.basic or args.polyblep:
		_do_plot(num_stages=0, **common_kwargs)
		_do_plot(num_stages=0, linblep=True, **common_kwargs)
		_do_plot(num_stages=0, polyblep_size=1, **common_kwargs)
		_do_plot(num_stages=0, polyblep_size=2, **common_kwargs)
		_do_plot(num_stages=0, polyblep_size=4, **common_kwargs)
		_do_plot(num_stages=0, polyblep_size=8, **common_kwargs)
		_do_plot(num_stages=0, polyblep_size=16, **common_kwargs)
	if args.basic:
		_do_plot(num_stages=1, ramp_output=False, **common_kwargs)
		_do_plot(num_stages=1, ramp_output=False, linblep=True, **common_kwargs)
	if args.basic or args.quadratic:
		_do_plot(num_stages=1, ramp_output=True, **common_kwargs)
		_do_plot(num_stages=1, quadratic_ramp=True, **common_kwargs)
	if args.basic or args.polyblep:
		_do_plot(num_stages=1, polyblep_size=1, ramp_output=False, **common_kwargs)
		_do_plot(num_stages=1, polyblep_size=2, ramp_output=False, **common_kwargs)
		_do_plot(num_stages=1, polyblep_size=4, ramp_output=False, **common_kwargs)
		_do_plot(num_stages=1, polyblep_size=8, ramp_output=False, **common_kwargs)
		_do_plot(num_stages=1, polyblep_size=16, ramp_output=False, **common_kwargs)
	if args.basic:
		_do_plot(num_stages=16, ramp_output=False, **common_kwargs)
		_do_plot(num_stages=16, polyblep_size=4, ramp_output=False, **common_kwargs)
	if args.basic or args.quadratic:
		_do_plot(num_stages=16, ramp_output=True, **common_kwargs)
		_do_plot(num_stages=16, quadratic_ramp=True, **common_kwargs)

	for clock_freq in [0.91, 1.21, 2.3]:
		for num_stages in [0, 16]:
			if args.fast_clock or args.polyblep:
				_do_plot(num_stages=num_stages, clock_freq=clock_freq, ramp_output=False, **common_kwargs)
			if args.fast_clock:
				_do_plot(num_stages=num_stages, clock_freq=clock_freq, ramp_output=False, linblep=True, **common_kwargs)
			if args.fast_clock or args.quadratic:
				_do_plot(num_stages=max(num_stages, 1), clock_freq=clock_freq, ramp_output=True, **common_kwargs)
				_do_plot(num_stages=max(num_stages, 1), clock_freq=clock_freq, ramp_output=True, quadratic_ramp=True, **common_kwargs)
			if args.polyblep:
				_do_plot(num_stages=num_stages, clock_freq=clock_freq, polyblep_size=8, ramp_output=False, **common_kwargs)

	# Variable clock rate examples

	if args.clock_step:
		_plot_clock_rate_step(ramp_output=False)
		_plot_clock_rate_step(ramp_output=True)

	if args.clock_ramp:
		_plot_clock_rate_ramp(num_stages=0, ramp_output=False)
		_plot_clock_rate_ramp(num_stages=1, ramp_output=False)
		_plot_clock_rate_ramp(num_stages=0, ramp_output=False, linblep=True)
		_plot_clock_rate_ramp(num_stages=0, ramp_output=False, polyblep_size=1)
		_plot_clock_rate_ramp(num_stages=0, ramp_output=False, polyblep_size=2)
		_plot_clock_rate_ramp(num_stages=0, ramp_output=False, polyblep_size=4)
		_plot_clock_rate_ramp(num_stages=0, ramp_output=False, polyblep_size=8)
		_plot_clock_rate_ramp(num_stages=1, ramp_output=True)
		_plot_clock_rate_ramp(num_stages=32, ramp_output=False)
		_plot_clock_rate_ramp(num_stages=32, ramp_output=True)

	if args.chorus:
		_plot_chorus(saw=False)
		_plot_chorus(saw=True)

	plt.show()


def main(args):
	plot(args, verbose=True)
