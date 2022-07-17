#!/usr/bin/env python3

"""
Delay with variable rate (e.g. clock rate or tape speed) but fixed length (e.g. stages/samples/distance)
Equivalent to behavior of BBD, PT2399, or most tape delays
"""

import argparse
from math import ceil
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Tuple

from delay_reverb.delay_line import DelayLine
from generation.signal_generation import gen_sine, gen_saw
from processor import ProcessorBase
from utils.utils import lerp, reverse_lerp, to_dB


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
	"""
	def __init__(self, num_stages: int, clock_freq: float, ramp_output=False):
		"""
		:param num_stages: number of delay line stages
		:param clock_freq: clock frequency
		:param ramp_output:
			If False, will use zero-order hold, for behavior akin to BBD without reconstruction filter. Will alias
				badly - not just around the clock rate (which may actually be desirable), but also around the operating
				sample rate (which is undesired).
			If True, will linearly interpolate output samples, for behavior more akin to tape. Also helps limit unwanted
				aliasing.
		"""

		self.num_stages = num_stages
		self.clock_freq = clock_freq
		self.ramp_output = ramp_output

		self.clock_phase = 0.0
		self.x_prev = 0.0
		self.y_prev = 0.0
		self.y_curr = 0.0
		self.delay_line = DelayLine(num_stages) if self.num_stages > 0 else None

	def set_clock_freq(self, clock_freq: float):
		# TODO: handle this case better (it will alias, but could still at least attempt to handle it)
		if clock_freq >= 1.0:
			raise ValueError('Clock rate too high!')
		self.clock_freq = clock_freq

	def reset(self) -> None:
		self.clock_phase = 0.0
		self.x_prev = 0.0
		self.y_curr = 0.0
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

		if clock_phase_1 >= 1.0:

			assert clock_phase_1 < 2.0

			x_interp_t = reverse_lerp((clock_phase_0, clock_phase_1), 1.0)

			# TODO: optional higher-order interpolation
			x_interp_val = lerp((self.x_prev, x), x_interp_t)

			# TODO: Optionally use polyblep to smooth step
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


def _interp_clock_phase(clock_phase):
	# Interpolate clock phase to get precise zero crossing locations

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
		num_samples=1024,
		ramp_output=False,
		plot_clock_phase=False,
		):

	t_offset = int(ceil(num_stages / clock_freq))
	plot_t_range = [t_offset, t_offset + num_samples_plot]
	if plot_t_range[1] > num_samples:
		raise ValueError('Must set num_samples higher at this combination of num_stages & clock_freq')

	x = gen_sine(freq, n_samp=num_samples)
	t = np.arange(num_samples)

	delay = VariableRateDelayLine(num_stages=num_stages, clock_freq=clock_freq, ramp_output=ramp_output)
	y, debug_info = delay.process_vector_debug(x)

	clock_phase = debug_info['clock_phase']
	x_interp_t = debug_info['x_interp_t']
	x_interp_val = debug_info['x_interp_val']

	fig, (ax_t, ax_f) = plt.subplots(2, 1)
	if num_stages:
		fig.suptitle(f'BBD style delay ({num_stages} stages), clock rate {clock_freq:g}')
	else:
		fig.suptitle(f'Zero-order hold, clock rate {clock_freq:g}')

	ax_f.axvline(freq, label='X frequency')
	ax_f.axvline(clock_freq, label='Clock', color='orange')

	ax_t.plot(t, x, '.-', label='X')
	
	fft_x, f = do_fft(x, window=True)
	ax_f.plot(f, fft_x, label='X')

	mask = np.isfinite(x_interp_t)
	ax_t.plot(t[mask] + x_interp_t[mask] - 1, x_interp_val[mask], '.', label='X samples')

	line, = ax_t.plot(t, y, '.-', label='Y')

	fft_y, f = do_fft(y, window=True)
	ax_f.plot(f, fft_y, label='Y', color=line.get_color(), zorder=-1)

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


def _plot_clock_rate_step(num_stages=32, freq=0.0126, clock_freq=(0.13, 0.21), num_samples=4096, ramp_output=False):

	x = gen_sine(freq, n_samp=num_samples)
	t = np.arange(num_samples)

	delay = VariableRateDelayLine(num_stages=num_stages, clock_freq=clock_freq[0], ramp_output=ramp_output)
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

	pass


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--clock', action='store_true', dest='plot_clock_phase', help='Plot clock phase')
	return parser


def plot(args):
	_do_plot(num_stages=0, plot_clock_phase=args.plot_clock_phase, ramp_output=False)
	_do_plot(num_stages=16, plot_clock_phase=args.plot_clock_phase, ramp_output=False)
	_do_plot(num_stages=0, plot_clock_phase=args.plot_clock_phase, ramp_output=True)
	_do_plot(num_stages=16, plot_clock_phase=args.plot_clock_phase, ramp_output=True)
	_plot_clock_rate_step(ramp_output=False)
	_plot_clock_rate_step(ramp_output=True)
	_plot_chorus(saw=False)
	_plot_chorus(saw=True)
	plt.show()


def main(args):
	plot(args)
