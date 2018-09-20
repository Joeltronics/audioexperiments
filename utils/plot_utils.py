#!/usr/bin/env python3

import numpy as np
import math
from matplotlib import pyplot as plt
from typing import Union, Optional, Iterable, List, Callable, Tuple

from utils import utils
from analysis.freq_response import get_freq_response


def plot_fft(
		data,
		sample_rate,
		fmt=None,
		nfft: Optional[int]=None,
		log=True,
		dB=True,
		window: Union[bool, Callable]=True,
		label=None,
		noise_floor=0.0):
	"""

	:param data: time-domain data
	:param sample_rate: sample rate of data, in Hz
	:param fmt: pyplot format
	:param nfft:
	:param log: plot as log-frequency
	:param dB: display FFT in dB
	:param window:
	:param label: pyplot label
	:param noise_floor: add a constant amount to FFT result, so that zero-values don't screw up dB conversion
	:return:
	"""

	if window is True:
		data = data * np.hamming(len(data))
	elif window:
		data = data * window(len(data))

	if nfft is not None:
		data_fft = np.fft.fft(data, n=nfft)
	else:
		data_fft = np.fft.fft(data)

	if fmt is None:
		fmt = '-'

	fft_len = len(data_fft)
	data_len = fft_len // 2
	data_fft = data_fft[0:data_len]
	data_fft = np.abs(data_fft)
	data_fft += noise_floor

	if dB:
		data_fft = utils.to_dB(data_fft)

	f = np.linspace(0, sample_rate/2.0, num=data_len, endpoint=False)
	
	# Make plot based on bin center instead of edge
	f += (f[1] - f[0]) / 2.0

	if log:
		plt.semilogx(f, data_fft, fmt, label=label)
	else:
		plt.plot(f, data_fft, fmt, label=label)


def plot_spectrogram(data, sample_rate, nfft=256, log=False):

	plt.specgram(data, NFFT=nfft, Fs=sample_rate)

	if log:
		ax = plt.gca()
		ax.set_yscale('log')
		plt.ylim([20.0, 0.5*sample_rate])


def plot_freq_resp(
		constructors,
		common_args: Optional[dict],
		args_list: Union[None, dict, Iterable[dict]],
		freqs: Union[np.ndarray, List[Union[float, int]]],
		sample_rate=48000.,
		n_samp: Optional[int]=None,
		zoom=False,
		phase=False,
		group_delay=False,
		freq_args: Optional[List[str]]=None,
		axes=None,
		main_plot_ylim=None):
	"""

	:param constructors: one or more ProcessorBase constructors
	:param common_args: args to be passed to all constructors
	:param args_list: changing args to be passed to constructors
	:param freqs: frequencies to perform DFT at
	:param sample_rate:
	:param n_samp:
	:param zoom: add subplot zoomed in to unity gain
	:param phase: add subplot with phase response (in degrees)
	:param group_delay: add subplot with group delay
	:param freq_args: args in args_list with these names will be displayed multiplied by sample rate, with units
	:param axes: axes to plot into - if given, length must match 1 + sum([zoom, phase, group_delay])
	:return:
	"""

	#
	# Handle subplots
	#

	n_plots = 1 + sum([zoom, phase, group_delay])
	if axes is None:
		plt.figure()
	elif len(axes) != n_plots:
		raise ValueError('Must give same number of axes as plots!')

	main_subplot = plt.subplot(n_plots, 1, 1) if axes is None else axes[0]

	if not zoom:
		zoom_subplot = None
	elif axes is None:
		zoom_subplot = plt.subplot(n_plots, 1, 2)
	else:
		zoom_subplot = axes[2]

	if not phase:
		phase_subplot = None
	elif axes is None:
		phase_subplot = plt.subplot(n_plots, 1, n_plots - 1 if group_delay else n_plots)
	else:
		phase_subplot = axes[-2] if group_delay else axes[-1]

	if not group_delay:
		group_delay_subplot = None
	elif axes is None:
		group_delay_subplot = plt.subplot(n_plots, 1, n_plots)
	else:
		group_delay_subplot = axes[-1]

	final_subplot = plt.subplot(n_plots, 1, n_plots) if axes is None else axes[-1]

	#
	# Sanitize args
	#

	if n_samp is None:
		n_samp = int(math.ceil(sample_rate / 4.))

	if common_args is None:
		common_args = dict()

	if args_list is None:
		add_legend = False
		args_list = [dict()]
	else:
		add_legend = True

	if not hasattr(constructors, "__iter__"):
		constructors = [constructors]

	if isinstance(args_list, dict):
		args_list = [args_list]

	#
	# Process
	#

	max_amp_seen = 0.0
	min_amp_seen = 0.0

	def _format_arg(key, value):
		if freq_args and key in freq_args:
			value *= sample_rate
			if value >= 1000.0:
				value /= 1000.0
				units = 'kHz'
			else:
				units = 'Hz'
			val_fmt = utils.to_pretty_str(value, num_decimals=3, point_zero=False)
			return '%s=%s %s' % (key, val_fmt, units)
		else:
			return '%s=%s' % (key, utils.to_pretty_str(value))

	common_args_list = [_format_arg(k, v) for k, v in common_args.items()]

	for constructor in constructors:
		for extra_args in args_list:

			this_args = common_args.copy()
			this_args.update(extra_args)

			full_label = ', '.join([_format_arg(k, v) for k, v in this_args.items()])
			label = ', '.join([_format_arg(k, v) for k, v in extra_args.items()])

			if full_label:
				print('Constructing %s(%s)' % (constructor.__name__, full_label))
			else:
				print('Constructing %s' % constructor.__name__)

			p = constructor(**this_args)

			print('Processing %s' % type(p).__name__)

			amps, phases, group_delay = get_freq_response(p, freqs, sample_rate, n_samp=n_samp, group_delay=True)

			amps = utils.to_dB(amps)

			max_amp_seen = max(max_amp_seen, np.amax(amps))
			min_amp_seen = min(min_amp_seen, np.amin(amps))

			phases_deg = np.rad2deg(phases)
			phases_deg = (phases_deg + 180.) % 360 - 180.

			main_subplot.semilogx(freqs, amps, label=label)

			if zoom_subplot is not None:
				zoom_subplot.semilogx(freqs, amps, label=label)

			if phase_subplot is not None:
				phase_subplot.semilogx(freqs, phases_deg, label=label)

			if group_delay_subplot is not None:
				group_delay_subplot.semilogx(freqs, group_delay, label=label)

	types_list = [type.__name__ for type in constructors]
	plot_title = ', '.join(types_list) + '; '

	if common_args_list:
		plot_title += ', '.join(common_args_list) + '; '

	plot_title += 'sample rate ' + utils.to_pretty_str(sample_rate / 1000.0, point_zero=False) + ' kHz'

	main_subplot.set_title(plot_title)
	main_subplot.set_ylabel('Amplitude (dB)')

	max_amp = math.ceil(max_amp_seen / 6.0) * 6.0
	min_amp = math.floor(min_amp_seen / 6.0) * 6.0

	main_subplot.set_yticks(np.arange(min_amp, max_amp + 6, 6))

	if main_plot_ylim is not None:
		main_subplot.set_ylim(main_plot_ylim)
	else:
		main_subplot.set_ylim([max(min_amp, -60.0), max(max_amp, 6.0)])

	main_subplot.grid()
	if add_legend:
		main_subplot.legend()

	if zoom_subplot is not None:
		zoom_subplot.set_ylabel('Amplitude (dB)')

		max_amp = math.ceil(max_amp_seen / 3.0) * 3.0
		min_amp = math.floor(min_amp_seen / 3.0) * 3.0

		yticks = np.arange(min_amp, max_amp + 3, 3)
		zoom_subplot.set_yticks(yticks)
		zoom_subplot.set_ylim([max(min_amp, -6.0), min(max_amp, 6.0)])
		zoom_subplot.grid()

	if phase_subplot is not None:
		phase_subplot.set_ylabel('Phase')
		phase_subplot.grid()
		phase_subplot.set_yticks([-180, -90, 0, 90, 180])

	if group_delay_subplot is not None:
		group_delay_subplot.grid()
		group_delay_subplot.set_ylabel('Group delay')

	# Whatever the final subplot is
	final_subplot.set_xlabel('Freq')
