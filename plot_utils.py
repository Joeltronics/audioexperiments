#!/usr/bin/env python3

import numpy as np
import math
from matplotlib import pyplot as plt
from typing import Union, Optional, Iterable, List

import utils
from processor import ProcessorBase
from filter_base import FilterBase
from freq_response import get_freq_response


def plot_fft(data, sample_rate, nfft=None, log=True, freq_range=(20., 20000.), label=None):

	if nfft is not None:
		data_fft = np.fft.fft(data, n=nfft)
	else:
		data_fft = np.fft.fft(data)

	fft_len = len(data_fft)
	data_len = fft_len // 2
	data_fft = data_fft[0:data_len]

	data_fft = utils.to_dB(np.abs(data_fft))

	f = np.linspace(0, sample_rate/2.0, num=data_len, endpoint=False)
	
	# Make plot based on bin center instead of edge
	f += (f[1] - f[0]) / 2.0

	if log:
		plt.semilogx(f, data_fft, label=label)
	else:
		plt.plot(f, data_fft, label=label)

	plt.xlim(freq_range)
	plt.grid()


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
		freq_args: Optional[List[str]]=None):
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
	:return:
	"""

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

	plt.figure()

	n_plots = 1 + sum([zoom, phase, group_delay])
	main_subplot = (n_plots, 1, 1)
	zoom_subplot = (n_plots, 1, 2) if zoom else None
	group_delay_subplot = (n_plots, 1, n_plots) if group_delay else None
	phase_subplot = (n_plots, 1, (n_plots - 1 if group_delay else n_plots)) if phase else None

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
			extra_args_list = [_format_arg(k, v) for k, v in extra_args.items()]
			full_label = ', '.join(common_args_list + extra_args_list)
			label = ', '.join(extra_args_list)

			if full_label:
				print('Constructing %s(%s)' % (constructor.__name__, full_label))
			else:
				print('Constructing %s' % constructor.__name__)

			p = constructor(**common_args, **extra_args)

			print('Processing %s' % type(p).__name__)

			amps, phases, group_delay = get_freq_response(p, freqs, sample_rate, n_samp=n_samp, group_delay=True)

			amps = utils.to_dB(amps)

			max_amp_seen = max(max_amp_seen, np.amax(amps))
			min_amp_seen = min(min_amp_seen, np.amin(amps))

			phases_deg = np.rad2deg(phases)
			phases_deg = (phases_deg + 180.) % 360 - 180.

			plt.subplot(*main_subplot)
			plt.semilogx(freqs, amps, label=label)

			if zoom_subplot:
				plt.subplot(*zoom_subplot)
				plt.semilogx(freqs, amps, label=label)

			if phase_subplot:
				plt.subplot(*phase_subplot)
				plt.semilogx(freqs, phases_deg, label=label)

			if group_delay_subplot:
				plt.subplot(*group_delay_subplot)
				plt.semilogx(freqs, group_delay, label=label)

	types_list = [type.__name__ for type in constructors]
	plot_title = ', '.join(types_list) + '; '

	if common_args_list:
		plot_title += ', '.join(common_args_list) + '; '

	plot_title += 'sample rate ' + utils.to_pretty_str(sample_rate / 1000.0, point_zero=False) + ' kHz'

	plt.subplot(*main_subplot)
	plt.title(plot_title)
	plt.ylabel('Amplitude (dB)')

	max_amp = math.ceil(max_amp_seen / 6.0) * 6.0
	min_amp = math.floor(min_amp_seen / 6.0) * 6.0

	plt.yticks(np.arange(min_amp, max_amp + 6, 6))
	plt.ylim([max(min_amp, -60.0), max(max_amp, 6.0)])
	plt.grid()
	if add_legend:
		plt.legend()

	if zoom_subplot:
		plt.subplot(*zoom_subplot)
		plt.ylabel('Amplitude (dB)')

		max_amp = math.ceil(max_amp_seen / 3.0) * 3.0
		min_amp = math.floor(min_amp_seen / 3.0) * 3.0

		yticks = np.arange(min_amp, max_amp + 3, 3)
		plt.yticks(yticks)
		plt.ylim([max(min_amp, -6.0), min(max_amp, 6.0)])
		plt.grid()

	if phase_subplot:
		plt.subplot(*phase_subplot)
		plt.ylabel('Phase')
		plt.grid()
		plt.yticks([-180, -90, 0, 90, 180])

	if group_delay_subplot:
		plt.subplot(*group_delay_subplot)
		plt.grid()
		plt.ylabel('Group delay')

	# Whatever the final subplot is
	plt.subplot(n_plots, 1, n_plots)
	plt.xlabel('Freq')
