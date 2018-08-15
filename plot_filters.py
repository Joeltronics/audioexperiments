#!/usr/bin/env python3

import numpy as np
import utils
from matplotlib import pyplot as plt
import math
from typing import Union, Optional, Iterable, List
from filter_base import FilterBase
from freq_response import get_freq_response


def plot_filters(
		constructors: Union[FilterBase, Iterable[FilterBase]],
		args_list: Optional[Iterable[dict]],
		freqs: Union[np.ndarray, List[Union[float, int]]],
		sample_rate=48000.,
		default_cutoff=1000.,
		n_samp: Optional[int]=None,
		zoom=True,
		phase=True,
		group_delay=False):

	if n_samp is None:
		n_samp = int(math.ceil(sample_rate / 4.))

	plt.figure()

	if args_list is None:
		add_legend = False
		args_list = [dict()]
	else:
		add_legend = True

	max_amp_seen = 0.0
	min_amp_seen = 0.0

	if not hasattr(constructors, "__iter__"):
		constructors = [constructors]

	n_plots = 1 + sum([zoom, phase, group_delay])
	main_subplot = (n_plots, 1, 1)
	zoom_subplot = (n_plots, 1, 2) if zoom else None
	group_delay_subplot = (n_plots, 1, n_plots) if group_delay else None
	phase_subplot = (n_plots, 1, (n_plots - 1 if group_delay else n_plots)) if phase else None

	for filter_type in constructors:

		for extra_args in args_list:

			label = ', '.join(['%s=%s' % (key, utils.to_pretty_str(value)) for key, value in extra_args.items()])
			if label:
				print('Processing %s, %s' % (filter_type.__name__, label))
			else:
				print('Processing %s' % filter_type.__name__)

			if 'cutoff' in extra_args.keys():
				cutoff = extra_args['cutoff']
				extra_args.pop('cutoff')
			else:
				cutoff = default_cutoff

			if 'f_norm' in extra_args.keys():
				extra_args['w_norm'] = extra_args['f_norm'] / sample_rate
				extra_args.pop('f_norm')

			filt = filter_type(wc=(cutoff / sample_rate), verbose=True, **extra_args)
			amps, phases, group_delay = get_freq_response(filt, freqs, sample_rate, n_samp=n_samp, group_delay=True)

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

	name = ', '.join([type.__name__ for type in constructors])

	plt.subplot(*main_subplot)
	plt.title('%s, sample rate %.0f' % (name, sample_rate))
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
