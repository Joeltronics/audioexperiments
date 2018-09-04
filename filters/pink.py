#!/usr/bin/env python3

from .filter_base import IIRFilter
import numpy as np
from utils import utils
from typing import Tuple, List
from .one_pole import BasicOnePole
from processor import ParallelProcessors
from .filter_base import ParallelFilters


class BasicPinkFilter(IIRFilter):
	"""Pink noise (1/f) filter
	Intended for 44.1-48 kHz; will work at higher sample rates but low frequency response will be off
	"""
	def __init__(self, wc=None, verbose=False):
		# https://ccrma.stanford.edu/~jos/sasp/Example_Synthesis_1_F_Noise.html
		super().__init__(
			a=[1, -2.494956002, 2.017265875, -0.522189400],
			b=[0.049922035, -0.095993537, 0.050612699, -0.004408786])


class PinkFilter(ParallelProcessors):
	def __init__(self, sample_rate):
		self.sample_rate = sample_rate
		freqs, gains = self._calc_individual_filters(self.sample_rate)
		filters = [BasicOnePole(wc=f/sample_rate, gain=g) for f, g in zip(freqs, gains)]
		super().__init__(filters)

	@staticmethod
	def _calc_individual_filters(sample_rate, pole_zero_ratio=2.0) -> Tuple[np.ndarray, np.ndarray]:
		# http://www.firstpr.com.au/dsp/pink-noise/#Filtering
		# http://www.cooperbaker.com/home/code/pink%20noise/

		freqs = []
		gains = []

		gain = 1.0
		hz = sample_rate / (2 * np.pi)

		while hz > 1:
			freqs.append(hz)
			gains.append(gain)

			gain *= pole_zero_ratio
			hz /= (2.0 * pole_zero_ratio)

		freqs = np.array(freqs)
		gains = np.array(gains) / sum(gains)

		return freqs, gains


def test(verbose=False):
	import numpy as np
	from unit_test.processor_unit_test import ProcessorUnitTest
	from unit_test.unit_test import run_unit_tests

	freqs = np.array([50., 100., 200., 400., 800., 1600., 3200., 6400., 12800.]) / 44100.
	expected_vals = -3.0 * np.arange(len(freqs)) - 4.0
	expected_dB = [(val - 0.5, val + 0.5) for val in expected_vals]

	tests = [
		ProcessorUnitTest(
			"BasicPinkFilter()",
			lambda: BasicPinkFilter(),
			freqs_to_test=freqs,
			expected_freq_response_range_dB=expected_dB,
			expected_phase_response_range_degrees=None,
			deterministic=True,
			linear=True
		)
	]
	return run_unit_tests(tests, verbose=verbose)


def plot(args):
	from matplotlib import pyplot as plt
	import numpy as np

	from utils.plot_utils import plot_freq_resp

	sample_rate = 48000.

	freqs = np.logspace(np.log10(1.0), np.log10(20000.0), 32, base=10)

	plot_freq_resp(BasicPinkFilter, None, None, freqs, sample_rate, n_samp=48000)
	plot_freq_resp(PinkFilter, None, dict(sample_rate=48000), freqs, 48000, n_samp=48000)
	plot_freq_resp(PinkFilter, None, dict(sample_rate=96000), freqs, 96000, n_samp=96000)

	freqs, gains = PinkFilter._calc_individual_filters(sample_rate)
	args_list = [dict(wc=freq/sample_rate, gain=gain) for freq, gain in zip(freqs, gains)]
	plot_freq_resp(BasicOnePole, None, args_list, freqs, 48000, n_samp=48000, freq_args=['wc'])
	plt.title('Individual filters in PinkFilter')

	print('Showing plots')
	plt.show()


def main(args):
	test(args)
	plot(args)
