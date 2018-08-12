#!/usr/bin/env python3


import numpy as np
from filter import BiquadLowpass
import overdrive
import math


class Rossum92Biquad(BiquadLowpass):
	"""
	2-pole lowpass filter based on Dave Rossum's 1992 paper
	"Making Digital Filters Sound 'Analog'"
	"""

	def __init__(self, wc, Q=0.5, verbose=False):
		super().__init__(wc, Q=Q, verbose=verbose)

	def process_sample(self, x: float) -> float:
		# Assign these to make the math readable below
		a1 = self.a1; a2 = self.a2
		b0 = self.b0; b1 = self.b1; b2 = self.b2

		x1 = self.x1; x2 = self.x2
		y1 = self.y1; y2 = self.y2

		# DF1
		y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

		# Update state
		self.x2 = self.x1; self.x1 = overdrive.tanh(x)
		self.y2 = self.y1; self.y1 = y

		return y

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		y = np.zeros_like(vec)
		for n, x in enumerate(vec):
			y[n] = self.process_sample(x)
		return y


class OverdrivenInputBiquad(BiquadLowpass):
	"""
	2-pole lowpass filter with just the input overdriven
	"""

	def __init__(self, wc, Q=0.5, verbose=False):
		super().__init__(wc, Q=Q, verbose=verbose)

	def process_sample(self, x: float) -> float:
		return super().process_sample(overdrive.tanh(x))

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		return super().process_vector(overdrive.tanh(vec))


def main():
	import signal_generation
	from matplotlib import pyplot as plt
	import wavfile
	import argparse
	import utils
	from utils import to_pretty_str

	parser = argparse.ArgumentParser()
	parser.add_argument('outfile', nargs='?')
	args = parser.parse_args()

	freq = 220.

	freq1, freq2, freq3 = freq + 0.6, freq + 0.291, freq - 0.75

	cutoff_start = 10000.
	cutoff_end = 100.

	sample_rate = 96000
	n_samp = sample_rate * 2

	wc = cutoff_start / sample_rate

	Q = 1.0 / math.sqrt(2.0)  # 0.7071

	filters = [
		(BiquadLowpass, dict(wc=wc, Q=Q, gain=1.0)),
		(Rossum92Biquad, dict(wc=wc, Q=Q, gain=1.0)),
		(OverdrivenInputBiquad, dict(wc=wc, Q=Q, gain=1.0)),

		(BiquadLowpass, dict(wc=wc, Q=2.0, gain=1.0)),
		(Rossum92Biquad, dict(wc=wc, Q=2.0, gain=1.0)),
		(OverdrivenInputBiquad, dict(wc=wc, Q=2.0, gain=1.0)),

		(Rossum92Biquad, dict(wc=wc, Q=Q, gain=10.0)),
		(OverdrivenInputBiquad, dict(wc=wc, Q=Q, gain=10.0)),
	]

	saws = \
		signal_generation.gen_saw(freq1 / sample_rate, n_samp) + \
		signal_generation.gen_saw(freq2 / sample_rate, n_samp) + \
		signal_generation.gen_saw(freq3 / sample_rate, n_samp)

	saws /= 3.0

	t = signal_generation.sample_time_index(n_samp, sample_rate)

	plt.figure()

	plt.plot(t, saws, label='Input')

	data_out = np.copy(saws) if args.outfile else None

	for constructor, filt_args in filters:

		args_list = ', '.join([
			'%s=%s' % (k, to_pretty_str(v, num_decimals=3))
			for k, v in filt_args.items()])

		name = '%s(%s)' % (constructor.__name__, args_list)

		print('Processing %s' % name)

		gain = filt_args.pop('gain') if 'gain' in filt_args else 1.0

		filt = constructor(**filt_args)

		x = saws * gain
		y = filt.process_freq_sweep(
			x,
			cutoff_start / sample_rate,
			cutoff_end / sample_rate,
			log=True)
		y /= gain

		if data_out is not None:
			data_out = np.append(data_out, y)

		plt.plot(t, y, label=name)

	plt.xlabel('Time (s)')
	plt.legend()

	plt.grid()

	if args.outfile:
		print('Saving %s' % args.outfile)
		data_out = utils.normalize(data_out)
		wavfile.export_wavfile(data_out, sample_rate, args.outfile, allow_overwrite=True)

	plt.show()


if __name__ == "__main__":
	main()
