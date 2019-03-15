#!/usr/bin/env python3

from .lfsr import get_maximal_lfsr
from utils import wavfile

import argparse
import math
import numpy as np
import os


def main(args):
	parser = argparse.ArgumentParser()
	parser.add_argument('bits', nargs='+', type=int, help='Bit depths of cascaded LFSRs')
	parser.add_argument('filename', type=str, help='Output filename')
	#parser.add_argument('-f', '--freq', type=float, help='Fundamental frequency', default=441.0)
	parser.add_argument('-s', '--samplerate', type=int, help='Sample rate (Hz)', default=44100)
	parser.add_argument('-d', '--duration', type=float, help='Duration in seconds', default=10.)
	parser.add_argument('--overwrite', action='store_true', help='Allow overwrite of output file')
	args = parser.parse_args(args)

	final_lfsr_period = args.bits[-1]

	duration_samples = int(math.ceil(args.duration * args.samplerate))

	data = np.zeros(duration_samples)

	clock_lfsrs = [get_maximal_lfsr(bits, verbose=True) for bits in args.bits[:-1]]
	last_lfsr = get_maximal_lfsr(args.bits[-1], verbose=True)

	base_clock_period = 20

	if os.path.exists(args.filename) and not args.overwrite:
		raise FileExistsError(args.filename)

	print('Processing')

	ampl = 0.5

	clock_lfsr_last_state = [False for _ in clock_lfsrs]

	out = -ampl
	for sample_idx in range(duration_samples):

		if (sample_idx % base_clock_period) == 0:

			for lfsr_idx, lfsr in enumerate(clock_lfsrs):

				this_lfsr_out = lfsr()
				this_lfsr_prev_out = clock_lfsr_last_state[lfsr_idx]
				clock_lfsr_last_state[lfsr_idx] = this_lfsr_out

				# Rising edge only
				if this_lfsr_out and not this_lfsr_prev_out:
					pass
				else:
					break

			else:
				out = ampl if last_lfsr() else -ampl

		data[sample_idx] = out

	print('Saving %s' % args.filename)
	wavfile.export_wavfile(data, sample_rate=args.samplerate, filename=args.filename, allow_overwrite=args.overwrite)
