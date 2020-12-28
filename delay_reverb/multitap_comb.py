#!/usr/bin/env python3

import argparse
from matplotlib import pyplot as plt
import numpy as np
import scipy.fft

from .delay_line import DelayLine
from utils import utils

"""
First, something indented with tabs so that the block diagrams
	below don't make the IDE think the file uses spaces

Idea & derivations:

The idea is to plot the frequency response of a comb filter where the feedback time is not the
same as the delay time, like with an MN3011-based flanger like the ADA STD-1, or a multi-tap tape
echo where the feedback can come from a different tap as the output, like the Strymon Volante

First, here's a basic comb filter:

                       fb
              -------- (x) <-----
              |                 |
              V                 |
        ---> (+) ---> [ dl ] ---+----> [     ]
        |                              [     ]
x[n] ---+                              [ mix ] -----> y[n]
        |                              [     ]
        -----------------------------> [     ]

Series delay, where feedback is shorter than output

                       fb
              +------- (x) <-----+
              |                  |
              V                  |
        +--> (+) ---> [ dl ] ----+---> [ dl ] ---> [     ]
        |                                          [     ]
x[n] ---+                                          [ mix ] -----> y[n]
        |                                          [     ]
        +----------------------------------------> [     ]

Series delay, where feedback is longer than output:

                       fb
              +------- (x) <-------------------+
              |                                |
              V                                |
        +--> (+) ---> [ dl ] ---+---> [ dl ] --+
        |                       |
        |                       +---------------> [     ]
        |                                         [     ]
x[n] ---+                                         [ mix ] -----> y[n]
        |                                         [     ]
        +---------------------------------------> [     ]

Now, assuming the delay line is LTI, then these are equivalent:

                 +--> y1
                 |
x ----> [ dl ] --+
                 |
                 +--> y2

    +-> [ dl ] -----> y1
    |
x --+
    |
    +-> [ dl ] -----> y2

(In fact, this is also true if it's non-LTI, as long as the delay lines have the same
nonlinearities & time-variance)

2 delay lines can also be combined into 1:

x ---> [ dl 1 ] ---> [ dl 2 ] ---> y
x ---> [ dl 1+2 ] ---> y

(Again, this is subject to the same assumptions about nonlinearities/time-variance)

That means we can separate the 2 series delay cases:

                       fb
              +------- (x) <-----+
              |                  |
              V                  |
        +--> (+) ---> [ dl ] ----+---> [ dl ] ---> [     ]
        |                                          [     ]
x[n] ---+                                          [ mix ] -----> y[n]
        |                                          [     ]
        +----------------------------------------> [     ]


              +------- (x) <-----+
              |                  |
              |                  |
              |   +-> [ dl ] ----+
              V   |
        +--> (+) -+
        |         |
        |         +-> [ dl ] --------> [ dl ] ---> [     ]
        |                                          [     ]
x[n] ---+                                          [ mix ] -----> y[n]
        |                                          [     ]
        +----------------------------------------> [     ]


              +------- (x) <-----+
              |                  |
              |                  |
              |   +-> [ dl ] ----+
              V   |
        +--> (+) -+
        |         |
        |         +-> [ dl ] ---> [     ]
        |                         [     ]
x[n] ---+                         [ mix ] -----> y[n]
        |                         [     ]
        +-----------------------> [     ]

Or, in the longer feedback case:

                       fb
              +------- (x) <-------------------+
              |                                |
              V                                |
        +--> (+) ---> [ dl ] ---+---> [ dl ] --+
        |                       |
        |                       +---------------> [     ]
        |                                         [     ]
x[n] ---+                                         [ mix ] -----> y[n]
        |                                         [     ]
        +---------------------------------------> [     ]

                       fb
              +------- (x) <---------------+
              |                            |
              |                            |
              |   +-> [ dl ] ---> [ dl ] --+
              V   |
        +--> (+) -+
        |         |
        |         +-> [ dl ] -------------------> [     ]
        |                                         [     ]
x[n] ---+                                         [ mix ] -----> y[n]
        |                                         [     ]
        +---------------------------------------> [     ]

              +------- (x) <-----+
              |                  |
              |                  |
              |   +-> [ dl ] ----+
              V   |
        +--> (+) -+
        |         |
        |         +-> [ dl ] ---> [     ]
        |                         [     ]
x[n] ---+                         [ mix ] -----> y[n]
        |                         [     ]
        +-----------------------> [     ]

This is the same as the shorter-feedback case!

"""


def parse_args(args):
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-d', '--delay', dest='delay_time', metavar='MS',
		type=float, default=1,
		help='Delay (ms); default 1; will be rounded to integer number of samples',
	)
	parser.add_argument(
		'-s', '--sample-rate', metavar='HZ', dest='sample_rate',
		type=int, default=48000,
		help='Sample Rate (Hz); default 48000',
	)

	grp = parser.add_argument_group('mix')

	grp.add_argument(
		'-m', '--mix', dest='mix',
		type=float, default=None,
		help='Mix (sets both --wet and --dry); can be negative; default 0.5',
	)
	grp.add_argument(
		'--wet', dest='wet_mix',
		type=float, default=None,
		help='Wet mix; can be negative; default 0.5',
	)
	grp.add_argument(
		'--dry', dest='dry_mix',
		type=float, default=None,
		help='Dry mix; can be negative; default 0.5',
	)

	grp = parser.add_argument_group('Feedback')

	grp.add_argument(
		'-f', '--feedback', metavar='FB', dest='feedback',
		type=float, default=0.0,
		help='Feedback amount; valid range (-1 < FB < 1)',
	)
	grp.add_argument(
		'--fdelay', metavar='MS', dest='feedback_delay_time',
		type=float, default=None,
		help='Feedback delay time (ms); default same as delay time; will be rounded to integer number of samples',
	)

	grp = parser.add_argument_group('Analysis')

	grp.add_argument(
		'--eps', metavar='dB', dest='eps_dB',
		type=float, default=-120.0,
		help='Epsilon value - stop processing once level below this; default -120 dB'
	)
	grp.add_argument(
		'--min-len', metavar='SAMPLES', dest='min_len',
		type=int, default=1024,
		help='Minimum number of samples to process; default 1024',
	)
	grp.add_argument(
		'--max-len', metavar='SAMPLES', dest='max_len',
		type=int, default=65536,
		help='Maximum number of samples to process; default 65536',
	)
	grp.add_argument(
		'--phase', dest='show_phase',
		action='store_true',
		help='Show phase plot',
	)

	args = parser.parse_args(args)

	if not (-1.0 < args.feedback < 1.0):
		raise ValueError('Feedback must be in range (-1 < fb < 1)')

	if args.mix is not None:

		if (args.wet_mix is not None) or (args.dry_mix is not None):
			raise ValueError('Cannot give --dry or --wet if also giving -m/--mix')

		args.wet_mix = args.mix
		args.dry_mix = 1.0 - abs(args.wet_mix)

	else:

		if args.wet_mix is None:
			args.wet_mix = 0.5
		
		if args.dry_mix is None:
			args.dry_mix = 0.5

	return args


def determine_buffer_size(delay_time: int, fb: float, wet_mix=1.0, eps_dB=-120) -> int:

	# TODO: Is the +1 actually necessary?

	if not fb:
		return delay_time + 1

	fb = abs(fb)

	if not (fb < 1.0):
		raise ValueError('Feedback must be in range (-1 < fb < 1)')

	eps = utils.from_dB(eps_dB)

	# TODO: solve this properly (not iteratively)
	num_periods = 1
	level = abs(wet_mix)
	while level > eps:
		level *= fb
		num_periods += 1

	return num_periods * delay_time + 1


def process(x: np.ndarray, dry_mix, wet_mix, fb, delay_line, fb_delay_line=None) -> np.ndarray:

	if not wet_mix:
		return x

	y = np.zeros_like(x)


	if fb_delay_line is None:
		
		# Simple case - just 1 delay line

		for n in range(len(y)):

			"""
			                       fb
			              +------- (x) <----+
			              |                 |
			              V                 |
			        +--> (x) ---> [ dl ] ---+---> [     ]
			        |                             [     ]
			x[n] ---+                             [ mix ] ---> y[n]
			        |                             [     ]
			        +---------------------------> [     ]
			"""

			xn = x[n]

			wet = delay_line.peek_front()

			delay_in = xn + (wet * fb)
			delay_line.push_back(delay_in)

			yn = xn * dry_mix + wet * wet_mix

			y[n] = yn

	else:

		# Complex case - separate feedback delay line

		for n in range(len(y)):

			"""
			                           fb
			              +----------- (x) <----+
			              |                     |
			              |                     |
			              |     +---> [ dl ] ---+
			              V     |
			        +--> (+) ---+
			        |           |
			        |           +---> [ dl ] ---> [     ]
			        |                             [     ]
			x[n] ---+                             [ mix ] ---> y[n]
			        |                             [     ]
			        +---------------------------> [     ]
			"""

			xn = x[n]

			wet = delay_line.peek_front()

			fb_out = fb_delay_line.peek_front()
			delay_in = xn + (fb_out * fb)
			delay_line.push_back(delay_in)
			fb_delay_line.push_back(delay_in)

			yn = xn * dry_mix + wet * wet_mix

			y[n] = yn

	return y


def do_plot(x: np.ndarray, y: np.ndarray, sample_rate: int, show_phase=False):

	if len(x) != len(y):
		raise ValueError('len(x) != len(y)')

	# No windowing (this is intentional)

	fft_x = scipy.fft.fft(x)
	fft_x = fft_x[:len(fft_x) // 2]

	fft_y = scipy.fft.fft(y)
	fft_y = fft_y[:len(fft_y) // 2]

	f = np.fft.fftfreq(len(x), d=1.0/sample_rate)
	f = f[:len(f) // 2]

	amp_x = utils.to_dB(np.abs(fft_x), min_dB=-200)
	amp_y = utils.to_dB(np.abs(fft_y), min_dB=-200)
	amp = amp_y - amp_x

	max_amp = np.amax(amp)
	min_amp = max(np.amin(amp), -80)

	phase = None
	if show_phase:
		phase_x = np.angle(fft_x)
		phase_y = np.angle(fft_y)

		phase = phase_y - phase_x
		phase = np.rad2deg(phase)
		#phase = (phase + 180) % 360 - 180
		#phase = (phase % 360) - 360

	fig = plt.figure()
	fig.suptitle('Comb filter')  # TODO: details

	num_rows = 3 if show_phase else 2

	plt.subplot(num_rows, 1, 1)
	plt.plot(y)
	plt.grid()
	plt.ylabel('Impulse response')

	plt.subplot(num_rows, 2, 3)
	plt.plot(f, amp)
	plt.grid()
	plt.title('Linear freq')
	plt.ylabel('Amplitude (dB)')
	plt.xlim([0, sample_rate / 2])
	plt.ylim([min_amp, max_amp])

	if show_phase:
		plt.subplot(num_rows, 2, 5)
		plt.plot(f, phase)
		plt.grid()
		plt.xlim([0, sample_rate / 2])
		plt.ylabel('Phase (degrees)')

	plt.subplot(num_rows, 2, 4)
	plt.semilogx(f, amp)
	plt.grid()
	plt.xlim([20, sample_rate / 2])
	plt.ylim([min_amp, max_amp])
	plt.title('Log freq')
	plt.ylabel('Amplitude (dB)')

	if show_phase:
		plt.subplot(num_rows, 2, 6)
		plt.semilogx(f, phase)
		plt.grid()
		plt.xlim([20, sample_rate / 2])
		plt.ylabel('Phase (degrees)')


def plot(args):
	args = parse_args(args)

	delay_samples_float = (args.delay_time / 1000.0) * args.sample_rate
	delay_samples = int(round(delay_samples_float))

	print('Delay: %g ms @ %g kHz = %g samples, fundamental %g Hz' % (
		args.delay_time,
		args.sample_rate,
		delay_samples_float,
		1000.0 / args.delay_time,
	))

	if delay_samples_float != delay_samples:
		print('Rounding to %i samples, fundamental %g Hz' % (delay_samples, args.sample_rate / delay_samples))

	dl = DelayLine(delay_samples)

	max_delay_time = delay_samples

	fbdl = None
	if args.feedback_delay_time:

		feedback_delay_samples_float = (args.feedback_delay_time / 1000.0) * args.sample_rate
		feedback_delay_samples = int(round(feedback_delay_samples_float))

		print('Feedback delay: %g ms = %i samples, fundamental %g Hz' % (
			args.feedback_delay_time,
			feedback_delay_samples,
			args.sample_rate / feedback_delay_samples,
		))

		fbdl = DelayLine(feedback_delay_samples)

		max_delay_time = max(delay_samples, feedback_delay_samples)

	ideal_buffer_size = determine_buffer_size(max_delay_time, args.feedback, wet_mix=args.wet_mix, eps_dB=args.eps_dB)

	# Pad to power of 2
	actual_buffer_size = utils.clip(ideal_buffer_size, (args.min_len, args.max_len))
	actual_buffer_size = int(2 ** np.ceil(np.log2(actual_buffer_size)))

	if ideal_buffer_size > actual_buffer_size:
		print('Ideal buffer size %i, greater than max (%i), using max' % (ideal_buffer_size, actual_buffer_size))
	else:
		print('Ideal buffer size %i, using %i' % (ideal_buffer_size, actual_buffer_size))

	x = np.zeros(actual_buffer_size)
	x[0] = 1.0

	print('Dry mix: %g, wet mix: %g, feedback: %g' % (args.dry_mix, args.wet_mix, args.feedback))

	print('Processing')
	y = process(x, delay_line=dl, fb_delay_line=fbdl, dry_mix=args.dry_mix, wet_mix=args.wet_mix, fb=args.feedback)

	print('Plotting')
	do_plot(x, y, sample_rate=args.sample_rate, show_phase=args.show_phase)
	plt.show()



def main(args):
	plot(args)


if __name__ == "__main__":
	import sys
	main(sys.argv)
