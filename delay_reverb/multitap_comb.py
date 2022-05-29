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

Ideas & derivations:

===== Original idea =====

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

===== New idea =====

Same as above, but both the feedback & mix can be any arbitrary mix of delays 1 & 2

              +-------------------------+
              |                         |
              |               [                  ]
              |               [        mix       ]
              |               [                  ]
              |                 A              A
              V                 |              |
        +--> (+) ---> [ dl ] ---+---> [ dl ] --+
        |                       |              |
        |                       |              +---> [     ]
        |                       |                    [     ]
x[n] ---+                       +------------------> [ mix ] -----> y[n]
        |                                            [     ]
        +------------------------------------------> [     ]

(fb control would be unnecessary due to fb mix)

Once again, can split:

              +----------------------------+
              |                            |
              |                  [                  ]
              |                  [        mix       ]
              |                  [                  ]
              |                    A              A
              |                    |              |
              V     +--> [ dl ] ---+              |
        +--> (+) ---+              |              |
        |           +------------- | --> [ dl ] --+
        |                          |              |
        |                          |              +---> [     ]
        |                          |                    [     ]
x[n] ---+                          +------------------> [ mix ] -----> y[n]
        |                                               [     ]
        +---------------------------------------------> [     ]

Essentially this is just 2 mixers, 1 at the input of the delay lines, and 1 at the output
"""


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)

	parser.add_argument(
		'-s', '--sample-rate', metavar='HZ', dest='sample_rate',
		type=int, default=48000,
		help='Sample Rate (Hz); default 48000',
	)

	grp = parser.add_argument_group('delay')

	grp.add_argument(
		'--d1', dest='delay_1_time', metavar='MS',
		type=float, default=1,
		help='Delay 1 (ms); default 1; will be rounded to integer number of samples',
	)
	grp.add_argument(
		'--d2', dest='delay_2_time', metavar='MS',
		type=float, default=1,
		help='Delay 2 (ms); default 1; will be rounded to integer number of samples',
	)

	grp = parser.add_argument_group('mix')

	grp.add_argument(
		'--dry', metavar='MIX', dest='dry_mix',
		type=float, default=0.5,
		help='Dry mix; can be negative; default 0.5',
	)
	grp.add_argument(
		'--mix1', metavar='MIX', dest='wet_1_mix',
		type=float, default=0.25,
		help='Delay 1 mix; can be negative; default 0.25',
	)
	grp.add_argument(
		'--mix2', metavar='MIX', dest='wet_2_mix',
		type=float, default=0.25,
		help='Delay 2 mix; can be negative; default 0.25',
	)

	grp = parser.add_argument_group('Feedback')

	grp.add_argument(
		'--fb1', metavar='FB', dest='feedback_1',
		type=float, default=0.0,
		help='Feedback 1 amount; default 0; valid range (-1 < FB < 1)',
	)
	grp.add_argument(
		'--fb2', metavar='FB', dest='feedback_2',
		type=float, default=0.0,
		help='Feedback 2 amount; default 0; valid range (-1 < FB < 1)',
	)

	grp = parser.add_argument_group('Analysis')

	grp.add_argument(
		'--eps', metavar='dB', dest='eps_dB',
		type=float, default=-120.0,
		help='Epsilon value - stop processing once level below this; default -120 dB'
	)
	grp.add_argument(
		'--minlen', metavar='SAMPLES', dest='min_len',
		type=int, default=1024,
		help='Minimum number of samples to process; default 1024',
	)
	grp.add_argument(
		'--maxlen', metavar='SAMPLES', dest='max_len',
		type=int, default=65536,
		help='Maximum number of samples to process; default 65536',
	)
	grp.add_argument(
		'--phase', dest='show_phase',
		action='store_true',
		help='Show phase plot',
	)

	return parser


def determine_buffer_size(delay_time: int, fb: float, max_len: int, wet_mix=1.0, eps_dB=-120) -> int:

	# TODO: Is the +1 actually necessary?

	if not fb:
		return delay_time + 1

	fb = abs(fb)

	if not (fb < 1.0):
		raise ValueError('Feedback must be in range (-1 < fb < 1)')

	eps = utils.from_dB(eps_dB)

	# TODO: solve this properly (not iteratively), it's just exponential decay
	num_periods = 1
	level = abs(wet_mix)
	while level > eps:
		level *= fb
		num_periods += 1

		if (num_periods * delay_time) + 1 > max_len:
			return max_len

	return num_periods * delay_time + 1


def process(x: np.ndarray, dry_mix, wet_1_mix, wet_2_mix, fb_1, fb_2, delay_line_1, delay_line_2) -> np.ndarray:
	if not (wet_1_mix or wet_2_mix):
		return x

	y = np.zeros_like(x)

	for n in range(len(y)):
		"""
		              +----------------------------+
		              |                            |
		              |                  [                  ]
		              |                  [        mix       ]
		              |                  [                  ]
		              |                    A              A
		              |                    |              |
		              V     +--> [ dl ] ---+              |
		        +--> (+) ---+              |              |
		        |           +------------- | --> [ dl ] --+
		        |                          |              |
		        |                          |              +---> [     ]
		        |                          |                    [     ]
		x[n] ---+                          +------------------> [ mix ] -----> y[n]
		        |                                               [     ]
		        +---------------------------------------------> [     ]
		"""

		xn = x[n]

		dl1 = delay_line_1.peek_front()
		dl2 = delay_line_2.peek_front()

		delay_in = xn + (dl1 * fb_1) + (dl2 * fb_2)

		delay_line_1.push_back(delay_in)
		delay_line_2.push_back(delay_in)

		yn = (xn * dry_mix) + (dl1 * wet_1_mix) + (dl2 * wet_2_mix)

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

	if not (-1.0 < args.feedback_1 < 1.0 and -1.0 < args.feedback_2 < 1.0):
		raise ValueError('Feedbacks must each be in range (-1 < fb < 1)')

	if not (-1.0 < (args.feedback_1 + args.feedback_2) < 1.0):
		raise ValueError('Sum of feedback 1 + 2 must be in range (-1 < fb < 1)')

	delay_1_samples_float = (args.delay_1_time / 1000.0) * args.sample_rate
	delay_1_samples = int(round(delay_1_samples_float))

	delay_2_samples_float = (args.delay_2_time / 1000.0) * args.sample_rate
	delay_2_samples = int(round(delay_2_samples_float))

	print('Delay 1: %g ms @ %g kHz = %g samples, fundamental %g Hz' % (
		args.delay_1_time,
		args.sample_rate,
		delay_1_samples_float,
		1000.0 / args.delay_1_time,
	))

	if delay_1_samples_float != delay_1_samples:
		print('Rounding to %i samples, fundamental %g Hz' % (delay_1_samples, args.sample_rate / delay_1_samples))

	print('Delay 2: %g ms @ %g kHz = %g samples, fundamental %g Hz' % (
		args.delay_2_time,
		args.sample_rate,
		delay_2_samples_float,
		1000.0 / args.delay_2_time,
	))

	if delay_2_samples_float != delay_2_samples:
		print('Rounding to %i samples, fundamental %g Hz' % (delay_2_samples, args.sample_rate / delay_2_samples))

	dl1 = DelayLine(delay_1_samples)
	dl2 = DelayLine(delay_2_samples)

	max_delay_time = max(delay_1_samples, delay_2_samples)

	# FIXME: this won't work as expected with dual feedback
	ideal_buffer_size = determine_buffer_size(max_delay_time, (args.feedback_1 + args.feedback_2), max_len=args.max_len, wet_mix=(args.wet_1_mix + args.wet_2_mix), eps_dB=args.eps_dB)

	# Pad to power of 2
	actual_buffer_size = utils.clip(ideal_buffer_size, (args.min_len, args.max_len))
	actual_buffer_size = int(2 ** np.ceil(np.log2(actual_buffer_size)))

	if ideal_buffer_size > actual_buffer_size:
		print('Ideal buffer size %i, greater than max (%i), using max' % (ideal_buffer_size, actual_buffer_size))
	else:
		print('Ideal buffer size %i, using %i' % (ideal_buffer_size, actual_buffer_size))

	x = np.zeros(actual_buffer_size)
	x[0] = 1.0

	print('Processing')
	y = process(x, delay_line_1=dl1, delay_line_2=dl2, dry_mix=args.dry_mix, wet_1_mix=args.wet_1_mix, wet_2_mix=args.wet_2_mix, fb_1=args.feedback_1, fb_2=args.feedback_2)

	print('Plotting')
	do_plot(x, y, sample_rate=args.sample_rate, show_phase=args.show_phase)
	plt.show()


def main(args):
	plot(args)
