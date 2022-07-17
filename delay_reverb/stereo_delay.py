#!/usr/bin/env python3

from typing import Any, List, Tuple, Union

from numpy import exp as exp, log as ln, sqrt
import numpy as np
from matplotlib import pyplot as plt

from processor import StereoProcessorBase
from delay_reverb.delay_line import DelayLine


PHI = (1.0 + sqrt(5.0)) / 2.0
SILVER_RATIO = 1.0 + sqrt(2.0)


MODE_NORMAL = 'independent'
MODE_CROSSFEED = 'crossfeed'
MODE_PINGPONG = 'pingpong'


"""
Exponential decay formula: y = y0 * exp(-k*t)

y0 = 1 here, always

For t=1 side:

	fb = 1 * exp(-k*1)
	ln(fb) = -k

Uncompensated, for t=delay_ratio side:

	fb = 1 * exp(-k * ratio)
	ln(fb) = -k * ratio
	k = -ln(fb) / ratio

Compensating t=delay_ratio side for same decay rate:

	k = -ln(fb)

	fb_ratio_side = 1 * exp(-k * ratio)

	fb_ratio_side = exp(ln(fb) * ratio)

And, to calculate RT60:

	-60 dB = exp(-k * t60)
	0.001 = exp(-k * t60)
	-k * t60 = ln(0.001)
	t60 = -ln(0.001) / k
	t60 = ln(0.001) / ln(fb)
"""


class StereoDelay(StereoProcessorBase):
	def __init__(
			self,
			delay_times: Tuple[int, int],
			fb: Union[float, Tuple[float, float]],
			mode='independent',
			compensate_fb=True,
			):
		
		if not (delay_times[0] > 0 and delay_times[1] > 0):
			raise ValueError('Delay must be > 0')

		if mode not in [MODE_NORMAL, MODE_CROSSFEED, MODE_PINGPONG]:
			raise ValueError('Invalid mode: %s' % mode)
		self.mode = mode

		self.crossfeed = (mode == MODE_CROSSFEED)
		self.pingpong = (mode == MODE_PINGPONG)

		self.delay_line_1 = DelayLine(delay_samples=delay_times[0])
		self.delay_line_2 = DelayLine(delay_samples=delay_times[1])
		
		if isinstance(fb, tuple):
			self.fb_1 = fb[0]
			self.fb_2 = fb[1]

		elif isinstance(fb, float):
			# TODO: does pingpong/crossfeed affect feedback normalization?
			self.fb_1 = fb

			if not compensate_fb:
				self.fb_2 = self.fb_1
			else:
				delay_ratio = delay_times[1] / delay_times[0]
				self.fb_2 = exp(ln(self.fb_1) * delay_ratio)
		else:
			raise ValueError('Invalid fb type')

	def reset(self) -> None:
		self.delay_line_1.reset()
		self.delay_line_2.reset()

	def get_state(self) -> Tuple[Any, Any]:
		return self.delay_line_1.get_state(), self.delay_line_2.reset()

	def set_state(self, state: Tuple[Any, Any]) -> None:
		state1, state2 = state
		self.delay_line_1.set_state(state1)
		self.delay_line_2.set_state(state2)

	def process_sample(self, x: Tuple[float, float]) -> Tuple[float, float]:

		x1, x2 = x

		dry_1 = 1.0
		dry_2 = 1.0
		wet_1 = self.fb_1
		wet_2 = self.fb_2

		y1 = self.delay_line_1.peek_front()
		y2 = self.delay_line_2.peek_front()

		if self.crossfeed:
			assert not self.pingpong
			xf1 = xf2 = 0.5 * (y1 * self.fb_1 + y2 * self.fb_2)

		elif self.pingpong:
			xf1 = y2 * self.fb_1
			xf2 = y1 * self.fb_2

		else:
			xf1 = y1 * self.fb_1
			xf2 = y2 * self.fb_2

		# TODO: do we want special case input for crossfeed too?

		if self.pingpong:
			self.delay_line_1.push_back(x2 + xf1)
			self.delay_line_2.push_back(x1 + xf2)

		else:
			self.delay_line_1.push_back(x1 + xf1)
			self.delay_line_2.push_back(x2 + xf2)

		return dry_1*x1 + wet_1*y1, dry_2*x2 + wet_2*y2


def delay_processor_to_delay_times(delay: StereoDelay, num_samples) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:

	x_impulse = np.zeros(num_samples)
	x_zero = np.zeros(num_samples)

	x_impulse[0] = 1.0

	delays_out = [[], []]

	for channel in [0, 1]:
		delay.reset()

		inputs = (x_zero, x_impulse) if channel else (x_impulse, x_zero)

		yv0, yv1 = delay.process_vector(inputs)

		delays = delays_out[channel]

		for n, (y0, y1) in enumerate(zip(yv0, yv1)):

			has_y0 = bool(abs(y0))
			has_y1 = bool(abs(y1))

			if has_y0 and has_y1:
				delays.append((n, y0 if channel == 0 else -y0))
				delays.append((n, y1 if channel == 1 else -y1))
			elif has_y0:
				delays.append((n, y0 if channel == 0 else -y0))
			elif has_y1:
				delays.append((n, y1 if channel == 1 else -y1))

	return delays_out[0], delays_out[1]


def plot_stereo_delay(delay_ratio, fb=0.75, mode='independent', t_max=16, compensate_fb=True):

	if mode not in [MODE_NORMAL, MODE_CROSSFEED, MODE_PINGPONG]:
		raise ValueError('Invalid mode: %s' % mode)

	pingpong = False
	crossfeed = False
	if mode == MODE_CROSSFEED:
		crossfeed = True
	elif mode == MODE_PINGPONG:
		pingpong = True

	sample_rate = 1000

	delay_times = (
		int(round(delay_ratio * sample_rate)),
		int(round(sample_rate))
	)

	delay = StereoDelay(delay_times=delay_times, fb=fb, mode=mode, compensate_fb=compensate_fb)

	delays_1, delays_2 = delay_processor_to_delay_times(delay, num_samples=sample_rate*t_max)

	t60 = ln(0.001) / ln(fb)

	if compensate_fb:
		fb_ratio_side = exp(ln(fb) * delay_ratio)
	else:
		fb_ratio_side = fb

	if crossfeed:
		title = 'Stereo XF delay'
	elif pingpong:
		title = 'Ping-pong delay'
	else:
		title = 'Stereo delays'

	title += f', time ratio {delay_ratio:.3}, '

	if compensate_fb:
		title += f' fb={fb}/{fb_ratio_side:.3}'
	else:
		title += f' {fb=}'

	title += f', t60={t60:.1f}'

	fig, (ax1, ax2) = plt.subplots(2, 1)
	fig.suptitle(title)

	t = np.linspace(0, t_max, 1024)

	k1 = -ln(delay.fb_1) / (delay_times[0] / sample_rate)
	y = exp(-k1 * t)
	line, = ax1.plot(t, y)
	ax1.plot(t, -y, color=line.get_color())

	k2 = -ln(delay.fb_2) / (delay_times[1] / sample_rate)
	y = exp(-k2 * t)
	line, = ax2.plot(t, y)
	ax2.plot(t, -y, color=line.get_color())

	#print(f'ratio={delay_ratio:.3}, {compensate_fb=}, fb1={fb_ratio_side:.3}, {k1=:.3}, fb2={fb:.3}, {k2=:.3}')

	delays_1.sort(key=lambda d: d[0])
	delays_2.sort(key=lambda d: d[0])

	d1t = np.array([d[0] for d in delays_1]) / sample_rate
	d1y = np.array([d[1] for d in delays_1])

	d2t = np.array([d[0] for d in delays_2]) / sample_rate
	d2y = np.array([d[1] for d in delays_2])

	ax1.stem(d1t, d1y)
	ax2.stem(d2t, -d2y)

	for ax in [ax1, ax2]:
		ax.grid()
		ax.set_ylim([-1, 1])

	ax1.set_ylabel('From L')
	ax2.set_ylabel('From R')


def main(args):

	plot_stereo_delay(delay_ratio=1/PHI, fb=0.25)
	plot_stereo_delay(delay_ratio=1/PHI, fb=0.5)
	plot_stereo_delay(delay_ratio=1/PHI, fb=0.75, compensate_fb=True)
	plot_stereo_delay(delay_ratio=1/PHI, fb=0.75, compensate_fb=False)
	plot_stereo_delay(delay_ratio=1/PHI, fb=0.95)
	#plot_stereo_delay(delay_ratio=1/PHI, crossfeed=True)

	plot_stereo_delay(delay_ratio=1/PHI, fb=0.75, mode=MODE_CROSSFEED)
	plot_stereo_delay(delay_ratio=1/PHI, fb=0.75, mode=MODE_PINGPONG)

	plot_stereo_delay(delay_ratio=PHI)
	plot_stereo_delay(delay_ratio=SILVER_RATIO)

	plt.show()
