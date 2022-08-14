#!/usr/bin/env python3


from processor import ProcessorBase
from typing import Union, Tuple

from delay_reverb import delay_line  # Somehow this works despite the circular dependency?!


class FractionalDelayAllpass(ProcessorBase):
	"""Delay < 1 sample using allpass filter"""

	# https://ccrma.stanford.edu/~jos/pasp/First_Order_Allpass_Interpolation.html
	def __init__(self, delay_samples: float):
		self.z1 = 0.
		self.eta = None
		self.set_delay(delay_samples)

	def set_delay(self, delay_samples: float):
		self.eta = (1.0 - delay_samples) / (1.0 + delay_samples)

	def process_sample(self, x: float) -> float:
		y, zin = self.process_sample_no_state_update(x)
		self.z1 = zin
		return y

	def process_sample_no_state_update(self, x: float) -> Tuple[float, float]:
		zin = x - self.eta*self.z1
		y = self.eta*zin + self.z1
		return y, zin

	def reset(self):
		self.z1 = 0.

	def get_state(self) -> float:
		return self.z1

	def set_state(self, state: float):
		self.z1 = state


class AllpassFilter(ProcessorBase):
	"""
	First-order allpass filter

	Often represented as one of two equivalent block diagrams:

	#               ------ k ------
	#              |               |
	#              |               V
	# >--- ( + ) -----> Z -----> ( + ) ----->
	#        A              |
	#        |              |
	#         ----- -k -----

	# >--- ( + ) -------
	#         A     /   |
	#          \   k    |
	#           \ /     V
	#            X      Z
	#           / \     |
	#          /  -k    |
	#         V     \   |
	# <--- ( + ) -------

	"""

	def __init__(self, k, delay_samples: Union[float, int], allpass_interpolate=False):
		"""
		:param k: feedforward coefficient, and negative of feedback coefficient; may be positive or negative
		:param delay_samples: delay time, in samples
		"""
		self.k = k

		if allpass_interpolate:
			self.dl = delay_line.FractionalAllpassDelayLine(delay_samples)
		else:
			self.dl = delay_line.DelayLine(delay_samples)

	def __getitem__(self, index: Union[int, float]):
		return self.dl[index]

	def process_sample(self, x: float, delay_samples: Union[None, float, int]=None):

		#               -----  k ------
		#              |               |
		#   x          |       zout    V      y
		# >--- ( + ) -----> Z -----> ( + ) ----->
		#        A    zin       |
		#        |              |
		#         ----- -k -----

		# TODO: use delay_samples

		zout = self.dl.peek_front()

		zin = x - self.k*zout
		y = self.k*zin + zout

		self.dl.push_back(zin)
		return y

	def reset(self):
		self.dl.reset()

	def get_state(self):
		return self.dl.get_state()

	def set_state(self, state):
		self.dl.set_state(state)


def _plot_allpass(args, delay=4, n_samp=32):
	from matplotlib import pyplot as plt
	import numpy as np

	d = delay

	x = np.zeros(n_samp)
	x[0] = 1.

	ks = [-0.99, -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 0.99]

	plt.figure()

	for n, k in enumerate(ks):
		lattice = AllpassFilter(k=k, delay_samples=d)
		y = lattice.process_vector(x)
		plt.subplot(len(ks), 1, n+1)

		if '--stem' in args:
			plt.stem(y, label='k=%g' % k)
		else:
			plt.plot(x, '.-', label='Input')
			plt.plot(y, '.-', label='Output, k=%g' % k)

		if n == 0:
			plt.title('Allpass filter (delay time %i)' % d)

		plt.grid()
		plt.legend()
		plt.ylim([-1.1, 1.1])
		plt.yticks([-1, -0.5, 0, 0.5, 1])


def _plot_interp():
	from matplotlib import pyplot as plt
	import numpy as np
	import scipy.signal

	plt.figure()
	plt.title('Allpass interpolator')

	x = np.zeros(32)
	x[0] = 1.

	for delay in [0., 0.25, 0.5, 0.75, 1.]:
		interp = FractionalDelayAllpass(delay)
		y = interp.process_vector(x)
		plt.plot(y, '.-', label='delay=%g' % delay)

	plt.grid()
	plt.legend()


	plt.figure()
	plt.title('Allpass interpolator')

	n_samp = 128

	x = np.zeros(n_samp)
	x[n_samp // 2] = 1.

	delays = [0., 0.25, 0.5, 0.75, 1., 1.25, 1.5, 2.0]

	upsample_ratio = 16

	t = np.arange(-(n_samp // 2), n_samp // 2)

	for n, delay in enumerate(delays):
		plt.subplot(len(delays), 1, n+1)

		interp = FractionalDelayAllpass(delay)
		y = interp.process_vector(x)
		y_up, t_up = scipy.signal.resample(y, n_samp*upsample_ratio, t=t)
		plt.plot(t, x, '.', label='input', zorder=1)
		plt.plot(t, y, '.', label='delay=%g' % delay, zorder=3)
		plt.plot(t_up, y_up, '-', label='Upsampled', zorder=2)
		plt.xlim([-4, 4])

		plt.grid()
		plt.legend()


def _plot_interp_allpass():
	from matplotlib import pyplot as plt
	import numpy as np

	plt.figure()

	n_samp = 512
	d = 19.75
	d_rounded = int(round(d))
	x = np.zeros(n_samp)
	x[0] = 1.

	plt.title('Fractional delay allpass filter (%g samples)' % d)

	lattice_basic = AllpassFilter(k=0.5, delay_samples=d_rounded)
	lattice_lerp = AllpassFilter(k=0.5, delay_samples=d, allpass_interpolate=False)
	lattice_ap = AllpassFilter(k=0.5, delay_samples=d, allpass_interpolate=True)

	y_basic = lattice_basic.process_vector(x)
	y_lerp = lattice_lerp.process_vector(x)
	y_ap = lattice_ap.process_vector(x)

	plt.plot(y_basic, '.-', label='Non-fractional delay (%i)' % d_rounded)
	plt.plot(y_lerp, '.-', label='Lerp')
	plt.plot(y_ap, '.-', label='Allpass interp')

	plt.grid()
	plt.legend()


def plot(args):
	from matplotlib import pyplot as plt

	_plot_allpass(args)
	_plot_interp()
	_plot_interp_allpass()

	plt.show()


def main(args):
	plot(args)
