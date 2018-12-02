#!/usr/bin/env python3


from processor import ProcessorBase
from delay_reverb import delay_line
from typing import Union


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

	def __init__(self, k, delay_samples: int):
		"""
		:param k: feedforward coefficient, and negative of feedback coefficient; may be positive or negative
		:param delay_samples: delay time, in samples
		"""
		self.k = k
		self.delay_samples = delay_samples
		self.dl = delay_line.DelayLine(self.delay_samples)

	def __getitem__(self, index: Union[int, float]):
		return self.dl[index]

	def process_sample(self, x: float):

		#               -----  k ------
		#              |               |
		#   x          |       zout    V      y
		# >--- ( + ) -----> Z -----> ( + ) ----->
		#        A    zin       |
		#        |              |
		#         ----- -k -----

		k = self.k
		zout = self.dl.peek_front()

		zin = x - k*zout
		y = k*zin + zout

		self.dl.push_back(zin)
		return y

	def reset(self):
		self.dl.reset()

	def get_state(self):
		return self.dl.get_state()

	def set_state(self, state):
		self.dl.set_state(state)


def plot(args):
	from matplotlib import pyplot as plt
	import numpy as np

	n_samp = 32
	delay_time = 4

	x = np.zeros(n_samp)
	x[0] = 1.

	ks = [-0.99, -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, 0.99]

	plt.figure()

	for n, k in enumerate(ks):
		lattice = AllpassFilter(k=k, delay_samples=delay_time)
		y = lattice.process_vector(x)
		plt.subplot(len(ks), 1, n+1)

		if '--stem' in args:
			plt.stem(y, label='k=%g' % k)
		else:
			plt.plot(x, '.-', label='Input')
			plt.plot(y, '.-', label='Output, k=%g' % k)

		if n == 0:
			plt.title('Allpass filter (delay time %i)' % delay_time)

		plt.grid()
		plt.legend()
		plt.ylim([-1.1, 1.1])
		plt.yticks([-1, -0.5, 0, 0.5, 1])

	plt.show()


def main(args):
	plot(args)
