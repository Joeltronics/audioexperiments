#!/usr/bin/env python3


import numpy as np
from typing import Union
from utils import utils
from processor import StatelessProcessorBase


def clip(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""Hard clipping at [-1, 1]"""
	return utils.clip(vals, (-1., 1.))


class Clipper(StatelessProcessorBase):
	def __init__(self, gain=1.0):
		self.gain = gain

	def process_sample(self, x):
		return clip(x * self.gain)

	def process_vector(self, x):
		return clip(x * self.gain)


def tanh(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return np.tanh(vals)


class TanhProcessor(StatelessProcessorBase):
	def __init__(self, gain=1.0):
		self.gain = gain

	def process_sample(self, x):
		return tanh(x * self.gain)

	def process_vector(self, x):
		return tanh(x * self.gain)


def sigmoid(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""x / (1 + |x|) sigmoid; even softer clipping than tanh"""
	return vals / (1.0 + np.abs(vals))


class Sigmoid(StatelessProcessorBase):
	def __init__(self, gain=1.0):
		self.gain = gain

	def process_sample(self, x):
		return sigmoid(x * self.gain)

	def process_vector(self, x):
		return sigmoid(x * self.gain)


def plot(args):
	from matplotlib import pyplot as plt

	x = np.linspace(-10., 10., 1024)

	plt.figure()

	plt.plot(x, clip(x), label='clip')
	plt.plot(x, tanh(x), label='tanh')
	plt.plot(x, sigmoid(x), label='Sigmoid')
	plt.grid()
	plt.legend()

	plt.show()


def main(args):
	plot(args)
