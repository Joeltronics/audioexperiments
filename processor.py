#!/usr/bin/env python3

import numpy as np


class Processor:
	def process_vector(self, vec: np.array) -> np.array:
		y = np.zeros_like(vec)
		for n, x in enumerate(vec):
			y[n] = self.process_sample(x)
		return y
	
	def process_sample(self, sample: float) -> float:
		raise NotImplementedError('process_sample() to be implemented by the child class!')

	def reset(self):
		raise NotImplementedError('reset() to be implemented by the child class!')


class CascadedProcessors(Processor):
	def __init__(self, processors):
		self.processors = processors
	
	def reset(self):
		[p.reset() for p in self.processors]

	def process_sample(self, x):
		y = x
		for p in self.processors:
			y = p.process_sample(y)
		return y
	
	def process_vector(self, vec: np.array) -> np.array:
		y = np.zeros_like(vec)
		for n, x in enumerate(vec):
			for m, p in enumerate(self.processors):
				y[n] = p.process_sample(x if m == 0 else y[n])
		return y
