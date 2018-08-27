#!/usr/bin/env python3

import numpy as np
from typing import Union


class ProcessorBase:
	def process_sample(self, sample: float) -> float:
		raise NotImplementedError('process_sample() to be implemented by the child class!')

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		y = np.zeros_like(vec)
		for n, x in enumerate(vec):
			y[n] = self.process_sample(x)
		return y

	def process(self, input: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
		"""Calls process_sample or process_vector depending on type of input"""
		if np.isscalar(input):
			return self.process_sample(input)
		else:
			return self.process_vector(input)

	def reset(self) -> None:
		"""Reset Processor state (but not parameters)"""
		pass

	def __call__(self, *args, **kwargs):
		return self.process(*args, **kwargs)


class CascadedProcessors(ProcessorBase):
	def __init__(self, processors):
		self.processors = processors
	
	def reset(self):
		for p in self.processors:
			p.reset()

	def process_sample(self, x):
		y = x
		for p in self.processors:
			y = p.process_sample(y)
		return y
	
	def process_vector(self, vec: np.array) -> np.array:
		y = vec
		for p in self.processors:
			y = p.process_vector(y)
		return y
