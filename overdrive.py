#!/usr/bin/env python3


import numpy as np
from typing import Union
import utils


def clip(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""Hard clipping at [-1, 1]"""
	return utils.clip(vals, (-1., 1.))


def tanh(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return np.tanh(vals)


def sigmoid(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""1 / (1 + |x|) sigmoid; even softer clipping than tanh"""
	return 1.0 / (1.0 + np.abs(vals))
