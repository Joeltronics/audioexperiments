#!/usr/bin/env python3

import math
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from analysis import linearity
from utils import utils
from unit_test import unit_test
from processor import ProcessorBase
from generation import signal_generation


def generate_impulse(n_samp, amplitude=1.0) -> np.ndarray:
	x = np.zeros(n_samp, dtype=np.float64)
	x[0] = amplitude
	return x


def generate_step(n_samp, amplitude=1.0) -> np.ndarray:
	return np.ones(n_samp) * amplitude


def generate_ramp(n_samp, slope=1.0) -> np.ndarray:
	y = (np.arange(n_samp) + 1).astype(np.float64) * slope
	assert utils.approx_equal(y[0], slope)
	assert utils.approx_equal(y[1], 2*slope)
	return y


def get_impulse_response(system, n_samp, amplitude=1.0, reset=True, negative=False) -> np.ndarray:

	# Assuming system is LTI & causal, and that system.reset() works as it should,
	# we can ignore negative half of impulse/step response, as zero-input will have zero-output

	x = generate_impulse(n_samp, amplitude)

	if negative:
		x = -x

	if reset:
		system.reset()

	return system.process_vector(x)


def get_step_response(system, n_samp, amplitude=1.0, reset=True, negative=False) -> np.ndarray:
	x = generate_step(n_samp, amplitude)

	if negative:
		x = -x

	if reset:
		system.reset()

	return system.process_vector(x)


def get_ramp_response(system, n_samp, slope=1.0, reset=True, negative=False) -> np.ndarray:
	x = generate_ramp(n_samp, slope)

	if negative:
		x = -x

	if reset:
		system.reset()

	return system.process_vector(x)
