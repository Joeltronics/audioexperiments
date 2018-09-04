#!/usr/bin/env python3

from processor import ProcessorBase
from utils import utils

from math import pi

import numpy as np
import scipy.stats


class SineToPhase(ProcessorBase):
	def __init__(self):
		self.prev_s = 0.

	def reset(self):
		self.prev_s = 0
	
	def process_sample(self, s):
		base_angle = np.arcsin(s)  # Angle in range -pi/2, pi/2
		phase = base_angle / (2*pi)  # Phase in range -0.25, 0.25

		slope = s - self.prev_s
		self.prev_s = s

		if slope < 0:
			phase = 0.5 - phase

		return np.mod(phase, 1.0)


def phase_diff(input, output, normalize=True, radians=False, degrees=False):
	"""Calculate phase difference between 2 sinusoidal signals

	:param input:
	:param output:
	:param normalize: if True, will normalize input signals to range [-1, 1]
	:param radians: if True, will return value in radians (incompatible with degrees=True)
	:param degrees: if True, will return value in degrees (incompatible with radians=True)
	:return: phase difference in range  [-0.5, 0.5], unless radians or degrees is given
	"""

	if radians and degrees:
		raise ValueError('Cannot set both radians and degrees!')

	sig_in = input
	sig_out = output

	if normalize:
		sig_in = utils.normalize(sig_in)
		sig_out = utils.normalize(sig_out)

	phase_input = SineToPhase().process_vector(sig_in)
	phase_output = SineToPhase().process_vector(sig_out)

	diff = scipy.stats.circmean(phase_output - phase_input, low=-0.5, high=0.5)
	
	if radians:
		return diff * 2. * pi
	elif degrees:
		return diff * 360.
	else:
		return diff


def plot(args):
	from matplotlib import pyplot as plt
	from generation import signal_generation

	freq = 440
	sample_rate = 48000
	freq_norm = freq / sample_rate

	n_samp = 1024

	true_phase = signal_generation.gen_phase(freq_norm, n_samp)
	sine = signal_generation.phase_to_sine(true_phase)

	# Test phase diff

	# TODO: more extensive unit test of phase_diff

	true_phase_diff = -0.1
	sine_phase2 = signal_generation.gen_sine(freq_norm, n_samp=n_samp, start_phase=true_phase_diff)
	calc_phase_diff = phase_diff(sine, sine_phase2)
	print('Phase difference: true %f, calculated %f' % (true_phase_diff, calc_phase_diff))

	t = signal_generation.sample_time_index(n_samp, sample_rate)

	s2p = SineToPhase()
	reconstructed_phase = s2p.process_vector(sine)

	plt.figure()
	plt.subplot(211)

	plt.plot(t, true_phase, label='Phase')
	plt.plot(t, sine, label='Sine')
	plt.plot(t, reconstructed_phase, label='Reconstructed phase')

	plt.grid()
	plt.legend()
	plt.title('Sine to phase')

	plt.subplot(212)
	plt.plot(t, reconstructed_phase - true_phase, 'r')
	plt.grid()
	plt.ylabel('Error')
	plt.xlabel('Time (seconds)')

	plt.show()


def main(args):
	plot(args)
