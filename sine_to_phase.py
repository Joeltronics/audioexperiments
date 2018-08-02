#!/usr/bin/env python3

from processor import Processor
import utils

from math import pi

import numpy as np
import math


class SineToPhase(Processor):
	def __init__(self):
		self.prev_s = 0.
	
	def process_sample(self, s):
		base_angle = np.arcsin(s)  # Angle in range -pi/2, pi/2
		phase = base_angle / (2*pi)  # Phase in range -0.25, 0.25

		slope = s - self.prev_s
		self.prev_s = s

		if slope < 0:
			phase = 0.5 - phase

		return np.mod(phase, 1.0)

	def reset(self):
		self.prev_s = 0


def phase_diff(input, output, normalize=True, radians=False, degrees=False):
	if radians and degrees:
		raise ValueError('Cannot set both radians and degrees!')
	
	# FIXME I don't think this works properly...

	phase_input = SineToPhase().process_vector(utils.normalize(input))
	phase_output = SineToPhase().process_vector(utils.normalize(output))
	diff = np.mean(utils.wrap05(phase_output - phase_input))
	
	if radians:
		return diff * 2. * pi
	elif degrees:
		return diff * 360.
	else:
		return diff


def main():
	from matplotlib import pyplot as plt

	freq = 440
	sample_rate = 48000
	n_samp = 4096

	true_phase = utils.gen_phase(freq / sample_rate, n_samp)
	sine = utils.phase_to_sine(true_phase)

	t = np.linspace(0., n_samp/sample_rate, n_samp)

	s2p = SineToPhase()
	reconstructed_phase = s2p.process_vector(sine)

	plt.figure()

	plt.plot(t, true_phase, label='Phase')
	plt.plot(t, sine, label='Sine')
	plt.plot(t, reconstructed_phase, label='Reconstructed phase')
	plt.plot(t, reconstructed_phase - true_phase, label='Error')

	plt.grid()
	plt.legend()
	plt.xlabel('Time')
	plt.title('Sine to phase')

	plt.show()

if __name__ == "__main__":
	main()
