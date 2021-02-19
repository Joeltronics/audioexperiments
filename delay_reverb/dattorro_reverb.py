#!/usr/bin/env python3


from processor import ProcessorBase, CascadedProcessors
from delay_reverb import delay_line
from filters import one_pole, allpass
import numpy as np
from typing import Optional


class GriesingerDattorroFigure8Reverb(ProcessorBase):
	"""
	Plate reverb based on "figure 8" algorithm "in the style of Griesinger" as described by Jon Dattorro in
	"Effect Design Part 1: Reverberator and Other Filters",

	Original sample rate: 29761 Hz
	Could run at any sample rate, but modifications to various coeffs and delay times would be needed in order to get
	the same parameters

	Note: delay line modulation is not yet implemented!
	"""

	def __init__(
			self,
			predelay_samples: int,
			decay=0.5,
			damping=0.0005,
			wet_gain=1.0,
			dry_gain=0.0,
			modulation_freq: Optional[float]=None):
		"""
		:param predelay_samples: in addition to the algorithm's natural predelay
		:param decay:
		:param damping:
		:param wet_gain:
		:param dry_gain:
		:param modulation_freq: frequency, in cycles per sample
		"""

		self.predelay_samples = predelay_samples

		self.wet_gain = wet_gain
		self.dry_gain = dry_gain

		self.decay = decay
		decay_diffusion_1 = 0.7
		decay_diffusion_2 = 0.5

		input_diffusion_1 = -0.75
		input_diffusion_2 = 0.625

		bandwidth = 0.9995
		self.damping = damping

		if modulation_freq is not None and not (0. < modulation_freq < 0.5):
			raise ValueError('Invalid modulation freq, must be between 0 and 0.5')

		self.mod = bool(modulation_freq)
		self.mod_phase = 0.
		self.mod_freq = modulation_freq
		self.mod_excursion = 16 if self.mod else 0

		self.input = CascadedProcessors(
			[delay_line.DelayLine(self.predelay_samples)] if self.predelay_samples else [] + [
			one_pole.BasicOnePole(wc=None, b0=bandwidth),
			allpass.AllpassFilter(input_diffusion_1, 142),
			allpass.AllpassFilter(input_diffusion_1, 107),
			allpass.AllpassFilter(input_diffusion_2, 379),
			allpass.AllpassFilter(input_diffusion_2, 277),
		])

		self.dif1 = [
			allpass.AllpassFilter(-decay_diffusion_1, 672 + self.mod_excursion),
			allpass.AllpassFilter(-decay_diffusion_1, 908 + self.mod_excursion),
		]

		self.del1 = [
			delay_line.DelayLine(4453),
			delay_line.DelayLine(4217),
		]

		self.damp = [
			one_pole.BasicOnePole(wc=None, b0=(1.0 - self.damping), gain=self.decay),
			one_pole.BasicOnePole(wc=None, b0=(1.0 - self.damping), gain=self.decay),
		]

		self.dif2 = [
			allpass.AllpassFilter(decay_diffusion_2, 1800),
			allpass.AllpassFilter(decay_diffusion_2, 2656),
		]

		self.del2 = [
			delay_line.DelayLine(3720),
			delay_line.DelayLine(3163),
		]

		self.stateful_components = \
			self.input.processors + self.dif1 + self.del1 + self.damp + self.dif2 + self.del2

	def process_sample(self, x: float):

		after_input = self.input(x)
		paths = [0., 0.]

		taps = [672, 908]

		if self.mod:
			mod_phase = self.mod_phase * 2.0 * np.pi
			mods = 0.5 * self.mod_excursion * (1 + np.array([np.sin(mod_phase), np.cos(mod_phase)]))
			taps[0] += mods[0]
			taps[1] += mods[1]
			self.mod_phase += self.mod_freq
			self.mod_phase %= 1.0

		for n in (0, 1):
			paths[n] = after_input + self.decay * self.del2[1 - n].peek_front()
			# TODO: use self.mod_phase (right now the allpass filters don't support it)
			paths[n] = self.dif1[n].process_sample(paths[n], delay_samples=taps[n])

		del1_in = tuple(paths)

		for n in (0, 1):
			paths[n] = self.del1[n].peek_front()
			paths[n] = self.damp[n].process_sample(paths[n])
			paths[n] = self.dif2[n].process_sample(paths[n])

		del2_in = tuple(paths)

		yl  = 0.6 * self.del1[1][266]   # node48_54[266]
		yl += 0.6 * self.del1[1][2974]  # node48_54[2974]
		yl -= 0.6 * self.dif2[1][1913]  # node55_59[1913]
		yl += 0.6 * self.del2[1][1996]  # node59_63[1996]
		yl -= 0.6 * self.del1[0][1990]  # node24_30[1990]
		yl -= 0.6 * self.dif2[0][187]   # node31_33[187]
		yl -= 0.6 * self.del2[0][1066]  # node33_39[1066]

		yr  = 0.6 * self.del1[0][353]   # node24_30[353]
		yr += 0.6 * self.del1[0][3267]  # node24_30[3267]
		yr -= 0.6 * self.dif2[0][1228]  # node31_33[1228]
		yr += 0.6 * self.del2[0][2673]  # node33_39[2673]
		yr -= 0.6 * self.del1[1][2111]  # node48_54[2111]
		yr -= 0.6 * self.dif2[1][355]   # node55_59[355]
		yr -= 0.6 * self.del2[1][121]   # node59_63[121]

		for n in [0, 1]:
			self.del1[n].push_back(del1_in[n])
			self.del2[n].push_back(del2_in[n])

		# TODO: Find a way to support stereo in ProcessorBase API
		return self.dry_gain*x + self.wet_gain*0.5*(yl + yr)

	def reset(self):
		for c in self.stateful_components:
			c.reset()

	def get_state(self):
		s = []
		for c in self.stateful_components:
			s.append(c.get_state())
		return s

	def set_state(self, state):
		for c, s in zip(self.stateful_components, state):
			c.set_state(s)


def plot(args):
	from matplotlib import pyplot as plt
	from analysis.reverb import analyze_reverb_ir

	predelay_ms = 0.
	time_s = 3.

	sample_rate = 29761
	n_samp = round(time_s * sample_rate)

	predelay_samples = round(1.e-3 * predelay_ms * sample_rate)

	verb_nomod = GriesingerDattorroFigure8Reverb(predelay_samples=predelay_samples, modulation_freq=None)
	#verb_mod = GriesingerDattorroFigure8Reverb(predelay_samples=predelay_samples, modulation_freq=1./sample_rate)

	x = np.zeros(n_samp)
	x[0] = 1.0

	print('Processing')
	y_nomod = verb_nomod.process_vector(x)
	#y_mod = verb_mod.process_vector(x)

	print('Plotting')

	analyze_reverb_ir(y_nomod, sample_rate, title='Griesinger-Dattorro "Figure 8" Reverb (no modulation)')
	#analyze_reverb_ir(y_mod, sample_rate, title='Griesinger-Dattorro "Figure 8" Reverb (with modulation)')
	plt.show()


def main(args):
	plot(args)
