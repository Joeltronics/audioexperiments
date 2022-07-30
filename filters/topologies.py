#!/usr/bin/env python3

import argparse
from math import sqrt
import re
from typing import Callable, Iterable, List, Optional, Tuple, Union

from cycler import cycler
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from matplotlib import pyplot as plt
from matplotlib import cm
import scipy.optimize
import schemdraw
from schemdraw import dsp
import schemdraw.elements as e


DEFAULT_COLORMAP = 'Spectral'
REVERSE_DEFAULT_COLORMAP = True


TWO_PI = 2.0 * np.pi
S_PLUS_1 = Polynomial([1, 1])


"""
TODOs:
* Pole mixing, and otherwise outputs besides just last stage
* Filters based off highpass blocks, or mix of highpass & lowpass
* Sallen Key Filters
* State Variable Filters
* Filters where different stages have different cutoff frequencies (e.g. Oberheim Matrix/Xpander)

Also lots of TODOs in the code below
"""



"""
Generic cascade filter:

Each pole response = 1/(s+1)

y1 = y0/(s+1)
y2 = y1/(s+1) = y0/(s+1)^2
y3 = y2/(s+1) = y0/(s+1)^2
y4 = y3/(s+1) = y0/(s+1)^4

To put everything in terms of y4 (will be useful later):

y3 = y4*(s+1)
y2 = y4*(s+1)^2
y1 = y4*(s+1)^3
y0 = y4*(s+1)^4

Or in terms of y2:

y1 = y2*(s+1)
y0 = y2*(s+1)^2
"""

"""
Normal 4-pole filter feedback:

y = y4
y0 = x - f*y4

Putting it together:

y4 = y0/(s+1)^4
y0 = x - f*y4

y = y0/(s+1)^4
y = (x - f*y) / (s+1)^4
y * (s+1)^4 = x - f*y
x = y * (s+1)^4 + f*y
H(s) = y/x = 1 / ((s+1)^4 + f)

Or as ratio of polynomials:
numer = 1
denom = (s+1)^4 + f = x^4 + 4*x^3 + 6*x^2 + 4*x + (1 + f)
"""

"""
4-pole bandpass feedback:

y = y4
y0 = x + f * fb_sig

fb_sig = y2 - 2*y3 + y4
fb_sig = y4*(s+1)^2 - 2*y4*(s+1) + y4
fb_sig = y*(s+1)^2 - 2*y*(s+1) + y
fb_sig = y * ((s+1)^2 - 2*(s+1) + 1)

y0 = x + f * y * ((s+1)^2 - 2*(s+1) + 1)
y*(s+1)^4 = x + f * y * ((s+1)^2 - 2*(s+1) + 1)
x = y*(s+1)^4 - f * y * ((s+1)^2 - 2*(s+1) + 1)
x/y = (s+1)^4 - f * ((s+1)^2 - 2*(s+1) + 1)
H(s) = y/x = 1 / ((s+1)^4 - f * ((s+1)^2 - 2*(s+1) + 1))
"""

"""
2-pole filter with 2-pole bandpass feedback:

y = y2
y0 = x - f * fb_sig

fb_sig = y2 - y1
fb_sig = y - y*(s+1)

y0 = x - f * (y - y*(s+1))

y = y0/(s+1)^2
y * (s+1)^2 = y0
y * (s+1)^2 = x - f * (y - y*(s+1))
x = y * (s+1)^2 + f * (y - y*(s+1))
x/y = (s+1)^2 + f * (1 - (s+1))
H(s) = y/x = 1 / ((s+1)^2 + f * (1 - (s+1)))
"""

"""
The arbitrary formula (for a cascade with 1 output tap) is:

                1
H(s) =  -------------------
        (s+1)^n - fb * G(s)

where G(s) is the feedback response but with the powers inverted - i.e. (s+1)^n-1 becomes (s+n)^1

(Come up with formula for this)

e.g. for 4-pole:

1 / H(s) = (s+1)^n + fb * ( FB1*(s+1)^3 + FB2*(s+1)^2 + FB3*(s+1) + FB4 )

"""


def _make_custom_cycler(n: int, colormap=DEFAULT_COLORMAP, reverse=REVERSE_DEFAULT_COLORMAP):

	cmap = cm.get_cmap(colormap)

	colors = [
		cmap(idx / (n - 1))
		for idx in (reversed(range(n)) if reverse else range(n))
	]

	return cycler(color=colors)


def _get_plot_freqs(resonant_freq: Optional[float] = None) -> np.ndarray:
	if (resonant_freq is not None) and (0.35 < resonant_freq < 2.5):
		return np.concatenate((
			np.logspace(np.log10(0.1), np.log10(resonant_freq * 0.3), num=128, base=10, endpoint=False),
			np.logspace(np.log10(resonant_freq * 0.3), np.log10(resonant_freq * 0.9), num=256, base=10, endpoint=False),
			np.logspace(np.log10(resonant_freq * 0.9), np.log10(resonant_freq * 1.1), num=256, base=10, endpoint=False),
			np.logspace(np.log10(resonant_freq * 1.1), np.log10(resonant_freq * 3), num=256, base=10, endpoint=False),
			np.logspace(np.log10(resonant_freq * 3), np.log10(10), num=128, base=10, endpoint=True),
		))
	else:
		return np.logspace(np.log10(0.1), np.log10(10.0), num=1024, base=10, endpoint=True)


def _mag(vals):
	return 20.0 * np.log10(np.abs(vals))


def _phase(vals):
	return np.mod(np.degrees(np.angle(vals)), -360.0)


def _plot_root_locus(f_polynomials, ax, max_fb, markers=False):

	# First, figure out number of zeros and poles

	numer = f_polynomials[0](0.0)
	denom = f_polynomials[1](0.0)

	zeros = numer.roots()
	poles = denom.roots()

	num_zeros = len(zeros)
	num_poles = len(poles)

	if not num_poles:
		raise ValueError('No poles!')

	fb_vals = np.linspace(0.0, max_fb, num=101, endpoint=True)

	# Calculate loci

	zeros_arrs = [np.zeros(len(fb_vals), dtype=np.cdouble) for _ in range(num_zeros)]
	poles_arrs = [np.zeros(len(fb_vals), dtype=np.cdouble) for _ in range(num_poles)]

	for fb_idx, fb in enumerate(fb_vals):

		# We've already calculated the first ones, no point recalculating
		if fb_idx > 0:
			numer = f_polynomials[0](fb)
			denom = f_polynomials[1](fb)

			zeros = numer.roots()
			poles = denom.roots()

		largest_real_pole = np.amax(np.real(poles))

		if largest_real_pole > 1e-9:
			break

		if len(zeros) != num_zeros:
			raise ValueError('Number of zeros changed when feedback changed!')

		if len(poles) != num_poles:
			raise ValueError('Number of poles changed when feedback changed!')

		# HACK:
		# Poles/zeros come out sorted by real component, which means they can swap order in certain cases (e.g. 4p bandpass fb)
		# Sort them in a way where they should always keep the same order
		# Still doesn't work 100% (particularly between very first 2 fb vals), but works better than without it

		def pz_sort_key(val: complex):
			return 1e6 * np.imag(val) + np.real(val)

		poles = sorted(list(poles), key=pz_sort_key)
		zeros = sorted(list(zeros), key=pz_sort_key)

		for zero_idx, zero in enumerate(zeros):
			zeros_arrs[zero_idx][fb_idx] = zero

		for pole_idx, pole in enumerate(poles):
			poles_arrs[pole_idx][fb_idx] = pole

	for zero_arr in zeros_arrs:
		ax.plot(np.real(zero_arr), np.imag(zero_arr), '-', color='blue')
		if markers:
			ax.scatter(np.real(zero_arr[0]), np.imag(zero_arr[0]), facecolors='none', edgecolors='blue')
			ax.scatter(np.real(zero_arr[-1]), np.imag(zero_arr[-1]), facecolors='none', edgecolors='blue')

	for pole_arr in poles_arrs:
		ax.plot(np.real(pole_arr), np.imag(pole_arr), '-', color='grey')
		if markers:
			ax.plot(np.real(pole_arr[0]), np.imag(pole_arr[0]), 'x', color='grey')
			ax.plot(np.real(pole_arr[-1]), np.imag(pole_arr[-1]), 'x', color='grey')


def _calculate_group_delay(w: np.ndarray, response: np.ndarray) -> np.ndarray:
	# TODO: actually calculate this properly, not discretely
	phase = np.angle(response)
	phase = np.unwrap(phase)
	return -np.gradient(phase, w)


def _plot_filter(
		w: np.ndarray,
		title: str,
		f_numerator: Callable,
		f_denominator: Callable,
		unstable_fb: Optional[float] = None,
		fb_plot_values: Optional[Iterable[float]] = None,
		f_drawing: Optional[Callable] = None,
		f_open_loop_response: Optional[Callable] = None,
		plot_group_delay = False,
		):

	f_numer = f_numerator
	f_denom = f_denominator

	if fb_plot_values is None:
		if unstable_fb is None:
			raise ValueError('Must provide either fb_plot_values or unstable_fb')

		fb_plot_values = unstable_fb * np.array([
			0.0,
			0.25,
			0.5,
			0.75,
			0.9,
			0.95,
			0.99,
			0.999,
		])

	ax_drawing = ax_pole_zero = ax_fb_open_loop = None

	fig = plt.figure()

	if f_drawing is not None:
		#gs = plt.GridSpec(4, 3, figure=fig)
		#ax_drawing = fig.add_subplot(gs[0,:-1])
		gs = plt.GridSpec(4, 2, figure=fig)
		ax_drawing = fig.add_subplot(gs[0,-1])
		ax_pole_zero = fig.add_subplot(gs[1:,-1])
	else:
		#gs = plt.GridSpec(3, 3, figure=fig)
		gs = plt.GridSpec(3, 2, figure=fig)
		ax_pole_zero = fig.add_subplot(gs[:,-1])

	if plot_group_delay:
		ax_mag = fig.add_subplot(gs[0,:-1])
		ax_ph = fig.add_subplot(gs[1,:-1], sharex=ax_mag)
		ax_gd = fig.add_subplot(gs[2,:-1], sharex=ax_mag)
	else:
		ax_mag = fig.add_subplot(gs[0:2,:-1])
		ax_ph = fig.add_subplot(gs[2,:-1], sharex=ax_mag)
		ax_gd = None

	ax_fb_open_loop_mag = fig.add_subplot(gs[3,:-1], sharex=ax_mag)
	ax_fb_open_loop_phase = ax_fb_open_loop_mag.twinx()

	# Custom color cycler
	cycler = _make_custom_cycler(len(fb_plot_values))
	for ax in [ax_mag, ax_ph, ax_gd]:
		if ax is not None:
			ax.set_prop_cycle(cycler)

	fig.suptitle(title)

	s = 1j * w

	# Draw bode plot in background of ax_mag
	num_poles = f_denom(0).degree()
	ax_mag.plot(
		[0.1, 1., 10.],
		[0.0, 0.0, -20. * num_poles],
		color='darkgrey',
	)

	ax_pole_zero.axhline(0, color='darkgrey')
	ax_pole_zero.axvline(0, color='darkgrey')

	_plot_root_locus((f_numer, f_denom), ax_pole_zero, max_fb=unstable_fb)

	for idx, fb in enumerate(fb_plot_values):
		numer = f_numer(fb)
		denom = f_denom(fb)
		resp = numer(s) / denom(s)

		label = f'{fb=:g}'

		mag = _mag(resp)
		line, = ax_mag.semilogx(w, mag, label=label, zorder=len(fb_plot_values) - idx)
		color = line.get_color()

		phase = _phase(resp)
		ax_ph.semilogx(w, phase, color=color, label=label, zorder=len(fb_plot_values) - idx)

		if ax_gd is not None:
			gd = _calculate_group_delay(w, resp)
			ax_gd.semilogx(w, gd, color=color, label=label, zorder=len(fb_plot_values) - idx)

		if idx in [0, 1, len(fb_plot_values) - 1]:

			zeros = numer.roots()
			poles = denom.roots()

			# TODO: label multiple identical poles/zeros

			ax_pole_zero.plot(np.real(poles), np.imag(poles), 'x', color=color)
			if len(zeros) > 0:
				ax_pole_zero.plot(np.real(zeros), np.imag(zeros), '.', color=color)

	if f_open_loop_response is not None:

		unstable_fb_is_1 = np.isclose(unstable_fb, 1.0)

		cmap = cm.get_cmap(DEFAULT_COLORMAP)
		color_max_fb = cmap(0.0 if REVERSE_DEFAULT_COLORMAP else 1.0)
		color_phase = cmap(1.0 if REVERSE_DEFAULT_COLORMAP else 0.0)

		if unstable_fb_is_1:
			color_1 = color_max_fb
		elif unstable_fb > 1.0:
			cm_idx_1 = 1.0 / unstable_fb
			color_1 = cmap((1.0 - cm_idx_1) if REVERSE_DEFAULT_COLORMAP else cm_idx_1)
		else:
			color_1 = (1.0, 0.0, 0.0)

		open_loop_resp = f_open_loop_response(w)
		mag = _mag(open_loop_resp)
		phase = _phase(open_loop_resp)

		if not unstable_fb_is_1:
			ax_fb_open_loop_mag.semilogx(w, _mag(f_open_loop_response(w, fb=1)), label='fb=1', color=color_1)
		ax_fb_open_loop_mag.semilogx(w, mag, label=f'fb={unstable_fb:g}', color=color_max_fb)

		ax_fb_open_loop_phase.semilogx(w, phase, '--', color=color_phase, label='Phase')

	#ax_mag.legend()
	ax_mag.set_ylim([-72, 48])
	ax_mag.set_yticks([-72, -60, -48, -36, -24, -12, 0, 12, 24, 36, 48])
	ax_mag.set_ylabel('Magnitude (dB)')

	ax_fb_open_loop_mag.set_ylabel('FB open-loop gain (dB)')
	ax_fb_open_loop_mag.set_ylim([-24, 24])
	ax_fb_open_loop_mag.set_yticks([-24, -12, 0, 12, 24])
	ax_fb_open_loop_mag.legend()
	#ax_fb_open_loop_phase.legend()

	for ax in [ax_ph, ax_fb_open_loop_phase]:
		ax.set_ylabel('Phase (degrees)')
		ax.set_ylim([-360, 0])
		ax.set_yticks([-360, -270, -180, -90, 0])

	if ax_gd is not None:
		# Units are weird here, since "w" is relative to cutoff frequency
		ax_gd.set_ylabel('Group delay')
		ax_gd.set_ylim([0, 10])

	for ax in [ax_mag, ax_ph, ax_gd, ax_fb_open_loop_mag, ax_fb_open_loop_phase]:
		if ax is None:
			continue
		ax.grid()
		ax.set_xlabel(r'$\omega$')

	if ax_drawing is not None:
		d = schemdraw.Drawing(show=False)
		f_drawing(d)
		d.draw(ax=ax_drawing, show=False)
		ax_drawing.axis('equal')

	if ax_pole_zero is not None:
		ax_pole_zero.grid()
		ax_pole_zero.set_xlabel('Re(s)')
		ax_pole_zero.set_ylabel('Im(s)')
		ax_pole_zero.axis('equal')
		ax_pole_zero.set_title('Pole-Zero')


def determine_unstable_fb(f_denom: Callable, initial_guess=1.0) -> float:

	def f_stability(fb):
		poles = f_denom(fb).roots()
		return np.amax(np.real(poles))

	bracketing_range = [0.0, initial_guess]

	if f_stability(0) >= 0:
		raise ValueError('Filter is unstable even at 0 feedback!')

	NUM_ITER = 8
	for _ in range(NUM_ITER):
		stab = f_stability(bracketing_range[1])

		if abs(stab) < 1e-9:
			return bracketing_range[1]

		if stab >= 0.0:
			break
		else:
			bracketing_range[0] = bracketing_range[1]
			bracketing_range[1] *= 2.0
	else:
		raise ValueError(f'Failed to determine filter stability in {NUM_ITER} searches')

	return scipy.optimize.brentq(f_stability, bracketing_range[0], bracketing_range[1])


def determine_unstable_fb_and_resonant_frequencies(f_denom: Callable, initial_guess=1.0) -> Tuple[float, List[float]]:

	unstable_fb = determine_unstable_fb(f_denom=f_denom, initial_guess=initial_guess)

	poles_at_instability = f_denom(unstable_fb).roots()

	# It's possible for there to be multiple resonant frequencies - e.g. 1LP 1HP 1NT feedback

	unstable_poles = [p for p in poles_at_instability if np.real(p) > -1e-6]
	assert all(np.real(p) < 1e-6 for p in unstable_poles)

	unstable_positive_imag_poles = [p for p in unstable_poles if np.imag(p) >= 0.0]

	resonant_freqs = [np.imag(p) for p in unstable_positive_imag_poles]
	assert all(f >= 0.0 for f in resonant_freqs)

	return unstable_fb, resonant_freqs


class CascadeFilterArchitecture:

	numer_poly = Polynomial(1)

	@staticmethod
	def derive_denom_fb_poly(
		fb_taps: Iterable[float],
		num_stages: Optional[int] = None,
		) -> Callable[[float], Polynomial]:

		if num_stages is None:
			num_stages = len(fb_taps)

		fb_poly = 0.0

		for stage_idx, tap in enumerate(fb_taps):
			if not tap:
				continue

			power = num_stages - stage_idx - 1
			fb_poly += tap * (S_PLUS_1 ** power)

		return fb_poly

	def __init__(
			self,
			name: str,
			fb_taps: Iterable[float],
			#output_taps: Optional[Iterable[float]] = None,  # TODO: support output_taps
			num_poles: Optional[int] = None,
			):

		output_taps = None

		self.name = name

		self.fb_taps = fb_taps

		if num_poles is not None:
			self.num_poles = num_poles
		elif output_taps is None:
			self.num_poles = len(fb_taps)
		else:
			self.num_poles = max(len(fb_taps), len(output_taps) - 1)

		if output_taps is None:
			self.output_taps = [0.0] * self.num_poles + [1.0]
		else:
			self.output_taps = output_taps

		self.denom_non_fb_poly = S_PLUS_1 ** self.num_poles
		self.denom_fb_poly = self.derive_denom_fb_poly(fb_taps=fb_taps, num_stages=self.num_poles)

		# TODO: better handling of multiple resonant frequencies
		self.unstable_fb, self.resonant_freqs = determine_unstable_fb_and_resonant_frequencies(self.get_denom_poly, initial_guess=float(self.num_poles))

	def get_denom_poly(self, fb):
		return self.denom_non_fb_poly - fb * self.denom_fb_poly

	def plot(self, w: Optional[Iterable[float]] = None, plot_group_delay = False):

		if w is None:
			w = _get_plot_freqs(self.resonant_freqs[0] if len(self.resonant_freqs) == 1 else None)

		_plot_filter(
			w=w,
			title=self.name,
			unstable_fb=self.unstable_fb,
			f_numerator=lambda fb: self.numer_poly,
			f_denominator=self.get_denom_poly,
			f_drawing=lambda d: self.draw(d),
			f_open_loop_response=self.open_loop_response,
			plot_group_delay=plot_group_delay,
		)

	def response(self, w: Union[float, np.ndarray], fb: float) -> Union[float, np.ndarray]:
		s = 1j * w
		denom = self.get_denom_poly(fb)
		return self.numer_poly(s) / denom(s)

	def open_loop_response(self, w: Union[float, np.ndarray], fb: Optional[float] = None) -> Union[float, np.ndarray]:
		if fb is None:
			fb = self.unstable_fb

		s = 1j * w

		resp = None
		for tap_idx, tap_ampl in enumerate(self.fb_taps):
			denom_poly = S_PLUS_1 ** (tap_idx + 1)
			this_pole_resp = tap_ampl / denom_poly(s)
			if resp is None:
				resp = this_pole_resp
			else:
				resp += this_pole_resp

		assert resp is not None
		resp *= fb
		return resp


	def draw(self, d):

		#single_fb_tap = 1 == sum(ampl != 0.0 for ampl in self.fb_taps)
		single_output = 1 == sum(val != 0 for val in self.output_taps)

		fb_tap_idxs = [idx for idx, tap in enumerate(self.fb_taps) if tap]
		output_tap_idxs = [idx for idx, tap in enumerate(self.output_taps) if tap]

		# Draw input to FB summing node

		d += dsp.Dot().label('Input')
		d += dsp.Arrow().right(d.unit/2)
		d += (fb_sum := dsp.Sum())
		sum_size = fb_sum.E.x - fb_sum.W.x

		# Draw FB amp & arrows

		d += (fb_arrow := dsp.Arrow().at(fb_sum.N, dy=1.25*d.unit).down(d.unit/2))
		d += dsp.Amp().label('res')
		d += dsp.Arrow().toy(fb_sum.N)
		d += dsp.Line().at(fb_sum.E).right(d.unit/2)
		last_fb_anchor = fb_arrow.start

		# Draw stages

		is_first_fb = True

		for idx in range(self.num_poles):

			is_last_stage = idx == self.num_poles - 1

			this_fb_tap = self.fb_taps[idx] if idx < len(self.fb_taps) else 0.0
			this_output_tap = self.output_taps[idx + 1] if idx <= len(self.output_taps) else 0.0

			# Draw stage input & output arrows

			d += dsp.Arrow().right(d.unit/2)
			d += dsp.Filter(d='right', response='lp')
			d += dsp.Line().right(d.unit/2)

			anchor = d.here

			needs_dot = (this_fb_tap and this_output_tap) if is_last_stage else (this_fb_tap or this_output_tap)
			if needs_dot:
				d += dsp.Dot()

			# Draw feedback tap

			if this_fb_tap:

				d.push()

				is_last_fb_tap = idx == fb_tap_idxs[-1]

				if is_last_fb_tap:
					# Last tap - no summing node needed
					anchor_E = anchor_S = anchor_W = (anchor.x, last_fb_anchor.y)
				else:
					d += (fb_sum := dsp.Sum().at((anchor.x - 0.5*sum_size, last_fb_anchor.y)))
					anchor_E = fb_sum.E
					anchor_S = fb_sum.S
					anchor_W = fb_sum.W

				if is_first_fb:
					d += dsp.Line().at(anchor_W).to(last_fb_anchor)
				else:
					d += dsp.Arrow().at(anchor_W).to(last_fb_anchor)
				is_first_fb = False

				last_fb_anchor = anchor_E

				if this_fb_tap == 1:
					d += (dsp.Line if is_last_fb_tap else dsp.Arrow)().at(anchor).toy(anchor_S)
				else:
					d += dsp.Arrow().at(anchor).up(d.unit/2)
					d += dsp.Amp().label(f'{this_fb_tap:g}')
					d += (dsp.Line if is_last_fb_tap else dsp.Arrow)().toy(anchor_S)

				d.pop()

			# Draw output tap

			if this_output_tap:

				d.push()

				if single_output and is_last_stage:
					d += dsp.Arrow().right(d.unit/2).label('Output', loc='right')
				else:
					pass  # TODO: similar summing logic as with FB
					if this_output_tap == 1:
						d += dsp.Arrow().at(anchor).down(d.unit)
					else:
						d += dsp.Arrow().at(anchor).down(d.unit/2)
						d += dsp.Amp().label(f'{this_output_tap:g}')
						d += dsp.Arrow().down(d.unit/2)

				d.pop()


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)

	parser.add_argument('--delay', action='store_true', dest='plot_group_delay', help='Include group delay')

	group = parser.add_mutually_exclusive_group()
	group.add_argument('--basic', action='store_true', help='Basic filters only')
	group.add_argument('--stages', action='store_true', help='Cascade with various number of stages')
	group.add_argument('--unstable', action='store_true', help="Filters that may not be DC stable")
	group.add_argument('--two', action='store_true', help="2-pole filters")
	group.add_argument('--three', action='store_true', help="3-pole filters")
	group.add_argument('--eight', action='store_true', help="8-pole filters")

	return parser


def _plot_filter_comparison(filters, resonance=0.75):

	fig = plt.figure()
	gs = plt.GridSpec(3, 1, figure=fig)

	ax_mag = fig.add_subplot(gs[:2,:])
	ax_phase = fig.add_subplot(gs[-1,:], sharex=ax_mag)

	for filt in filters:
		w = _get_plot_freqs(filt.resonant_freqs[0] if len(filt.resonant_freqs) == 1 else None)
		fb = resonance * filt.unstable_fb
		resp = filt.response(w, fb=fb)
		mag = _mag(resp)
		ph = _phase(resp)
		line, = ax_mag.semilogx(w, mag, label=filt.name)
		color = line.get_color()
		ax_phase.semilogx(w, ph, label=filt.name, color=color)

	ax_mag.set_ylabel('Magnitude (dB)')
	ax_phase.set_ylabel('Phase (degrees)')

	for ax in [ax_mag, ax_phase]:
		ax.grid()
		ax.legend()


def plot(args, verbose=False):

	filter_comparison_kwargs_list = [
		dict(fb_taps=[0, 0, 0, -1], name='4p cascade'),
		dict(fb_taps=[0, 1, -2, 1], name='4p cascade, 4p bandpass feedback'),
		dict(fb_taps=[0, 1, 0, -1], name='4p cascade, 4p alt bandpass feedback'),
		dict(fb_taps=[1, -1, 0, 0], name='4p cascade, 2p bandpass feedback'),
	]

	if args.basic:
		cascade_kwargs_list = [
				dict(fb_taps=[0, 0, -1], name='3p cascade'),
				dict(fb_taps=[0, 0, 0, -1], name='4p cascade'),
				dict(fb_taps=[0, 0, 0, 0, 0, 0, 0, -1], name='8p cascade'),
				dict(fb_taps=[1, -1], name='2p cascade, 2p bandpass feedback'),
			]
	elif args.stages:
		cascade_kwargs_list = [
			dict(fb_taps=[0, 0, -1], name='3p cascade'),
			dict(fb_taps=[0, 0, 0, -1], name='4p cascade'),
			dict(fb_taps=[0, 0, 0, 0, -1], name='5p cascade'),
			dict(fb_taps=[0, 0, 0, 0, 0, -1], name='6p cascade'),
			dict(fb_taps=[0, 0, 0, 0, 0, 0, -1], name='7p cascade'),
			dict(fb_taps=[0, 0, 0, 0, 0, 0, 0, -1], name='8p cascade'),
			dict(fb_taps=[0, 0, 0, -1, 0, 0, 0, 0], name='8p cascade, 4p feedback'),
		]
	elif args.unstable:
		cascade_kwargs_list = [
			dict(fb_taps=[0, 1], name='2p cascade, +FB'),
			dict(fb_taps=[0, 0, -1], name='3p cascade, -FB'),
			dict(fb_taps=[0, 0, 1], name='3p cascade, +FB'),
			dict(fb_taps=[0, 0, 0, -1], name='4p cascade, -FB'),
			dict(fb_taps=[0, 0, 0, 1], name='4p cascade, +FB'),
			dict(fb_taps=[0, 0, 0, 0, 0, 0, 0, -1], name='8p cascade, -FB'),
			dict(fb_taps=[0, 0, 0, 0, 0, 0, 0, 1], name='8p cascade, +FB'),
			dict(fb_taps=[1, -6, 12, -8], name='4p cascade, 1LP 3AP -FB'),
			dict(fb_taps=[-1, 6, -12, 8], name='4p cascade, 1LP 3AP +FB'),
			dict(fb_taps=[1, -2, 2], name='3p cascade, 1LP 1NT feedback'),
			dict(fb_taps=[0, 1, -2, 2], name='4p cascade, 1LP 2NT feedback'),
		]
	elif args.two:
		cascade_kwargs_list = [
			dict(fb_taps=[1, -1], name='2p cascade, 2p bandpass feedback'),
			dict(fb_taps=[1, -2], name='2p cascade, 1LP 1AP feedback'),
		]
	elif args.three:
		cascade_kwargs_list = [
			dict(fb_taps=[0, 0, -1], name='3p cascade'),
			dict(fb_taps=[-1, 4, -4], name='3p cascade, 1LP 2AP feedback'),
			dict(fb_taps=[1, -3, 2], name='3p cascade, 1LP 1HP 1AP feedback'),
		]
	elif args.eight:
		cascade_kwargs_list = [
			dict(fb_taps=[0, 0, 0, 0, 0, 0, 0, -1], name='8p cascade'),
			dict(fb_taps=[0, 0, 0, -1, 0, 0, 0, 0], name='8p cascade, 4p feedback'),
			dict(fb_taps=[0, 0, 0, -1], name='4p cascade'),
		]
	else:
		cascade_kwargs_list = [
			dict(fb_taps=[0, 0, -1], name='3p cascade'),
			dict(fb_taps=[0, 0, 0, -1], name='4p cascade'),
			dict(fb_taps=[0, 0, 0, 0, 0, 0, 0, -1], name='8p cascade'),
			dict(fb_taps=[0, 0, 0, -1, 0, 0, 0, 0], name='8p cascade, 4p feedback'),
			dict(fb_taps=[0, 0, -0.5, -0.5], name='4p cascade, "3.5 pole" feedback'),

			dict(fb_taps=[0, 1, -2, 1], name='4p cascade, 4p bandpass feedback'),
			dict(fb_taps=[0, 1, 0, -1], name='4p cascade, 4p alt bandpass feedback'),
			dict(fb_taps=[1, -1, 0, 0], name='4p cascade, 2p bandpass feedback'),
			dict(fb_taps=[1, -1], name='2p cascade, 2p bandpass feedback'),

			dict(fb_taps=[0, 0, 1, -1], name='4p cascade, 3LP 1HP feedback'),
			dict(fb_taps=[0, 0, -1, 1], name='4p cascade, inv 3LP 1HP feedback'),
			dict(fb_taps=[-1, 3, -3, 1], name='4p cascade, 1LP 3HP feedback'),
			dict(fb_taps=[1, -3, 3, -1], name='4p cascade, inv 1LP 3HP feedback'),

			dict(fb_taps=[-1, 4, -4], name='3p cascade, 1LP 2AP feedback'),
			dict(fb_taps=[1, -6, 12, -8], name='4p cascade, 1LP 3AP feedback'),
			
			dict(fb_taps=[1, -3, 4, -2], name='4p cascade, 1LP 1HP 1NT feedback'),
			dict(fb_taps=[1, -3, 2, 0], name='4p cascade, 1LP 1HP 1AP feedback'),
			dict(fb_taps=[1, -5, 8, -4], name='4p cascade, 1LP 1HP 2AP feedback'),

			dict(fb_taps=[1, 2-2*sqrt(2), 8-6*sqrt(2), 8-6*sqrt(2)], name='The "impossible filter" (4p out)'),
		]

	if verbose:
		print()
		print('%-40s %-5s %-8s %-8s %-8s %-8s %-8s %-8s %-8s %-8s' % ('', '',  'No FB',    '',         '',      '',      'Max FB', '',    '',         ''))
		print('%40s %5s %8s %8s %8s %8s %8s %8s %8s %8s' %       ('', 'poles', 'Slope',    'Slope',    'Phase', 'Phase', 'Value',  'DC',   'Slope',    'Resonant'))
		print('%40s %5s %8s %8s %8s %8s %8s %8s %8s %8s' %       ('', '',      'Low',      'High',     'Low',   'High',  '',       'Gain', 'Low',      'freq'))
		print('%40s %5s %8s %8s %8s %8s %8s %8s %8s %8s' %       ('', '',      '(dB/oct)', '(dB/oct)', '(deg)', '(deg)', '',       '(dB)', '(dB/oct)', ''))
		print()

	for kwargs in cascade_kwargs_list:

		filt = 	CascadeFilterArchitecture(**kwargs)

		if verbose:

			w_test_points = np.array([0.0, 0.001, 0.002, 500.0, 1000.0])

			r_test_no_fb = filt.response(w_test_points, 0.0)
			r_test_max_fb = filt.response(w_test_points, filt.unstable_fb * 0.9999)

			gain_zero_max_fb = _mag(r_test_max_fb[0])

			low_slope_no_fb = _mag(r_test_no_fb[2]) - _mag(r_test_no_fb[1])
			high_slope_no_fb = _mag(r_test_no_fb[4]) - _mag(r_test_no_fb[3])

			low_slope_max_fb = _mag(r_test_max_fb[2]) - _mag(r_test_max_fb[1])
			high_slope_max_fb = _mag(r_test_max_fb[4]) - _mag(r_test_max_fb[3])

			low_phase_no_fb = _phase(r_test_no_fb[0])
			high_phase_no_fb = _phase(r_test_no_fb[-1])

			low_phase_max_fb = _phase(r_test_max_fb[0])
			high_phase_max_fb = _phase(r_test_max_fb[-1])

			print('%-40s %5i %8i %8i %8i %8i %8.3f %8.1f %8i %8s' % (
				filt.name,
				filt.num_poles,

				int(round(low_slope_no_fb)),
				int(round(high_slope_no_fb)),
				int(round(low_phase_no_fb)),
				int(round(high_phase_no_fb)),

				filt.unstable_fb,
				gain_zero_max_fb,
				int(round(low_slope_max_fb)),

				' '.join(['%.3f' % f for f in filt.resonant_freqs]),
			))

		filt.plot(plot_group_delay=args.plot_group_delay)

	comparison_filters = [
		CascadeFilterArchitecture(**kwargs)
		for kwargs in filter_comparison_kwargs_list
	]

	_plot_filter_comparison(comparison_filters)

	plt.show()


def main(args):
	plot(args, verbose=True)

