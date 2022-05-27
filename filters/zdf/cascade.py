#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from math import pi, tanh
import math
from typing import Optional

from generation.signal_generation import gen_sine, gen_saw
from solvers.iter_stats import IterStats
from utils.utils import to_dB

from filters.zdf.onepole import ZdfOnePoleBase, TrapzOnePole, TanhInputTrapzOnePole, LadderOnePole, OtaOnePole, OtaOnePoleNegative

MAX_NUM_ITER = 20
EPS = 1e-5


"""
Other future things to implement in single filter stage:
- With nonlinear buffers (e.g. BJT, FET, Darlington - also crossover distortion?)
- Asymmetry
- CV leakage
"""

"""
Test cases to determine best algorithms/approximations/estimates:

Sawtooth

Combination of sines
	One paper I found used 110 Hz + 155 Hz, which seems good (IM is at 75/200, HD2 at 220/310)

Variety of gain levels

Variety of input frequencies

Variety cutoff frequencies

Instant transisions vs bandlimited

Square waves
	good case because they have fast transitions and are always at one end or the
	other (in heavy distortion region), yet the distortion wouldn't affect the
	wave if it weren't for the lowpass filtering

Different stages
1st stage might have different optimal parameters from 4th stage

Different resonance levels, including self-osc

Audio-rate FM
"""


def res_to_fb(res):
	# Res: 0-1, self-oscillation begins at 0.9

	if res < 0.:
		res = 0.
	elif res > 1.:
		res = 1.

	return (40. / 9.) * res


def fb_to_res(fb):
	res = (9. / 40.) * fb

	if res < 0.:
		res = 0.
	elif res > 1.:
		res = 1.

	return res


def res_to_gain_correction(res):
	return 1. + res_to_fb(res)


def linear_4p_equation(
		x: float,
		g: float,
		r: float,
		s,
		m: Optional[float] = None,
		mg: Optional[float] = None,
		mg4: Optional[float] = None,
) -> float:
	# m, mg, mg4 can be provided to make computation very slightly more efficient
	# (though this is Python, so that probably doesn't matter)

	if m is None:
		m = 1.0 / (1.0 + g)

	if mg is None:
		mg = m * g

	if mg4 is None:
		mg4 = math.pow(mg, 4)

	"""
	xr = x - (y * r)

	y0 = m*(g*xr + s[0])
	y1 = m*(g*y0 + s[1])
	y2 = m*(g*y1 + s[2])
	y3 = m*(g*y2 + s[3])

	y = m*(g*m*(g*m*(g*m*(g*xr + s[0]) + s[1]) + s[2]) + s[3])

	y = m*g*(m*g*(m*g*(m*g*xr + m*s[0]) + m*s[1]) + m*s[2]) + m*s[3]
	y = (m*m*m*m)*(g*g*g*g)*xr + (m*m*m*m)*(g*g*g)*s[0] + (m*m*m)*(g*g)*s[1] + (m*m)*g*s[2] + m*s[3]
	y = m4*g4*xr + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]

	y = m4*g4*xr + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]
	xr = x - y*r

	y = m4*g4*(x - y*r) + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]
	y = m4*g4*x - m4*g4*y*r + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3]

	y = ( m4*g4*x + m4*g3*s[0] + m3*g2*s[1] + m2*g*s[2] + m*s[3] ) / ( 1.0 + r*m4*g4 )
	y = ( mg*(mg*(mg*(mg*x + m*s[0]) + m*s[1]) + m*s[2]) + m*s[3] ) / ( 1.0 + r*m4*g4 )
	"""
	return (mg * (mg * (mg * (mg * x + m * s[0]) + m * s[1]) + m * s[2]) + m * s[3]) / (1.0 + r * mg4)


# TODO: implement ProcessorBase


class IterativeCascadeFilterBase:
	def __init__(self, wc, poles, stats_outer: IterStats, res=0):
		""""""

		self.stats_outer = stats_outer
		self.poles = poles

		self.iterate = True
		self.res = res
		self.prev_y = 0

		self.set_freq(wc)

	def set_freq(self, wc):
		for n in range(4):
			self.poles[n].set_freq(wc)

	def process_buf(self, input_sig):
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n] = self.process_tick(x)
		return y

	def get_estimate(self, x):
		return linear_4p_equation(x, self.poles[0].g, self.res, [p.s for p in self.poles])

	def process_tick(self, x):

		y_est = self.get_estimate(x)

		if self.iterate:

			"""
			The iterative solving here is simple approach, but likely not the fastest way of doing it:
			* Each pole solves itself iteratively until its error is low enough
			* Then we do 1 iteration of the outer loop
			* Then we re-solve each pole again
			* And so on, until the outer loop error is low enough

			Alternatively, could:
			* Each pole runs 1 iteration
			* Then we do 1 iteration of the outer loop
			* Then 1 more iteration of each pole again
			* And so on, until all errors are low enough

			I suspect this may be faster overall, although it's also possible it could have stability issues
			Would need to try it out and see.

			Related to this, the current way this uses IterStats doesn't really make sense.
			I suspect the individual pole IterStats would converge relatively slowly on the first outer loop, but then 
			faster on later iterations, as the outer loop converges as well.

			Or at least they could, except there's another problem here:
			Right now the individual poles save no state information from the previous outer loop solves - they solve
			from scratch every time!
			"""

			new_state = [0.0 for _ in range(4)]
			last_iter_y = [None for _ in range(4)]

			prev_abs_err = None
			prev_y = y_est

			# Stats vars
			estimate = y_est
			errs = []
			ys = [estimate]
			success = False

			for n_iter in range(MAX_NUM_ITER):

				# TODO: nonlinearities in output buffer path (which will feed back into resonance)

				xr = x - (y_est * self.res)

				y = xr

				for pole_idx, pole in enumerate(self.poles):
					estimate = last_iter_y[pole_idx]
					this_pole_y, new_state[pole_idx] = pole.process_tick_no_state_update(y, estimate=estimate)
					last_iter_y[pole_idx] = this_pole_y
					y = this_pole_y

				y_est = y

				ys += [y]

				err = y - prev_y
				abs_err = abs(err)
				errs += [abs_err]

				if abs_err < EPS:
					success = True
					break

				if (prev_abs_err is not None) and (abs_err >= prev_abs_err):
					print('Warning: failed to converge! Falling back to initial estimate')
					print('errs: ' + repr(errs))
					print('ys: ' + repr(ys))
					y = estimate
					break

				prev_y = y
				prev_abs_err = abs_err

			for pole, s in zip(self.poles, new_state):
				pole.s = s

			if self.stats_outer is not None:
				self.stats_outer.add(
					success=success,
					est=estimate,
					n_iter=n_iter + 1,
					final=y,
					err=errs)

		else:
			y = x - (y_est * self.res)
			for pole in self.poles:
				y = pole.process_tick(y)

		self.prev_y = y
		return y * res_to_gain_correction(fb_to_res(self.res))


# TODO: this is almost entirely duplicated with IterativeCascadeFilterBase, use that instead
class LinearCascadeFilterIterative:
	def __init__(self, wc, res=0, stats_outer=None):

		self.poles = [TrapzOnePole(wc) for n in range(4)]
		self.res = res
		self.prev_y = 0

		if stats_outer is None:
			self.stats_outer = IterStats('Linear Outer Loop')
		else:
			self.stats_outer = stats_outer

	def set_freq(self, wc):
		for n in range(4):
			self.poles[n].set_freq(wc)

	def process_buf(self, input_sig):
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n] = self.process_tick(x)
		return y

	def process_tick(self, x):

		out_est = self.prev_y  # Not a great estimate
		iterate = True

		if iterate:

			new_state = [0.0 for n in range(4)]

			prev_abs_err = None
			prev_y = out_est

			# Stats vars
			estimate = out_est
			errs = []
			ys = [estimate]
			success = False

			for n_iter in range(MAX_NUM_ITER):

				input_estimate = x - (out_est * self.res)

				y = input_estimate

				for n, pole in enumerate(self.poles):
					y, new_state[n] = pole.process_tick_no_state_update(y)

				out_est = y

				ys += [y]

				err = y - prev_y
				abs_err = abs(err)
				errs += [abs_err]

				if abs_err < EPS:
					success = True
					break

				if (prev_abs_err is not None) and (abs_err >= prev_abs_err):
					print('Warning: failed to converge! Falling back to initial estimate')
					print('errs: ' + repr(errs))
					print('ys: ' + repr(ys))
					# return estimate
					y = estimate
					break

				prev_y = y
				prev_abs_err = abs_err

			for pole, s in zip(self.poles, new_state):
				pole.s = s

			if self.stats_outer is not None:
				self.stats_outer.add(
					success=success,
					est=estimate,
					n_iter=n_iter + 1,
					final=y,
					err=errs)

		else:

			y = x - (out_est * self.res)
			for pole in self.poles:
				y = pole.process_tick(y)

		self.prev_y = y

		return y * res_to_gain_correction(fb_to_res(self.res))


# TODO: de-duplicate this with filters.cascade.LinearCascadeFilter
class LinearCascadeFilter:

	def __init__(self, wc, res=0):

		self.s = [0.0 for n in range(4)]
		self.res = res

		self.set_freq(wc)

	def set_freq(self, wc):
		self.g = self.g = ZdfOnePoleBase.freq_to_gain(wc)
		self.m = 1.0 / (self.g + 1.0)
		self.mg = self.m * self.g
		self.mg4 = pow(self.mg, 4)
		self.recipmg = 1.0 / self.mg

	def process_buf(self, input_sig):
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n] = self.process_tick(x)
		return y

	def process_tick(self, x):

		# This code would be a mess if we didn't define some abbreviated var names
		# Hopefully the Python JIT should be smart enough to optimize these out
		g = self.g
		m = self.m  # m = 1/(1+g)
		mg = self.mg  # g = m*g = g/(1+g)
		rmg = self.recipmg  # recipmg = 1/mg = (1+g)/g
		mg4 = self.mg4  # mg4 = mg^4

		s = self.s
		r = self.res

		y = linear_4p_equation(x, self.g, self.res, self.s, m=self.m, mg=self.mg, mg4=self.mg4)

		# These two methods are be the same
		# working backwards is probably slightly more efficient,
		# or at least if frequency is constant
		if True:
			# Work forwards
			xr = x - (y * r)
			y0 = m * (g * xr + s[0])
			y1 = m * (g * y0 + s[1])
			y2 = m * (g * y1 + s[2])
			y3 = m * (g * y2 + s[3])

		else:
			# Work backwards
			y3 = y
			y2 = (y3 - m * s[3]) * rmg
			y1 = (y2 - m * s[2]) * rmg
			y0 = (y1 - m * s[1]) * rmg

		s[0] = 2.0 * y0 - s[0]
		s[1] = 2.0 * y1 - s[1]
		s[2] = 2.0 * y2 - s[2]
		s[3] = 2.0 * y3 - s[3]

		return y * res_to_gain_correction(fb_to_res(self.res))


class LadderFilter(IterativeCascadeFilterBase):
	def __init__(self, wc, res=0):
		pole_stats = [IterStats('Ladder pole %i' % (i + 1)) for i in range(4)]
		poles = [LadderOnePole(wc, iter_stats=pole_stats[n]) for n in range(4)]
		super().__init__(wc=wc, res=res, poles=poles, stats_outer=IterStats('Ladder outer loop'))

	def get_estimate(self, x):
		# Pre-applying tanh to input makes it slightly more accurate than strictly linear case
		return super().get_estimate(tanh(x))


class OtaFilter(IterativeCascadeFilterBase):
	def __init__(self, wc, res=0):
		pole_stats = [IterStats('OTA pole %i' % (i + 1)) for i in range(4)]
		poles = [OtaOnePole(wc, iter_stats=pole_stats[n]) for n in range(4)]
		super().__init__(wc=wc, res=res, poles=poles, stats_outer=IterStats('OTA outer loop'))


class OtaNegFilter(IterativeCascadeFilterBase):
	def __init__(self, wc, res=0):
		pole_stats = [IterStats('OTA neg pole %i' % (i + 1)) for i in range(4)]
		poles = [OtaOnePoleNegative(wc, iter_stats=pole_stats[n]) for n in range(4)]
		super().__init__(wc=wc, res=res, poles=poles, stats_outer=IterStats('OTA neg outer loop'))


def do_fft(x, n_fft, window=False):
	
	if window:
		y = np.fft.fft(x * np.hamming(len(x)), n=n_fft)
	else:
		y = np.fft.fft(x, n=n_fft)
	
	f = np.fft.fftfreq(n_fft, 1.0)
	
	# Only take first half
	y = y[0:len(y)//2]
	f = f[0:len(f)//2]
	
	return y, f


def plot_nonlin_filter(fc=0.1, f_saw=0.01, res=1.5, gain=2.0, n_samp=2048):
	
	filts = [
		dict(name='4P Linear', filt=LinearCascadeFilter(fc, res)),
		dict(name='4P Ladder', filt=LadderFilter(fc, res)),
		dict(name='4P Ota', filt=OtaFilter(fc, res)),
		dict(name='4P Ota Negative', filt=OtaNegFilter(fc, res)),
	]

	x = gen_saw(f_saw, n_samp) * gain * 0.5
	
	X, f = do_fft(x, n_fft=n_samp, window=True)
	X = to_dB(np.abs(X))
	
	for filt in filts:
		
		y = filt['filt'].process_buf(x)
		
		Y, _ = do_fft(y, n_fft=n_samp, window=True)
		Y = to_dB(np.abs(Y))
		
		filt['y'] = y
		filt['Y'] = Y
	
	t = np.arange(n_samp)
	
	##### Plot filter responses #####

	fig = plt.figure()
	fig.suptitle('Nonlinear filters')

	plt.subplot(2, 1, 1)

	plt.title('f_in=%g, fc=%g, res=%g, gain=%g' % (f_saw, fc, res, gain))

	plt.plot(t, x, '.-', label='Input')

	for filt in filts:
		plt.plot(t, filt['y'], '.-', label=filt['name'])

	plt.legend()

	plt.xlim([0, 256])
	plt.grid()

	plt.subplot(2, 1, 2)

	plt.semilogx(f, X)
	for filt in filts:
		plt.semilogx(f, filt['Y'], label=filt['name'])

	plt.grid()

	#plt.subplot(3, 1, 3)
	#for n in filter_idxs_to_plot:
	#	plt.semilogx(f, filts[n]['Y'] - X)

	#plt.grid()


def plot_lin_4pole(fc=0.1, f_saw=0.01, res=0.0, n_samp=2048):
	
	filt_iterative = LinearCascadeFilterIterative(fc)
	filt_solved = LinearCascadeFilter(fc)
	
	filt_iterative.res = res
	filt_solved.res = res
	
	x = gen_saw(f_saw, n_samp) * 0.5
	
	y_iterative = filt_iterative.process_buf(x)
	y_solved    = filt_solved.process_buf(x)

	y_diff = y_iterative - y_solved
	
	amp_x, f         = do_fft(x, n_fft=n_samp, window=True)
	amp_iterative, _ = do_fft(y_iterative, n_fft=n_samp, window=True)
	amp_solved, _    = do_fft(y_solved, n_fft=n_samp, window=True)
	
	amp_x         = to_dB(np.abs(amp_x))
	amp_iterative = to_dB(np.abs(amp_iterative))
	amp_solved    = to_dB(np.abs(amp_solved))
	
	t = np.arange(n_samp)
	
	fig = plt.figure()
	fig.suptitle('Linear filter method comparison')
	
	plt.subplot(3, 1, 1)

	plt.title('f_in=%g, fc=%g, res=%g' % (f_saw, fc, res))

	plt.plot(t, x, '.-', t, y_iterative, '.-', t, y_solved, '.-')
	plt.legend(['Input', 'Linear solved iteratively', 'Linear solved algebraically'])
	plt.xlim([0, 256])
	plt.grid()
	
	plt.subplot(3, 1, 2)
	plt.semilogy(t, np.abs(y_diff), 'r.-')
	plt.xlim([0, 256])
	plt.grid()
	plt.ylabel('Diff')

	plt.subplot(3, 1, 3)
	plt.semilogx(f, amp_x, f, amp_iterative, f, amp_solved)
	plt.grid()

	print('Max difference between iterative & solved: %f' % np.max(np.abs(y_diff)))


def plot_step(fc: float, res: float, n_samp_per_level: int):
	levels = [
		0, 0.1, 0.2, 0.3,
		0, 0.25, 0.5, 0.75,
		0, 0.5, 1.0, 1.5,
		0, 1.0, 2.0, 3.0,
		0, 2.0, 4.0, 6.0,
		0, 4.0, 8.0, 12.0,
	]

	x = []
	for level in levels:
		x += [float(level)] * n_samp_per_level

	x = np.array(x)

	linear_filter = LinearCascadeFilter(fc, res)
	y_linear = linear_filter.process_buf(x)

	filter_specs = [
		dict(name='4P Ladder', filt=LadderFilter(fc, res)),
		dict(name='4P Ota', filt=OtaFilter(fc, res)),
	]

	fig, [ax_step, ax_slope, ax_err] = plt.subplots(3, 1)
	fig.suptitle('Step responses')

	#ax_step.plot(x, '-', label='Input')
	#next(ax_slope._get_lines.prop_cycler)
	#next(ax_err._get_lines.prop_cycler)

	ax_step.plot(y_linear, '-', label='Linear')
	ax_slope.plot(np.diff(y_linear), label='Linear')
	next(ax_err._get_lines.prop_cycler)

	for filter_spec in filter_specs:
		filt = filter_spec['filt']
		y = filt.process_buf(x)

		if 'negate' in filter_spec and filter_spec['negate']:
			y = -y

		delta = np.diff(y)

		err = y_linear - y

		ax_step.plot(y, label=filter_spec['name'])
		ax_err.plot(err, label=filter_spec['name'])
		ax_slope.plot(delta, label=filter_spec['name'])

	ax_step.grid()
	ax_step.legend()

	ax_slope.grid()
	ax_slope.legend()
	ax_slope.set_ylabel('Slope (sample delta)')

	ax_err.grid()
	ax_err.legend()
	ax_err.set_ylabel('Error from linear')


def plot(args=None):
	plot_nonlin_filter(fc=0.1, f_saw=0.01, gain=4.0, n_samp=2048)
	plot_lin_4pole(fc=0.1, f_saw=0.01, res=1.5, n_samp=2048)
	plot_step(fc=0.01, res=1.5, n_samp_per_level=1000)
	plt.show()


def main(args=None):
	plot(args)
