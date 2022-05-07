#!/usr/bin/env python3

from copy import deepcopy
from math import tanh, pi, pow
import math
import numpy as np
from typing import Optional, Tuple

from solvers.iter_stats import IterStats
import solvers.solvers as solvers


solvers.legacy_max_num_iter = 20
solvers.legacy_eps = 1e-5


def freq_to_gain(wc):
	return math.tan(pi*wc)


def res_to_fb(res):
	# Res: 0-1, self-oscillation begins at 0.9

	if res < 0.:
		res = 0.
	elif res > 1.:
		res = 1.

	return (40./9.) * res


def fb_to_res(fb):

	res = (9./40.) * fb

	if res < 0.:
		res = 0.
	elif res > 1.:
		res = 1.

	return res


def res_to_gain_correction(res):
	return 1. + res_to_fb(res)


def svf_res_q_mapping(res):
	"""
	:param res: 0-4, self-osc at 3.6 (0.9*4)
	:return: q (lowercase)
	"""

	# Q = 0.5 # 0.5 (no res) to infinity (max res)
	# q = 1. / Q # 2.0 to 0

	adj_res = res/4.0 * 1.0/0.9
	adj_res = math.sqrt(adj_res)
	q = 2.0 - 2.0*adj_res

	if q < 0.0:
		q = 0.0

	if q > 2.0:
		q = 2.0

	#print("res=%.1f, q=%.1f" % (res, q))

	return q

# ######### One-pole filters ##########


class ZdfOnePoleBase:

	def __init__(self, wc):
		self.set_freq(wc)
		self.s = 0.

		# These are just for (attempted) improvements of estimator
		self.prev_x = 0.
		self.prev_ext = 0.
		self.prev_y = 0.

	def set_freq(self, wc: float) -> None:
		self.g = freq_to_gain(wc)
		self.m = 1. / (self.g + 1.)

	def get_estimate(self, x: float) -> float:

		if True:
			# calculate linear case
			est = self.m * (self.g*x + self.s)

		elif False:
			# integrator state
			est = self.s

		else:
			# Use previous
			if abs(self.prev_x) > 0.01:
				est = (self.prev_y)/(self.prev_x) * x
			else:
				est = self.prev_y - self.prev_x + x

		# Improve linear estimate with knowledge of previous
		#est = self.prev_y - self.prev_ext + est

		self.prev_x = x
		self.prev_ext = est
		return est

	def process_tick_no_state_update(self, x: float) -> Tuple[float, float]:
		raise NotImplementedError('To be implemented by child class!')

	def process_tick(self, x: float) -> float:
		y, s = self.process_tick_no_state_update(x)
		self.s = s
		self.prev_y = y
		return y

	def process_buf(self, input_sig: np.ndarray) -> np.ndarray:
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n] = self.process_tick(x)
		return y


class TrapzOnePole(ZdfOnePoleBase):

	def __init__(self, wc):
		super().__init__(wc)
		#print('Linear filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def process_tick_no_state_update(self, x):

		# y = g*(x - y) + s
		# y = (g*x + s) / (g + 1)
		# y = m * (g*x + s) for m = 1/(1+g)

		y = self.m * (self.g*x + self.s)
		s = 2.*y - self.s

		return y, s


class TanhInputTrapzOnePole(ZdfOnePoleBase):

	def __init__(self, wc):
		super().__init__(wc)
		#print('Tanh filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def process_tick_no_state_update(self, x):

		# y = g*(tanh(x) - y) + s
		# y = (g*x + s) / (g + 1)
		# y = m * (g*x + s) for m = 1/(1+g)

		x = tanh(x)
		y = self.m * (self.g*x + self.s)
		s = 2.0*y - self.s

		return y, s


class LadderOnePole(ZdfOnePoleBase):

	def __init__(self, wc, iter_stats=None, use_newton=True):
		super().__init__(wc)
		if iter_stats is None:
			self.iter_stats = IterStats('1P Ladder')
		else:
			self.iter_stats = iter_stats
		#print('Ladder filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

		self.use_newton = use_newton

	def get_estimate(self, x):
		# Pre-applying tanh to input makes it slightly more accurate than strictly linear case
		return super().get_estimate(tanh(x))

	def process_tick_no_state_update(self, x):

		# y = g*(tanh(x) - tanh(y)) + s`

		def f(y): return y + self.g*(tanh(y) - tanh(x)) - self.s

		est = self.get_estimate(x)

		if self.use_newton:
			def df(y): return self.g * math.pow(math.cosh(y), -2.0) + 1
			y = solvers.solve_nr_legacy(f, df, est, iter_stats=self.iter_stats)

		else:
			y = solvers.solve_iterative_legacy(f, estimate=est, iter_stats=self.iter_stats)

		s = 2.0*y - self.s

		return y, s


class OtaOnePole(ZdfOnePoleBase):

	def __init__(self, wc, iter_stats=None, use_newton=True):
		super().__init__(wc)
		if iter_stats is None:
			self.iter_stats = IterStats('1P OTA')
		else:
			self.iter_stats = iter_stats
		self.use_newton = use_newton
		#print('OTA filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def process_tick_no_state_update(self, x):

		# y = g*tanh(x - y) + s

		def f(y): return y + self.g*tanh(y - x) - self.s

		est = self.get_estimate(x)

		if self.use_newton:
			def df(y): return self.g * math.pow(math.cosh(x - y), -2.0) + 1
			y = solvers.solve_nr_legacy(f, df, est, iter_stats=self.iter_stats)

		else:
			y = solvers.solve_iterative_legacy(f, estimate=est, iter_stats=self.iter_stats)

		s = 2.0*y - self.s

		return y, s


class OtaOnePoleNegative(ZdfOnePoleBase):

	def __init__(self, wc, iter_stats=None, use_newton=True):
		super().__init__(wc)
		if iter_stats is None:
			self.iter_stats = IterStats('1P OTA Negative')
		else:
			self.iter_stats = iter_stats
		self.use_newton = use_newton
		#print('OTA negative filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def get_estimate(self, x):

		# Calculate linear case
		# FIXME: this doesn't work
		#return (self.s - self.g*x) / (1.0 - self.g)
		#return self.m * (self.s - self.g*x)

		# integrator state
		return self.s

	def process_tick_no_state_update(self, x):

		# y = g*tanh(-x - y) + s

		est = self.get_estimate(x)

		def f(y): return y + self.g*tanh(x + y) - self.s

		if self.use_newton:
			def df(y): return self.g * math.pow(math.cosh(x + y), -2.0) + 1
			y = solvers.solve_nr_legacy(f, df, est, iter_stats=self.iter_stats)

		else:
			y = solvers.solve_iterative_legacy(f, estimate=est, iter_stats=self.iter_stats)

		s = 2.0*y - self.s
		return y, s


########## Cascade filters ##########


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
		m = 1.0/(1.0 + g)

	if mg is None:
		mg = m*g

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
	return (mg*(mg*(mg*(mg*x + m*s[0]) + m*s[1]) + m*s[2]) + m*s[3]) / (1.0 + r*mg4)


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

			new_state = [0.0 for n in range(4)]

			prev_abs_err = None
			prev_y = y_est

			# Stats vars
			estimate = y_est
			errs = []
			ys = [estimate]
			success = False

			for n_iter in range(solvers.legacy_max_num_iter):

				# TODO: nonlinearities in output buffer path (which will feed back into resonance)

				xr = x - (y_est * self.res)

				y = xr

				for n, pole in enumerate(self.poles):
					# TODO: use the result from the previous outer loop iteration as initial estimate (see block comment above)
					y, new_state[n] = pole.process_tick_no_state_update(y)

				y_est = y

				ys += [y]

				err = y - prev_y
				abs_err = abs(err)
				errs += [abs_err]

				if abs_err < solvers.legacy_eps:
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
					n_iter=n_iter+1,
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

			for n_iter in range(solvers.legacy_max_num_iter):

				input_estimate = x - (out_est * self.res)

				y = input_estimate

				for n, pole in enumerate(self.poles):
					y, new_state[n] = pole.process_tick_no_state_update(y)

				out_est = y

				ys += [y]

				err = y - prev_y
				abs_err = abs(err)
				errs += [abs_err]

				if abs_err < solvers.legacy_eps:
					success = True
					break

				if (prev_abs_err is not None) and (abs_err >= prev_abs_err):
					print('Warning: failed to converge! Falling back to initial estimate')
					print('errs: ' + repr(errs))
					print('ys: ' + repr(ys))
					#return estimate
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
					n_iter=n_iter+1,
					final=y,
					err=errs)

		else:

			y = x - (out_est * self.res)
			for pole in self.poles:
				y = pole.process_tick(y)

		self.prev_y = y

		return y * res_to_gain_correction(fb_to_res(self.res))


# TODO: de-duplicate this with cascade.LinearCascadeFilter
class LinearCascadeFilter:

	def __init__(self, wc, res=0):

		self.s = [0.0 for n in range(4)]
		self.res = res

		self.set_freq(wc)

	def set_freq(self, wc):
		self.g = self.g = freq_to_gain(wc)
		self.m = 1.0 / (self.g + 1.0)
		self.mg = self.m*self.g
		self.mg4 = pow(self.mg,4)
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
		m = self.m         # m = 1/(1+g)
		mg = self.mg       # g = m*g = g/(1+g)
		rmg = self.recipmg # recipmg = 1/mg = (1+g)/g
		mg4 = self.mg4     # mg4 = mg^4

		s = self.s
		r = self.res

		y = linear_4p_equation(x, self.g, self.res, self.s, m=self.m, mg=self.mg, mg4=self.mg4)

		# These two methods are be the same
		# working backwards is probably slightly more efficient,
		# or at least if frequency is constant
		if True:
			# Work forwards
			xr = x - (y * r)
			y0 = m*(g*xr + s[0])
			y1 = m*(g*y0 + s[1])
			y2 = m*(g*y1 + s[2])
			y3 = m*(g*y2 + s[3])

		else:
			# Work backwards
			y3 = y
			y2 = (y3 - m*s[3]) * rmg
			y1 = (y2 - m*s[2]) * rmg
			y0 = (y1 - m*s[1]) * rmg

		s[0] = 2.0*y0 - s[0]
		s[1] = 2.0*y1 - s[1]
		s[2] = 2.0*y2 - s[2]
		s[3] = 2.0*y3 - s[3]

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
		pole_stats = [IterStats('OTA pole %i' % (i+1)) for i in range(4)]
		poles = [OtaOnePole(wc, iter_stats=pole_stats[n]) for n in range(4)]
		super().__init__(wc=wc, res=res, poles=poles, stats_outer=IterStats('OTA outer loop'))


class OtaNegFilter(IterativeCascadeFilterBase):

	def __init__(self, wc, res=0):
		pole_stats = [IterStats('OTA neg pole %i' % (i+1)) for i in range(4)]
		poles = [OtaOnePoleNegative(wc, iter_stats=pole_stats[n]) for n in range(4)]
		super().__init__(wc=wc, res=res, poles=poles, stats_outer=IterStats('OTA neg outer loop'))


########## State-Variable Filters ##########


# Basic SVF, Forward Euler
class BasicSvf:

	def __init__(self, wc, res=0):

		self.q = svf_res_q_mapping(res)

		self.set_freq(wc)

		self.s = [0., 0.]

	def set_freq(self, wc):
		self.f = 2.*math.sin(pi*wc)

	def process_buf(self, input_sig, b_all_outs=False):

		if b_all_outs:
			y_lp, y_bp, y_hp = [np.zeros_like(input_sig) for n in range(2)]
			for n, x in enumerate(input_sig):
				y_lp[n], y_bp[n], y_hp[n] = self.process_tick(x)
			return y_lp, y_bp, y_hp
		else:
			y_lp = np.zeros_like(input_sig)
			for n, x in enumerate(input_sig):
				y_lp[n], _, _ = self.process_tick(x)
			return y_lp

	def process_tick(self, x):

		# Abbreviated names
		f = self.f
		s = self.s
		q = self.q

		y_lp = s[1] + f*s[0]
		y_hp = x - y_lp - q*s[0]
		y_bp = s[0] + f*y_hp

		s[1] = y_lp
		s[0] = y_bp

		return y_lp, y_bp, y_hp


class SvfLinear:

	def __init__(self, wc, res=0):

		self.q = svf_res_q_mapping(res)

		self.f = None
		self.set_freq(wc)

		self.stage1prev = 0.
		self.stage2prev = 0.

		self.s = [0., 0.]

	def set_freq(self, wc):
		self.f = 2.*math.sin(pi*wc)

	def process_buf(self, input_sig):
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n], _, _ = self.process_tick(x)
		return y

	def process_tick(self, x):

		zdf = True
		#zdf = False

		# Abbreviated names
		f = self.f * 0.5  # f always shows up below as f/2 (due to trapezoidal integration)
		q = self.q
		s = self.s  # reference

		if zdf:
			y_hp = (x - (f + q)*s[0] - s[1]) / (1 + f*f + q*f)
		else:
			y_hp = x - s[1] - q*s[0]

		y_bp = f*y_hp + s[0]
		y_lp = f*y_bp + s[1]

		s[0] = f*y_hp + y_bp
		s[1] = f*y_bp + y_lp

		return y_lp, y_bp, y_hp


class SvfLinearInputMixing:

	def __init__(self, wc, res=0):

		self.q = svf_res_q_mapping(res)

		self.set_freq(wc)

		self.stage1prev = 0.
		self.stage2prev = 0.

		self.s = [0., 0.]

	def set_freq(self, wc):
		self.f = 2.*math.sin(pi*wc)

	def process_buf(self, in_lp, in_bp=None, in_hp=None):

		n_samp = len(in_lp)

		if in_bp is None:
			in_bp = np.zeros_like(in_lp)
		else:
			assert(len(in_bp) == n_samp)

		if in_hp is None:
			in_hp = np.zeros_like(in_lp)
		else:
			assert(len(in_hp) == n_samp)

		y = np.zeros_like(in_lp)
		for n, x_lp, x_bp, x_hp in zip(range(n_samp), in_lp, in_bp, in_hp):
			y[n] = self.process_tick(x_lp, x_bp, x_hp)
		return y

	def process_tick(self, x_lp, x_bp, x_hp):

		#zdf = True
		zdf = False

		# Abbreviated names
		f = self.f * 0.5  # f always shows up below as f/2 (due to trapezoidal integration)
		q = self.q
		s = self.s  # reference

		s_prev = deepcopy(s)  # For debug outputs only

		# See http://www.cytomic.com/files/dsp/SvfInputMixing.pdf?

		"""
		Output mixing code:

		if zdf:
			y_hp = (x - (f + q)*s[0] - s[1]) / (1 + f*f + q*f)
		else:
			y_hp = x - s[1] - q*s[0]

		y_bp = f*y_hp + s[0]
		y_lp = f*y_bp + s[1]

		s[0] = f*y_hp + y_bp
		s[1] = f*y_bp + y_lp
		"""

		# In-progress stuff I don't understand anymore:
		"""
		Vlp = v1 + x_bp
		Ivccs = g * (Vlp - (Vout + Vln))
		Icap = (Vout - Vhp) + s

		y0 = f*(x_lp - y) + s[0]
		y1 =
		"""

		if abs(y) > 100.0:
			print('Error: integrator blew up')
			print('f = %f' % f)
			print('q = %f' % q)
			#print('y_hp = %f' % y_hp)
			#print('y_bp = %f' % y_bp)
			#print('y_lp = %f' % y_lp)
			print('Prev: s0 = %f, s1 = %f' % (s_prev[0], s_prev[1]))
			print('New:  s0 = %f, s1 = %f' % (s[0], s[1]))
			exit(1)

		return y


class NonlinSvf:
	"""
	Note: Unlimited resonance
	"""

	def __init__(self, wc, res=0, iter_stats=None, res_limit=None, fb_nonlin=False):
		# res_limit: one of None, 'tanh', 'hard'

		if not res_limit:
			self.res_limit = lambda x: x
		else:
			if res_limit == 'tanh':
				self.res_limit = lambda x: tanh(x)
			elif res_limit == 'hard':
				# Clip to 0.3, then
				clip_thresh = 0.3
				clip_mix = 0.75
				self.res_limit = lambda x: clip_mix*min(max(x, -clip_thresh), clip_thresh) + (1.0 - clip_mix)*x
			else:
				assert False, "Invalid res_limit: %s" % res_limit

		self.fb_nonlin = fb_nonlin

		if iter_stats is None:
			self.iter_stats = IterStats('tanh SVF')

		self.q = svf_res_q_mapping(res)

		self.set_freq(wc)

		self.stage1prev = 0.
		self.stage2prev = 0.

		self.s = [0., 0.]

	def set_freq(self, wc):
		self.f = 2.*math.sin(pi*wc)

	def process_buf(self, input_sig):
		y = np.zeros_like(input_sig)
		for n, x in enumerate(input_sig):
			y[n], _, _ = self.process_tick(x)
		return y

	def process_tick(self, x):

		# Abbreviated names
		f = self.f * 0.5  # f always shows up below as f/2 (due to trapezoidal integration)
		q = self.q
		s = self.s  # reference

		# First calculate linear code

		y_hp = (x - (f + q)*s[0] - s[1]) / (1. + f*f + q*f)
		y_bp = f * y_hp + s[0]
		y_lp = f * y_bp + s[1]

		# Now iterate

		for n in range(solvers.legacy_max_num_iter):

			# TODO: exit early if close enough
			#if abs_err < solvers.eps:
			#	success = True
			#	break

			# Bandpass is tanh'ed from Hp
			y_bp = f * tanh(y_hp) + s[0]

			# Lowpass is tanh'ed from Lp
			y_lp = f * tanh(y_bp) + s[1]

			# Highpass is only tanh'ed if resonance limited

			if self.fb_nonlin:
				res = q * tanh(y_bp)
			else:
				res = q * y_bp

			# Resonance limiting
			# (limits the max difference from y_bp, not the max res amount)

			# Like this, I think:
			#res = self.res_limit(res - y_hp) + y_hp

			# This appears to be another way:
			res += 0.07*self.res_limit(y_bp)

			y_hp = x - y_lp - res


		s[0] = f*y_hp + y_bp
		s[1] = f*y_bp + y_lp

		return y_lp, y_bp, y_hp
