#!/usr/bin/env python3

from math import pi, tanh, cosh, exp, tan
from typing import Tuple, Optional, Union, Any

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

from filters.allpass import FractionalDelayAllpass
from filters.filter_base import FilterBase
from filters.zdf.onepole import ZdfOnePoleBase, Rk4OnePoleBase
from generation.signal_generation import gen_sine, gen_saw
from solvers.iter_stats import IterStats
import solvers.solvers as solvers
from utils.utils import clip, to_dB


THERMAL_VOLTAGE = 0.02585

V_DROP_POS = 3.0
V_DROP_NEG = 1.5

Vt = THERMAL_VOLTAGE
Is = 1.1e-18


def lm13700(
		v_in_pos: Union[float, np.ndarray],
		v_in_neg: Union[float, np.ndarray],
		v_out: Union[float, np.ndarray],
		i_abc: Union[float, np.ndarray],
		v_supply_pos = 12.0,
		v_supply_neg = -12.0) -> Union[float, np.ndarray]:
	"""
	Model of LM13700

	All parameters in volts & amps

	Returns output current

	Note that output voltage must be known already - this is intended to be used with an iterative solver
	"""

	"""
	This model is from Mystran
	https://www.kvraudio.com/forum/viewtopic.php?t=497961&start=15

	Models tanh and output stage nonlinearities

	Does not model individual inputs clipping to voltage rails (e.g. large common-mode voltage)
	Though this scenario shouldn't ever happen in a typical filter, due to input voltage divider
	
	TODO: try out the improvements suggested on KVR (rounder corner on positive rail)
	"""

	tanh_arg = (v_in_pos - v_in_neg)/(2.0*Vt)
	clip_pos_exp_arg = (v_out + V_DROP_POS - v_supply_pos)/Vt
	clip_neg_exp_arg = (v_supply_neg + V_DROP_NEG - v_out)/Vt
	return i_abc*np.tanh(tanh_arg) - Is * np.exp(clip_pos_exp_arg) + Is * np.exp(clip_neg_exp_arg)


class Lm13700OnePolePositive(ZdfOnePoleBase):
	"""
	Simulates LM13700-based OTA pole, using input into positive OTA terminal and feedback into negative OTA terminal
	(Note that buffer does not clip to supply voltages like a real buffer would!)
	"""

	V_SUPPLY = 12

	# Voltage divider
	#D = 560.0 / (68000.0 + 560.0)  # IR3109
	D = 220.0 / (10000.0 + 220.0)  # Mutable Instruments SMR4

	# Capacitance
	#C = 240.0e-12  # IR3109
	C = 1.0e-9  # SMR4

	def __init__(self, wc, iter_stats: Optional[IterStats] = None, verbose=False):
		self.i_abc = None
		super().__init__(wc, iter_stats=iter_stats, use_newton=False)

		if verbose:
			print('LM13700 non-inverting filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def set_freq(self, wc: float) -> None:
		super().set_freq(wc)
		"""
		Solve i_abc from g

		"g" is half of actual circuit gain, to account for trapezoidal integration
			g = D * 0.5 * gm / C

		Linear OTA transconductance equation:
			gm = i_abc / (2 * Vt)

		Putting this all together:
			g = D * 0.5 * i_abc / (2 * Vt * C)
			i_abc = 4 * g * Vt * C / D
		"""
		self.i_abc = 4.0 * self.g * THERMAL_VOLTAGE * self.C / self.D

	def get_estimate(self, x: float) -> float:

		# First, calculate linear estimate
		y = self.m * (self.g*x + self.s)
		#return y  # DEBUG

		# Then, calculate ideal OTA nonlinearity
		tanh_arg = self.D / (2.0 * Vt) * (x + y)
		y = self.s - (0.5 / self.C) * self.i_abc * np.tanh(tanh_arg)
		#return y  # DEBUG

		# Finally, add output stage nonlinearity (but just as a cheap hard clip)
		y = clip(y, (-self.V_SUPPLY + 0.6, self.V_SUPPLY - 2.1))

		return y

	def process_sample_no_state_update(self, x: float, estimate=None, plot_solving_ax=None) -> Tuple[float, float]:
		"""
		:returns: (output, state)
		"""

		"""
		Equations:

		(Here lm13700() is the function above, but with the only arguments being differential gain, and V_ota_out)

			V_in = x
			V_out = y = V_c

			V_ota_neg = D*y
			V_ota_pos = D*x
			V_ota_diff = V_ota_pos - V_ota_neg = D*x - D*y
			
			V_ota_out = V_out = y

			I_ota_out = lm13700(V_ota_diff, V_ota_out)
			          = lm13700(V_ota_pos, V_ota_out)
					  = lm13700(D*x - D*y, V_ota_out)

			V_c = 1 / C * Integral(I_c)
			I_c = I_ota_out

		So, putting this together:

			y = 1 / C * Integral(I_ota_out)
			  = Integral(I_ota_out / C)

			I_ota_out = lm13700(D*x - D*y, y)

		Trapezoidal integration:

		ix							iy
		->( * )-+---> ( + ) ----+---->
		   0.5	|		A		|
				|		| s1	|
				|	  [z-1]		|
				|		A		|
				|       | s0    |
				+---> ( + ) <---+

			iy = s1 + 0.5*ix
			s0 = iy + 0.5*ix

		Where:

			ix = I_ota_out / C
			iy = y

		Putting it together:

			y = s1 + 0.5*I_ota_out/C
			s0 = y + 0.5*I_ota_out/C

			I_ota_out = lm13700(D*x - D*y, y)

		So that gives us the equations:

			y = s1 + 0.5/C * lm13700(D*x - D*y, y)
			s0 = y + 0.5/C * lm13700(D*x - D*y, y)

		To put in terms of f(y)=0:
			0 = s1 - y + 0.5/C * lm13700(D*x - D*y, y)
		
		TODO: try Runge-Kutta 4 instead of trapezoidal
		"""

		i_abc = self.i_abc
		s1 = self.s
		v_supply_pos = self.V_SUPPLY
		v_supply_neg = -self.V_SUPPLY
		C = self.C
		D = self.D

		def f(y: float) -> float:
			return s1 - y + (0.5 / C) * lm13700(
				v_in_pos=D*x,
				v_in_neg=D*y,
				v_out=y,
				i_abc=i_abc,
				v_supply_pos=v_supply_pos,
				v_supply_neg=v_supply_neg)

		def df(y: float) -> float:
			"""
			FIXME: I can't for the life of me get this to work, it gives different results from gradient of f(x)
			I've re-checked my math several times and can't find the mistake

			df(0) is off from gradient method around y=0 by a factor of 1.1-1.9, which seems to depend on s1
			But why would the slope depend on s1? It's a constant term in f(y), it shouldn't affect derivative
			ret *= 1.8 gets the results close enough that at least Newton converges

			Anyway, here's my math that has an error in it somewhere:

			f(y) = s1 - y + 0.5/C * lm13700(D*x - D*y, y)
			f'(y) = -1 + 0.5/C * lm13700'(D*x - D*y, y)

			lm13700() =
				i_abc*np.tanh(tanh_arg) - Is * np.exp(clip_pos_exp_arg) + Is * np.exp(clip_neg_exp_arg)

			lm13700'() =
				tanh_arg' * i_abc * tanh'(tanh_arg)
				- Is * clip_pos_exp_arg' * exp'(clip_pos_exp_arg)
				+ Is * clip_neg_exp_arg' * exp'(clip_neg_exp_arg)

			tanh_arg = (x - y) * D / (2.0*Vt)
			         = x * D / (2.0*Vt) - y * D / (2.0*Vt)
			tanh_arg' = 0 - D / (2.0*Vt)
			          = - D / (2.0*Vt)
			
			clip_pos_exp_arg = (y + V_DROP_POS - v_supply_pos)/Vt
			                 = y/Vt + V_DROP_POS/Vt - v_supply_pos/Vt
			clip_pos_exp_arg' = 1/Vt + 0 + 0
			                  = 1/Vt

			clip_neg_exp_arg = (v_supply_neg + V_DROP_NEG - y)/Vt
			                 = v_supply_neg/Vt + V_DROP_NEG/Vt - y/Vt
			clip_neg_exp_arg' = 0 + 0 - 1/Vt
			                  = -1/Vt

			tanh' = sech^2(x) = 1 / cosh^2(x)
			exp' = exp

			So, lm13700'(D*x - D*y, y) =
				- D * i_abc / (2.0*Vt) * (cosh(tanh_arg) ^ 2)
				- Is / Vt * exp(clip_pos_exp_arg)
				- Is / Vt * exp(clip_neg_exp_arg)
			"""

			tanh_arg = (x - y) * D / (2.0 * Vt)
			clip_pos_exp_arg = (y + V_DROP_POS - v_supply_pos) / Vt
			clip_neg_exp_arg = (v_supply_neg + V_DROP_NEG - y) / Vt

			d_lm13700 = \
				-D * i_abc / (2.0 * Vt) * 1.0 / (np.cosh(tanh_arg) ** 2.0) \
				- (Is / Vt) * np.exp(clip_pos_exp_arg) \
				- (Is / Vt) * np.exp(clip_neg_exp_arg)

			return -1.0 - (0.5 / C) * d_lm13700

		method = 3

		if method == 1:
			# Secant method - succeeds at low gains, fails at high gains (4 succeeds, 25 fails)

			if estimate is None:
				estimate = self.get_estimate(x)

			y = scipy.optimize.newton(func=f, x0=estimate)

		elif method == 2:
			# Newton-Raphson - fails (df problems - see above)

			if estimate is None:
				estimate = self.get_estimate(x)

			y = scipy.optimize.newton(func=f, x0=estimate, fprime=df)

		elif method == 3:
			# Brent's method - always succeeds, if a & b are indeed bracketing

			# TODO: smarter estimates here
			# get_estimate() is probably pretty good in most cases, but we need something guaranteed bracketing

			#a, b = v_supply_neg, v_supply_pos
			#if estimate is None:
			#	estimate = self.get_estimate(x)
			#a, b = estimate - 5, estimate + 5
			#a, b = max(estimate - 5, v_supply_neg), min(estimate + 5, v_supply_pos)
			#a, b = -11.1, 9.5
			a, b = v_supply_neg + V_DROP_NEG, v_supply_pos - V_DROP_POS  # -12 + 1.5 = -10.5, 12 - 3 = 9

			#a, b = estimate - 2, estimate + 2
			#a, b = estimate - 0.1, estimate + 0.1

			try:
				y = scipy.optimize.brentq(f=f, a=a, b=b)
			except ValueError:
				# If previous estimate isn't bracketing
				# Should be guaranteed to converge (it's just slower)
				a, b = v_supply_neg, v_supply_pos
				y = scipy.optimize.brentq(f=f, a=a, b=b)

		elif method == 4:
			# Brent's method, then N-R - fails (df problems)
			# (No point to Brent then Secant - this is already built into Brent)
			a, b = v_supply_neg, v_supply_pos
			y = scipy.optimize.brentq(f=f, a=a, b=b, xtol=1e-3, rtol=1e-3)
			y = scipy.optimize.newton(func=f, x0=y, fprime=df)

		else:
			assert False

		if plot_solving_ax is not None:

			#a, b = v_supply_neg, v_supply_pos  # Blows up
			#a, b = v_supply_neg + V_DROP_NEG, v_supply_pos - V_DROP_POS

			plot_x = np.linspace(a, b, 1024)

			f_prev_y = f(self.prev_y)

			plot_y = f(plot_x)

			if estimate is None:
				estimate = self.get_estimate(x)

			estimate_y = f(estimate)

			plot_solving_ax.plot(plot_x, plot_y, label='Actual')
			plot_solving_ax.plot(y, 0.0, '.', label='Solution')
			plot_solving_ax.plot(estimate, estimate_y, '.', label='Estimate')
			plot_solving_ax.plot(self.prev_y, f_prev_y, '.', label='Previous y')
			#plot_solving_ax.axhline(s1, label=f's1={s1}')
			plot_solving_ax.axhline(self.s, label=f's1={self.s}')

		"""
		State update:
	
		i_ota_out = lm13700(D*x - D*y, y)
		s0 = y + 0.5/C * i_ota_out

		Or, without recalculating lm13700:

		y = s1 + 0.5/C * i_ota_out
		0.5/C * i_ota_out = y - s1
		i_ota_out = (C/0.5) * (y - s1)

		s0 = y + 0.5/C * i_ota_out
		s0 = y + (0.5/C) * (C/0.5) * (y - s1)
		s0 = y + y - s1
		s0 = 2*y - s1
		"""

		state_work_backwards = True
		if state_work_backwards:
			s0 = 2*y - s1
		else:
			s0 = y + 0.5/C * lm13700(
				v_in_pos=D*x,
				v_in_neg=D*y,
				v_out=y,
				i_abc=i_abc,
				v_supply_pos=v_supply_pos,
				v_supply_neg=v_supply_neg,
			)

		return y, s0


#class Lm13700OnePolePositiveRk4(FilterBase):
class Lm13700OnePolePositiveRk4(Rk4OnePoleBase):
	"""
	Simulates LM13700-based OTA pole, using input & feedback into negative OTA terminal (positive terminal grounded)
	(Note that buffer does not clip to supply voltages like a real buffer would!)
	"""
	V_SUPPLY = 12

	# Voltage divider
	#D = 560.0 / (68000.0 + 560.0)  # IR3109
	D = 220.0 / (10000.0 + 220.0)  # Mutable Instruments SMR4

	# Capacitance
	#C = 240.0e-12  # IR3109
	C = 1.0e-9  # SMR4

	def __init__(self, wc):
		self.i_abc = None
		super().__init__(wc)

	def set_freq(self, wc: float) -> None:
		super().set_freq(wc)
		self.i_abc = 2.0 * self.g * THERMAL_VOLTAGE * self.C / self.D

	def dydt(self, xt: float, yt: float) -> float:
		# FIXME: this blows up when it clips
		# Clipping either yt or result to (-V_SUPPLY, V_SUPPLY) stops that, but introduces other problems
		return 1.0 / self.C * lm13700(
			v_in_pos=self.D*xt,
			v_in_neg=self.D*yt,
			v_out=yt,
			i_abc=self.i_abc,
			v_supply_pos=self.V_SUPPLY,
			v_supply_neg=-self.V_SUPPLY)


class Lm13700OnePoleInverting(ZdfOnePoleBase):
	"""
	Simulates LM13700-based OTA pole, using input & feedback into negative OTA terminal (positive terminal grounded)
	(Note that buffer does not clip to supply voltages like a real buffer would!)
	"""

	V_SUPPLY = 12

	# Voltage divider
	#D = 560.0 / (68000.0 + 560.0)  # IR3109
	D = 220.0 / (10000.0 + 220.0)  # Mutable Instruments SMR4

	# Capacitance
	#C = 240.0e-12  # IR3109
	C = 1.0e-9  # SMR4

	def __init__(self, wc, iter_stats: Optional[IterStats] = None, verbose=False):
		self.i_abc = None
		super().__init__(wc, iter_stats=iter_stats, use_newton=False)

		if verbose:
			print('LM13700 non-inverting filter: wc=%f, actual g=%f, approx g=%f' % (wc, self.g, pi*wc))

	def set_freq(self, wc: float) -> None:
		super().set_freq(wc)
		self.i_abc = 4.0 * self.g * THERMAL_VOLTAGE * self.C / self.D

	def get_estimate(self, x: float) -> float:

		# First, calculate linear estimate (inverted)
		y = -self.m * (self.g*x + self.s)

		# Then, calculate ideal OTA nonlinearity
		tanh_arg = self.D / (2.0 * Vt) * (x + y)
		y = self.s - (0.5 / self.C) * self.i_abc * np.tanh(tanh_arg)

		# Add output stage nonlinearity (but just as a cheap hard clip)
		y = clip(y, (-self.V_SUPPLY + 0.6, self.V_SUPPLY - 2.1))

		return y

	def process_sample_no_state_update(self, x: float, estimate=None, plot_solving_ax=None) -> Tuple[float, Any]:
		"""
		:returns: (output, state)
		"""

		"""
		Equations:

		(Here lm13700() is the function above, but with the only arguments being differential gain, and V_ota_out)

			V_in = x
			V_out = y = V_c

			V_ota_neg = D*x + D*y
			V_ota_pos = 0
			V_ota_diff = V_ota_pos - V_ota_neg = 0 - (D*x + D*y) = -D*x - D*y
			
			V_ota_out = V_out = y

			I_ota_out = lm13700(V_ota_diff, V_ota_out)
			          = lm13700(V_ota_pos, V_ota_out)
					  = lm13700(-D*x - D*y), V_ota_out)

			V_c = 1 / C * Integral(I_c)
			I_c = I_ota_out

		So, putting this together:

			y = 1 / C * Integral(I_ota_out)
			  = Integral(I_ota_out / C)

			I_ota_out = lm13700(-D*x - D*y, y)

		Trapezoidal integration:

			y = s1 + 0.5*I_ota_out/C
			s0 = y + 0.5*I_ota_out/C

			I_ota_out = lm13700(-D*x - D*y, y)

		So that gives us the equations:

			y = s1 + 0.5/C * lm13700(-D*x - D*y, y)
			s0 = y + 0.5/C * lm13700(-D*x - D*y, y)

		To put in terms of f(y)=0:
			0 = s1 - y + 0.5/C * lm13700(-D*x - D*y, y)
		
		TODO: try Runge-Kutta 4 here as well as above
		"""

		i_abc = self.i_abc
		s1 = self.s
		v_supply_pos = self.V_SUPPLY
		v_supply_neg = -self.V_SUPPLY
		C = self.C
		D = self.D

		def f(y: float) -> float:
			return s1 - y + (0.5 / C) * lm13700(
				v_in_pos=0,
				v_in_neg=D*(x + y),
				v_out=y,
				i_abc=i_abc,
				v_supply_pos=v_supply_pos,
				v_supply_neg=v_supply_neg)

		method = 2

		if method == 1:
			# Secant method
			# Seems to succeed, even though non-inverting didn't - not sure why

			if estimate is None:
				estimate = self.get_estimate(x)

			y = scipy.optimize.newton(func=f, x0=estimate)

		elif method == 2:
			# Brent's method - always succeeds, if a & b are indeed bracketing
			a, b = v_supply_neg + V_DROP_POS, v_supply_pos - V_DROP_NEG  # -12 + 3 = -9, 12 - 1.5 = 10.5
			try:
				y = scipy.optimize.brentq(f=f, a=a, b=b)
			except ValueError:
				# If previous estimate isn't bracketing
				# Should be guaranteed to converge (it's just slower)
				a, b = v_supply_neg, v_supply_pos
				y = scipy.optimize.brentq(f=f, a=a, b=b)

		else:
			assert False

		if plot_solving_ax is not None:

			#a, b = v_supply_neg, v_supply_pos  # Blows up
			#a, b = v_supply_neg + V_DROP_NEG, v_supply_pos - V_DROP_POS

			plot_x = np.linspace(a, b, 1024)

			f_prev_y = f(self.prev_y)

			plot_y = f(plot_x)

			if estimate is None:
				estimate = self.get_estimate(x)

			estimate_y = f(estimate)

			plot_solving_ax.plot(plot_x, plot_y, label='Actual')
			plot_solving_ax.plot(y, 0.0, '.', label='Solution')
			plot_solving_ax.plot(estimate, estimate_y, '.', label='Estimate')
			plot_solving_ax.plot(self.prev_y, f_prev_y, '.', label='Previous y')
			#plot_solving_ax.axhline(s1, label=f's1={s1}')
			plot_solving_ax.axhline(self.s, label=f's1={self.s}')

		state_work_backwards = True
		if state_work_backwards:
			s0 = 2*y - s1
		else:
			s0 = y + 0.5/C * lm13700(
				v_in_pos=0,
				v_in_neg=D*(x + y),
				v_out=y,
				i_abc=i_abc,
				v_supply_pos=v_supply_pos,
				v_supply_neg=v_supply_neg,
			)

		return y, s0


def plot_open_loop():

	"""

	i_out = i_abc * tanh(v_in_diff/(2*Vt))
		- Is * exp((v_out + V_DROP_POS - v_supply_pos)/Vt)
		+ Is * exp((v_supply_neg + V_DROP_NEG - v_out)/Vt)

	Or, reducing this:

	v_in_diff = v_in_pos - v_in_neg

	tanh_arg = v_in_diff / (2.0*Vt)
	clip_pos_exp_arg = (v_out + V_DROP_POS - v_supply_pos) / Vt
	clip_neg_exp_arg = (v_supply_neg + V_DROP_NEG - v_out) / Vt

	i_out = i_abc * tanh(tanh_arg) - Is * exp(clip_pos_exp_arg) + Is * exp(clip_neg_exp_arg)

	v_out / R = i_abc * tanh(tanh_arg) - Is * exp(clip_pos_exp_arg) + Is * exp(clip_neg_exp_arg)

	i_abc * tanh(v_in_diff/(2*Vt)) =
		+ Is * exp((v_out + V_DROP_POS - v_supply_pos)/Vt)
		- Is * exp((v_supply_neg + V_DROP_NEG - v_out)/Vt)
		+ v_out / R

	i_abc * tanh(tanh_arg) = Is * exp(clip_pos_exp_arg) - Is * exp(clip_neg_exp_arg) + v_out / R

	tanh(v_in_diff/(2*Vt)) =
		Is / i_abc * exp((v_out + V_DROP_POS - v_supply_pos)/Vt)
		- Is / i_abc * exp((v_supply_neg + V_DROP_NEG - v_out)/Vt)
		+ v_out / (R * i_abc)

	tanh(tanh_arg) = Is / i_abc * exp(clip_pos_exp_arg) - Is / I_abc * exp(clip_neg_exp_arg) + v_out / (R * I_abc)

	For reverse solving:

	tanh_arg = (v_in_diff) / (2.0*Vt) = atanh(
		Is / i_abc * exp(clip_pos_exp_arg)
		- Is / i_abc * exp(clip_neg_exp_arg)
		+ v_out / (R * i_abc)
	) 

	v_in_diff = 2 * Vt * atanh(
		Is / i_abc * exp(clip_pos_exp_arg)
		- Is / i_abc * exp(clip_neg_exp_arg)
		+ v_out / (R * i_abc)
	)

	For forward solving:

	tanh(tanh_arg) = Is / i_abc * exp(clip_pos_exp_arg) - Is / I_abc * exp(clip_neg_exp_arg) + v_out / (R * I_abc)
	0 = Is / i_abc * exp(clip_pos_exp_arg) - Is / I_abc * exp(clip_neg_exp_arg) + v_out / (R * I_abc) - tanh(tanh_arg)
	0 = Is * exp(clip_pos_exp_arg) - Is * exp(clip_neg_exp_arg) + v_out / R - i_abc * tanh(tanh_arg)

	tanh_arg = v_in_diff / (2.0*Vt)
	clip_pos_exp_arg = (v_out + V_DROP_POS - v_supply_pos) / Vt
	clip_neg_exp_arg = (v_supply_neg + V_DROP_NEG - v_out) / Vt

	0 = Is * exp(clip_pos_exp_arg) - Is * exp(clip_neg_exp_arg) + v_out / R - i_abc * tanh(tanh_arg)
	"""


	def solve_inverse(i_abc: float, r_load: float, v_supply: float) -> Tuple[np.ndarray, np.ndarray]:
		
		v_supply_pos = v_supply
		v_supply_neg = -v_supply

		v_out = np.linspace(v_supply_neg, v_supply_pos, 1024)

		arctanh_arg = \
			Is / i_abc * np.exp((v_out + V_DROP_POS - v_supply_pos)/Vt) \
			- Is / i_abc * np.exp((v_supply_neg + V_DROP_NEG - v_out)/Vt) \
			+ v_out / (r_load * i_abc)

		arctanh_arg[arctanh_arg >= 1.0] = np.nan
		arctanh_arg[arctanh_arg <= -1.0] = np.nan

		v_in = 2 * Vt * np.arctanh(arctanh_arg)

		return v_in, v_out


	def solve_iterative(
			v_in: np.ndarray,
			i_abc: float,
			r_load: float,
			v_supply: float,
			use_brent=False,
			return_num_iter=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

		v_supply_pos = v_supply
		v_supply_neg = -v_supply

		v_out = np.zeros_like(v_in)

		num_iter = np.zeros(len(v_in), dtype=np.uint) if return_num_iter else None

		def f(y: float, x: float) -> float:
			return lm13700(
				v_in_pos=x,
				v_in_neg=0.0,
				v_out=y,
				i_abc=i_abc,
				v_supply_pos=v_supply,
				v_supply_neg=-v_supply
			) - y / r_load

		def df(y: float, x: float) -> float:

			"""

			tanh_arg = (v_in_pos - v_in_neg)/(2.0*Vt)
			clip_pos_exp_arg = (v_out + V_DROP_POS - v_supply_pos)/Vt
			clip_neg_exp_arg = (v_supply_neg + V_DROP_NEG - v_out)/Vt

			i_out = i_abc*np.tanh(tanh_arg) - Is * np.exp(clip_pos_exp_arg) + Is * np.exp(clip_neg_exp_arg)

			d/dy lm13700() - y/r_load =
				d/dy tanh_arg(y) * dtanh(tanh_arg)
				- d/dy clip_pos_exp_arg(y) * Is * np.exp(clip_pos_exp_arg)
				+ d/dy clip_neg_exp_arg(y) * Is * np.exp(clip_neg_exp_arg)
				- d/dy y / r_load

			d/dy tanh_arg(y) = 0
			d/dy clip_pos_exp_arg(y) = 1 / Vt
			d/dy clip_neg_exp_arg(y) = -1 / Vt

			tanh term is conveniently 0

			so d/dy lm13700() - y/r_load =
				- Is / Vt * np.exp(clip_pos_exp_arg)
				- Is / Vt * np.exp(clip_neg_exp_arg)
				- 1.0 / r_load
			"""

			clip_pos_exp_arg = (y + V_DROP_POS - v_supply_pos)/Vt
			clip_neg_exp_arg = (v_supply_neg + V_DROP_NEG - y)/Vt

			return 0 \
				- Is / Vt * np.exp(clip_pos_exp_arg) \
				- Is / Vt * np.exp(clip_neg_exp_arg) \
				- 1.0 / r_load

		for n, x in enumerate(v_in):

			if use_brent:
				a, b = (0.0, v_supply - 2.0) if (x >= 0.0) else (-v_supply + 0.5, 0.0)
				solver_ret = scipy.optimize.brentq(f=f, a=a, b=b, args=(x,), full_output=return_num_iter)
			else:
				i_estimate = i_abc * tanh(x / (2.0 * THERMAL_VOLTAGE))
				v_estimate = clip(i_estimate * r_load, (-v_supply + 0.6, v_supply - 2.1))
				solver_ret = scipy.optimize.newton(func=f, x0=v_estimate, fprime=df, args=(x,), full_output=return_num_iter)

			y, r = solver_ret if return_num_iter else (solver_ret, None)

			v_out[n] = y
			if return_num_iter:
				#num_iter[n] = r.iterations
				num_iter[n] = r.function_calls

		if return_num_iter:
			return v_out, num_iter
		else:
			return v_out

	#fig = plt.figure()
	fig, ax_v_out = plt.subplots(1, 1)
	fig.suptitle('LM13700 OTA model - solved inverse')

	fig, ax_i_out = plt.subplots(1, 1)
	fig.suptitle('LM13700 OTA model - solved inverse')

	fig, ax_output_nonlin = plt.subplots(1, 1)
	fig.suptitle('LM13700 OTA model - solved inverse')

	for i_abc, r_load, v_supply in [
			(0.5e-3, 1.0e3, 12.0),
			(1.0e-3, 1.0e3, 12.0),
			(1.0e-3, 5.0e3, 12.0),
			(1.0e-3, 10.0e3, 12.0),
			(1.0e-3, 10.0e3, 4.5),
			(1.0e-3, 20.0e3, 12.0),
			(1.0e-3, 50.0e3, 12.0),
			(0.5e-3, 100.0e3, 12.0),
			(0.2e-3, 100.0e3, 12.0),
			(1.0e-3, 100.0e3, 12.0),
			(0.1e-3, 1.0e6, 12.0),
			(0.01e-3, 10.0e6, 12.0),
			]:

		v_in, v_out = solve_inverse(i_abc=i_abc, r_load=r_load, v_supply=v_supply)
		label = 'I_abc = %g mA, R_load = %g k, V_supply = +/- %g' % (i_abc*1000, r_load/1000, v_supply)
		ax_v_out.plot(v_in, v_out, label=label)

		i_out = v_out / r_load
		ax_i_out.plot(v_in, i_out * 1000.0, label=label)

		v_out_ideal = i_abc * np.tanh(v_in / (2.0*Vt)) * r_load
		ax_output_nonlin.plot(v_out_ideal, v_out, label=label)

	for ax in [ax_v_out, ax_i_out, ax_output_nonlin]:
		ax.legend()
		ax.grid()

	ax_v_out.set_title('Voltage transfer function')
	ax_v_out.set_xlabel('V_in (V)')
	ax_v_out.set_ylabel('V_out (V)')

	ax_i_out.set_title('Current transfer function')
	ax_i_out.set_xlabel('V_in (V)')
	ax_i_out.set_ylabel('I_out (mA)')

	ax_output_nonlin.set_title('Output stage transfer function')
	ax_output_nonlin.set_xlabel('Ideal OTA V_out (V)')
	ax_output_nonlin.set_ylabel('Actual V_out (V)')

	i_abc = 1.0e-3
	r_load = 15.0e3
	v_supply = 12.0

	fig, (data_ax, iter_ax) = plt.subplots(2, 1)
	fig.suptitle('LM13700 OTA model - solved iteratively')

	v_in, v_out = solve_inverse(i_abc=i_abc, r_load=r_load, v_supply=v_supply)
	data_ax.plot(v_in, v_out, label='Inverse')
	next(iter_ax._get_lines.prop_cycler)
	v_in = np.linspace(-0.1, 0.1, 1024)
	v_out, num_iter = solve_iterative(v_in=v_in, i_abc=i_abc, r_load=r_load, v_supply=v_supply, use_brent=False, return_num_iter=True)
	data_ax.plot(v_in, v_out, label='Newton-Raphson')
	iter_ax.plot(v_in, num_iter, label='Newton-Raphson')
	v_out, num_iter = solve_iterative(v_in=v_in, i_abc=i_abc, r_load=r_load, v_supply=v_supply, use_brent=True, return_num_iter=True)
	data_ax.plot(v_in, v_out, label='Brent')
	iter_ax.plot(v_in, num_iter, label='Brent')

	data_ax.set_title('I_abc = %g mA, R_load = %g k, V_supply = +/- %g' % (i_abc*1000, r_load/1000, v_supply))
	data_ax.legend()
	data_ax.set_xlabel('V_in')
	data_ax.set_ylabel('V_out')
	data_ax.grid()

	iter_ax.legend()
	iter_ax.set_xlabel('V_in')
	iter_ax.set_ylabel('Number of function calls')
	iter_ax.grid()


def do_fft(x, n_fft, window=False):

	if window:
		y = np.fft.fft(x * np.hamming(len(x)), n=n_fft)
	else:
		y = np.fft.fft(x, n=n_fft)

	f = np.fft.fftfreq(n_fft, 1.0)

	# Only take first half
	y = y[0:len(y)//2]
	f = f[0:len(f)//2]

	y = to_dB(np.abs(y))

	return y, f


def plot_one_pole(fc=0.1, f_saw=0.01, gain=4.0, n_samp=2048):

	iter_stats_positive = IterStats('LM13700 Positive')
	filt_positive = Lm13700OnePolePositive(fc, iter_stats=iter_stats_positive)

	filt_positive_rk4 = Lm13700OnePolePositiveRk4(fc)

	iter_stats_negative = IterStats('LM13700 Inverting')
	filt_negative = Lm13700OnePoleInverting(fc, iter_stats=iter_stats_negative)

	fig, (time_plot, freq_plot) = plt.subplots(2, 1)
	fig.suptitle(f'LM13700 one-pole filter, {fc=:g}, {f_saw=:g}, {gain=:g}')

	t = np.arange(n_samp)
	x = gen_saw(f_saw, n_samp, start_phase=0.5) * gain * 0.5

	fft_x, f = do_fft(x, n_fft=n_samp, window=True)

	time_plot.plot(t, x, label='Input')
	freq_plot.semilogx(f, fft_x, label='Input')

	y_pos = filt_positive.process_vector(x)
	y_pos_rk4 = filt_positive_rk4.process_vector(x)
	y_neg = filt_negative.process_vector(x)

	fft_y_pos, f = do_fft(y_pos, n_fft=n_samp, window=True)
	fft_y_pos_rk4, f = do_fft(y_pos_rk4, n_fft=n_samp, window=True)
	fft_y_neg, f = do_fft(y_neg, n_fft=n_samp, window=True)
	time_plot.plot(t, y_pos, label='Non-inverting')
	time_plot.plot(t, y_pos_rk4, label='Non-inverting, RK4')
	time_plot.plot(t, -y_neg, label='-1 * inverting')
	freq_plot.semilogx(f, fft_y_pos, label='Non-inverting')
	freq_plot.semilogx(f, fft_y_pos_rk4, label='Non-inverting, RK4')
	freq_plot.semilogx(f, fft_y_neg, label='-1 * inverting')

	time_plot.legend()
	time_plot.set_xlim([0, 256])
	time_plot.grid()

	freq_plot.grid()
	
	#iter_stats.output()


def plot_solving_func(fc=0.1, f_saw=0.01, gain=1.0):
	filt = Lm13700OnePolePositive(fc)
	n_samp = 50
	x = gen_saw(f_saw, n_samp, start_phase=0.5) * gain * 0.5

	fig, subplots = plt.subplots(2, 3)
	fig.suptitle(f'Solver, {fc=:g}, {f_saw=:g}, {gain=:g}')

	for idx, x_sample in enumerate(x):

		if idx == 0:
			ax = subplots[0][0]
		elif idx == 1:
			ax = subplots[0][1]
		elif idx == 2:
			ax = subplots[0][2]
		elif idx == n_samp - 3:
			ax = subplots[1][0]
		elif idx == n_samp - 2:
			ax = subplots[1][1]
		elif idx == n_samp - 1:
			ax = subplots[1][2]
		else:
			ax = None

		_, s = filt.process_sample_no_state_update(x=x_sample, plot_solving_ax=ax)
		filt.s = s

	subplots[0][0].set_title('n=0')
	subplots[0][1].set_title('n=1')
	subplots[0][2].set_title('n=2')
	subplots[1][0].set_title(f'n={n_samp-3}')
	subplots[1][1].set_title(f'n={n_samp-2}')
	subplots[1][2].set_title(f'n={n_samp-1}')

	for row in subplots:
		for ax in row:
			ax.grid()
			ax.legend()


def plot(args=None):
	plot_open_loop()
	plot_one_pole(gain=0.1)
	plot_one_pole(gain=1.0)
	plot_one_pole(gain=4.0)
	plot_one_pole(gain=25.0)
	plot_one_pole(gain=100.0)
	plot_solving_func(gain=1.0)
	plot_solving_func(gain=100.0)
	plt.show()


def main(args=None):
	plot(args=args)
