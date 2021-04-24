#!/usr/bin/env python3

"""
Fast approximations for various trig functions
"""

from approx.cheby import cheby_poly, cheby_fit
from utils import utils

import math
from matplotlib import pyplot as plt
import numpy as np
from typing import Callable, Union, Optional, Tuple


PI = math.pi
HALF_PI = 0.5 * PI
TWO_PI = 2.0 * PI


#
# Utility functions
#


@np.vectorize
def _quadrant_1_4_wrapper(x: float, f: Callable, is_sine: bool) -> float:

	x = abs(x) % TWO_PI

	quadrant = int(math.floor(x / HALF_PI))

	return {
		0: f(x),

		# FIXME: figure out the actual math problem here instead of this hack
		1: f(math.pi - x) if is_sine else -f(math.pi - x),

		2: -f(x - math.pi),
		3: f(x - TWO_PI),
	}[quadrant]


@np.vectorize
def _quadrant_1_2_wrapper(x: float, f: Callable, is_sine: bool) -> float:
	x = abs(x) % TWO_PI

	quadrant_3_4 = bool(int(math.floor(x / PI)))

	if is_sine:
		return -f(x - PI) if quadrant_3_4 else f(x)
	else:
		raise NotImplementedError


@np.vectorize
def _quadrant_1_wrapper(x: float, f: Callable, is_sine: bool) -> float:

	x = abs(x) % TWO_PI

	quadrant = int(math.floor(x / HALF_PI))
	#return quadrant  # DEBUG

	if is_sine:
		raise NotImplementedError
		# TESTME
		return {
			0: f(x),
			1: f(HALF_PI - x),
			2: -f(x - PI),
			3: -f(PI - x),
		}[quadrant]

	else:
		return {
			0: f(x),
			1: -f(PI - x),
			2: -f(x - PI),
			3: f(TWO_PI - x),
		}[quadrant]


def _sin_from_cos(f_cos: Callable, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return f_cos(x - HALF_PI)


def _cos_from_sin(f_sin: Callable, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return f_sin(x + HALF_PI)


#
# Small-angle approximations
#


def cos_small_angle(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return 1.0 - 0.5 * np.square(x)


def sin_small_angle(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return x


#
# Taylor/Maclaurin approximations
#

# (not great, but just for comparison)


def sin_maclaurin_3s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return x - (x ** 3) / 6


def cos_maclaurin_4s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return 1.0 - (x ** 2) / 2 + (x ** 4) / 24


def sin_maclaurin_5s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return x - (x ** 3) / 6 + (x ** 5) / 120


def cos_maclaurin_6s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return 1.0 - (x ** 2) / 2 + (x ** 4) / 24 - (x ** 6) / 720


def sin_maclaurin_3(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_4_wrapper(x, sin_maclaurin_3s, True)


def cos_maclaurin_4(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_4_wrapper(x, cos_maclaurin_4s, False)


def sin_maclaurin_5(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_4_wrapper(x, sin_maclaurin_5s, True)


def cos_maclaurin_6(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_4_wrapper(x, cos_maclaurin_6s, False)


#
# Bhaskara I's sine approximation formula
#


def sin_bhaskara_positive(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = np.rad2deg(x)
	return (4 * x * (180 - x)) / (40500 - x * (180 - x))


def sin_bhaskara(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_2_wrapper(x, sin_bhaskara_positive, True)


#
# Chebyshev approximations (see cheby.py)
#


def cos_cheb2_evenodd_q1(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = utils.scale(x, (0, HALF_PI), (-1, 1))
	return \
		0.705741045518510246026267 + \
		-0.513625166679106959222167 * x + \
		-0.207092688525927492992906 * (x ** 2)


def cos_cheb2_evenodd(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_wrapper(x, cos_cheb2_evenodd_q1, is_sine=False)


def cos_cheb3_evenodd_q1(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = utils.scale(x, (0, HALF_PI), (-1, 1))
	return \
		0.705741045518510246026267 + \
		-0.554821269382182347129628 * x + \
		-0.207092688525927492992906 * (x ** 2) + \
		0.054928136937433862108104 * (x ** 3)


def cos_cheb3_evenodd(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_wrapper(x, cos_cheb3_evenodd_q1, is_sine=False)


def cos_cheb4_evenodd_q1(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = utils.scale(x, (0, HALF_PI), (-1, 1))
	return \
		0.707099715356600544424737 + \
		-0.554821269382182347129628 * x + \
		-0.217962047230650046714118 * (x ** 2) + \
		0.054928136937433862108104 * (x ** 3) + \
		0.010869358704722560660105 * (x ** 4)


def cos_cheb4_evenodd(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_wrapper(x, cos_cheb4_evenodd_q1, is_sine=False)


def cos_cheb5_evenodd_q1(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = utils.scale(x, (0, HALF_PI), (-1, 1))
	return \
		0.707099715356600544424737 + \
		-0.555357584854211272507030 * x + \
		-0.217962047230650046714118 * (x ** 2) + \
		0.057073398825549619128861 * (x ** 3) + \
		0.010869358704722560660105 * (x ** 4) + \
		-0.001716209510492606527335 * (x ** 5)


def cos_cheb5_evenodd(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_wrapper(x, cos_cheb5_evenodd_q1, is_sine=False)


def sin_cheb3s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = x * 2 / math.pi
	return \
		1.547863507573323804678012 * x + \
		-0.552287106348768208619049 * (x ** 3)


def sin_cheb5s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = x * 2 / math.pi
	return \
		1.570317078806098720633599 * x + \
		-0.642101391279867650396795 * (x ** 3) + \
		0.071851427944879600606676 * (x ** 5)


def cos_cheb2s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = x * 2 / math.pi
	return \
		0.971404474038641829736207 + \
		-0.998806516540814204319076 * (x ** 2)


def cos_cheb4s(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = x * 2 / math.pi
	return \
		0.999396553656189623460193 + \
		-1.222743153481196776155571 * (x ** 2) + \
		0.223936636940382599592070 * (x ** 4)


def sin_cheb2(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _sin_from_cos(cos_cheb2, x)


def sin_cheb3(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_4_wrapper(x, sin_cheb3s, True)


def sin_cheb4(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _sin_from_cos(cos_cheb4, x)


def sin_cheb5(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_4_wrapper(x, sin_cheb5s, True)


def cos_cheb2(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_4_wrapper(x, cos_cheb2s, False)


def cos_cheb3(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _cos_from_sin(sin_cheb3, x)


def cos_cheb4(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _quadrant_1_4_wrapper(x, cos_cheb4s, False)


def cos_cheb5(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return _cos_from_sin(sin_cheb5, x)


def cos_cheb4_branchfree(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = x % TWO_PI
	x = x / math.pi - 1
	return \
		-0.969474842882559806334086 + \
		4.364528972635080883435421 * (x ** 2) + \
		-2.422793242109031908171346 * (x ** 4)


def cos_cheb6_branchfree(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = x % TWO_PI
	x = x / math.pi - 1
	return \
		-0.998566776961081203900505 + \
		4.888183786048466927809386 * (x ** 2) + \
		-3.819206077878061655894726 * (x ** 4) + \
		0.930941890512686387459951 * (x ** 6)


def cos_cheb8_branchfree(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	x = x % TWO_PI
	x = x / math.pi - 1
	return \
		-0.999959031778776208376769 + \
		4.932735940214705294692976 * (x ** 2) + \
		-4.041966848709254378491096 * (x ** 4) + \
		1.287359123842594543773998 * (x ** 6) + \
		-0.178208616664954105912599 * (x ** 8)


def sinc_lobe_cheb2(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return \
		0.924409333585792136744885 + \
		-0.990956771464502783608452 * (x ** 2)


def sinc_lobe_cheb4(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return \
		0.995315379142853617899789 + \
		-1.558205135920994299780773 * (x ** 2) + \
		0.567248364456491516172321 * (x ** 4)


def sinc_lobe_cheb6(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return \
		0.999833206861104328844192 + \
		-1.639526034849507984958450 * (x ** 2) + \
		0.784104094932527750927420 * (x ** 4) + \
		-0.144570486984024165755258 * (x ** 6)


def sinc_lobe_cheb8(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return \
		0.999996152743521826700146 + \
		-1.644740303086868360438189 * (x ** 2) + \
		0.810175436119329850370718 * (x ** 4) + \
		-0.186284632882907530415650 * (x ** 6) + \
		0.020857072949441682330196 * (x ** 8)


#
# Main functions
#


def _plot_func(f: Callable, name: str, range: Optional[Tuple[float, float]]=None, f_actual: Optional[Callable]=None, plot_x_as_pi=False):

	if range is not None:
		x = np.linspace(range[0], range[1], num=256)

	else:
		x = np.linspace(0, TWO_PI, num=1024, endpoint=False)

	f_zero = f(0)  # Falcon kick!
	f_half_pi = f(HALF_PI)
	f_pi = f(PI)

	if f_actual is None:
		if math.isclose(f_zero, 1) and math.isclose(f_pi, 0):
			f_actual = np.sinc
		elif f_zero < f_half_pi:
			f_actual = np.sin
		else:
			f_actual = np.cos

	y_approx = f(x)
	y_actual = f_actual(x)

	err = np.abs(y_approx - y_actual)

	# TODO: plot derivative & its error, 4th derivative & its error

	ffty = np.fft.fft(y_approx)
	ffty /= len(ffty)
	ffty = np.abs(ffty)
	ffty = ffty[:len(ffty) // 2]
	ffty_dB = utils.to_dB(ffty, -200)

	thdn = sum(ffty[2:])
	thdn_dB = 20*np.log10(thdn) - ffty_dB[1]

	max_harmonic_dB = np.amax(ffty_dB[2:]) - ffty_dB[1]

	max_err = np.amax(err)

	if range is None:
		print('%20s %12g %12f %12g %12g %20g %12g' % (
			name,
			max_err,
			20 * np.log10(max_err),
			abs(f(0) - f_actual(0)),
			abs(f(HALF_PI) - f_actual(HALF_PI)),
			max_harmonic_dB,
			thdn_dB,
		))
	else:
		print('%20s %12g %12f %12g %12g' % (
			name,
			max_err,
			20 * np.log10(max_err),
			abs(f(0) - f_actual(0)),
			abs(f(HALF_PI) - f_actual(HALF_PI))
		))

	if plot_x_as_pi:
		x /= PI

	xlabel = 'x / pi' if plot_x_as_pi else 'x'

	fig = plt.figure()
	fig.suptitle(name)

	num_plots = 3 if range is None else 2

	plt.subplot(num_plots, 1, 1)
	plt.plot(x, y_actual, label='Exact')
	plt.plot(x, y_approx, label=name)
	plt.legend()
	plt.grid()
	plt.xlabel(xlabel)

	plt.subplot(num_plots, 1, 2)
	plt.plot(x, err, 'r')
	plt.grid()
	plt.ylabel('Error')
	plt.xlabel(xlabel)

	# TODO: others

	if range is None:
		plt.subplot(num_plots, 1, num_plots)
		plt.plot(ffty_dB[:64])
		plt.grid()
		plt.ylabel('FFT')
		plt.ylim([-160, 0])


def plot(args):
	main(args)


def main(args):

	print('%20s %12s %12s %12s %12s %20s %12s' % (
		'function', 'Max err', 'Max err (dB)', 'err f(0)', 'err f(pi/2)', 'max harmonic (dB)', 'THD (dB)'))

	for f_name, f in [
		('cos_small_angle', cos_small_angle),
		('sin_small_angle', sin_small_angle),
	]:
		_plot_func(f, name=f_name, range=[-PI/4, PI/4], plot_x_as_pi=True)

	for f_name, f in [
		('sin_maclaurin_3', sin_maclaurin_3),
		('cos_maclaurin_4', cos_maclaurin_4),
		('sin_maclaurin_5', sin_maclaurin_5),
		('cos_maclaurin_6', cos_maclaurin_6),
		('sin_bhaskara', sin_bhaskara),
		('cos_cheb2', cos_cheb2),
		('sin_cheb3', sin_cheb3),
		('cos_cheb4', cos_cheb4),
		('sin_cheb5', sin_cheb5),
		('cos_cheb2_evenodd', cos_cheb2_evenodd),
		('cos_cheb3_evenodd', cos_cheb3_evenodd),
		('cos_cheb4_evenodd', cos_cheb4_evenodd),
		('cos_cheb5_evenodd', cos_cheb5_evenodd),
		('cos_cheb4_branchfree', cos_cheb4_branchfree),
		('cos_cheb6_branchfree', cos_cheb6_branchfree),
		('cos_cheb8_branchfree', cos_cheb8_branchfree),
	]:
		_plot_func(f, name=f_name, plot_x_as_pi=True)

	_plot_func(sinc_lobe_cheb4, name='sinc_lobe_cheb4', range=[-1, 1], f_actual=np.sinc)
	_plot_func(sinc_lobe_cheb6, name='sinc_lobe_cheb6', range=[-1, 1], f_actual=np.sinc)

	plt.show()
