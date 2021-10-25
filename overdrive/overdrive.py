#!/usr/bin/env python3

"""
Various overdrive functions
"""


import numpy as np
from typing import Union
from utils import utils
from processor import StatelessProcessorBase
import math


def clip(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""
	Hard clipping at [-1, 1]
	"""
	return utils.clip(vals, (-1., 1.))


class Clipper(StatelessProcessorBase):
	def __init__(self, gain=1.0):
		self.gain = gain

	def process_sample(self, x):
		return clip(x * self.gain)

	def process_vector(self, x):
		return clip(x * self.gain)


def tanh(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""
	tanh overdrive
	output limited to [-1, 1]
	"""
	return np.tanh(vals)


class TanhProcessor(StatelessProcessorBase):
	def __init__(self, gain=1.0):
		self.gain = gain

	def process_sample(self, x):
		return tanh(x * self.gain)

	def process_vector(self, x):
		return tanh(x * self.gain)


def atan(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""
	arctan overdrive
	output not bound
	"""
	return np.arctan(vals)


class AtanProcessor(StatelessProcessorBase):
	def __init__(self, gain=1.0):
		self.gain = gain

	def process_sample(self, x):
		return atan(x * self.gain)

	def process_vector(self, x):
		return atan(x * self.gain)


def sigmoid(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	"""
	x / (1 + |x|) sigmoid
	even softer clipping than tanh
	output limited to [-1, 1]
	"""
	return vals / (1.0 + np.abs(vals))


class Sigmoid(StatelessProcessorBase):
	def __init__(self, gain=1.0):
		self.gain = gain

	def process_sample(self, x):
		return sigmoid(x * self.gain)

	def process_vector(self, x):
		return sigmoid(x * self.gain)


def ln_drive(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return utils.sgn(vals) * np.log(np.abs(vals) + 1.0)


def sqrt_drive(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	return utils.sgn(vals) * 2.0 * (np.sqrt(np.abs(vals) + 1.0) - 1.0)


def quadratic_drive(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	vals = np.clip(vals * 1.5, -3, 3)
	return utils.sgn(vals) * (np.abs(vals) - np.square(vals) / 6.0) / 1.5


def cubic_drive(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	scale = 4 * math.sqrt(3) / 9
	clip_val = 2.0 / math.sqrt(3.0)
	vals = np.clip(vals * scale, -clip_val, clip_val)
	return -0.25 * vals * (vals - 2.0) * (vals + 2.0) / scale


def three_halfs_drive(vals: Union[float, np.ndarray], bias=1., clip_top=None) -> Union[float, np.ndarray]:
	"""
	y = x^(3/2) overdrive
	characteristic of a triode tube (although this is only the most basic of tube models)
	Output bound to [-bias, clip_top]
	Note that this is asymmetric, so this may add some DC into the output signal
	Tuned so that small-signal gain will be unity and small signals will have no DC offset

	:param vals: input value(s)
	:param bias: y bias; output negative side will be clipped to this
	:param clip_top: optional; hard clip the top of the waveform, equivalent to the clipping from limited supply voltage
	"""

	if bias <= 0.:
		raise ValueError('Bias must be positive!')

	"""
	y = (A*x + B)^(3/2) - C
	dy/dx = 3/2*A*(A*x + B)^(1/2)
	x = ((y + C)^(2/3) - B) / A

	f(0) = 0
	f'(0) = 1
	"""

	C = bias
	B = C ** (2.0/3.0)
	A = 2.0 / (3.0 * math.sqrt(B))

	clip_bottom = -B/A

	if clip_top is not None:
		# invert function to get x value to clip inputs that will result in desired y clip value
		clip_top = ((clip_top + C)**(2./3.) - B) / A

	x = np.clip(vals, clip_bottom, clip_top)

	return np.power(A*x + B, 3./2.) - C


class ThreeHalfsDriver(StatelessProcessorBase):
	def __init__(self, gain=1.0, bias=1.0, clip_top=None):
		C = bias
		B = C**(2./3.)
		A = 2.0 / (3.0 * math.sqrt(B))

		clip_bottom = -B / A

		if clip_top is not None:
			# invert function to get x value to clip inputs to get desired y clip value
			clip_top = ((clip_top + C)**(2./3.) - B) / A

		A *= gain

		self.coeffs = A, B, C
		self.clip_bottom = clip_bottom
		self.clip_top = clip_top

	def process_sample(self, x):
		A, B, C = self.coeffs
		return (A*x + B)**(3./2.) - C

	def process_vector(self, x):
		A, B, C = self.coeffs
		return np.power(A*x + B, 3./2.) - C


def squarize(vals: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
	if np.isscalar(vals):
		return 1.0 if vals >= 0.0 else -1.0
	else:
		y = np.ones_like(vals)
		y[vals < 0.0] = -1.0
		return y


class Squarizer(StatelessProcessorBase):
	def process_sample(self, sample: float) -> float:
		return 1.0 if sample >= 0.0 else -1.0

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		y = np.ones_like(vec)
		y[vec < 0.0] = -1.0
		return y


def plot(args):
	from matplotlib import pyplot as plt

	x = np.linspace(-10., 10., 20001)

	clipx = clip(x)
	tanhx = tanh(x)
	atanx = atan(x)
	sigx = sigmoid(x)
	lnx = ln_drive(x)
	sqrtx = sqrt_drive(x)
	quadx = quadratic_drive(x)
	cubx = cubic_drive(x)
	thx = three_halfs_drive(x)
	sqi = squarize(x)

	dclipx = utils.derivatives(clipx, x, 3)
	dtanhx = utils.derivatives(tanhx, x, 3)
	datanx = utils.derivatives(atanx, x, 3)
	dsigx = utils.derivatives(sigx, x, 3)
	dlnx = utils.derivatives(lnx, x, 3)
	dsqrtx = utils.derivatives(sqrtx, x, 3)
	dquadx = utils.derivatives(quadx, x, 3)
	dcubx = utils.derivatives(cubx, x, 3)
	dthx = utils.derivatives(thx, x, 3)
	dsqi = utils.derivatives(sqi, x, 3)

	plt.figure()
	plt.subplot(4, 1, 1)

	plt.plot(x, clipx, label='clip')
	plt.plot(x, tanhx, label='tanh')
	plt.plot(x, atanx, label='atan')
	plt.plot(x, sigx, label='Sigmoid')
	plt.plot(x, lnx, label='ln')
	plt.plot(x, sqrtx, label='sqrt')
	plt.plot(x, quadx, label='Quadratic')
	plt.plot(x, cubx, label='Cubic')
	plt.plot(x, thx, label='3/2')
	plt.plot(x, sqi, label='squarizer')
	plt.ylabel('Transfer function')
	plt.grid()
	plt.legend()

	plt.subplot(4, 1, 2)
	plt.plot(x, dclipx[0], label='clip')
	plt.plot(x, dtanhx[0], label='tanh')
	plt.plot(x, datanx[0], label='atan')
	plt.plot(x, dsigx[0], label='Sigmoid')
	plt.plot(x, dlnx[0], label='ln')
	plt.plot(x, dsqrtx[0], label='sqrt')
	plt.plot(x, dquadx[0], label='Quadratic')
	plt.plot(x, dcubx[0], label='Cubic')
	plt.plot(x, dthx[0], label='3/2')
	plt.plot(x, dsqi[0], label='squarizer')
	plt.ylabel('1st derivative')
	plt.grid()

	plt.subplot(4, 1, 3)
	plt.plot(x, dclipx[1], label='clip')
	plt.plot(x, dtanhx[1], label='tanh')
	plt.plot(x, datanx[1], label='atan')
	plt.plot(x, dsigx[1], label='Sigmoid')
	plt.plot(x, dlnx[1], label='ln')
	plt.plot(x, dsqrtx[1], label='sqrt')
	plt.plot(x, dquadx[1], label='Quadratic')
	plt.plot(x, dcubx[1], label='Cubic')
	plt.plot(x, dthx[1], label='3/2')
	plt.plot(x, dsqi[1], label='squarizer')
	plt.ylabel('2nd derivative')
	plt.grid()
	plt.ylim([-2.1, 2.1])

	plt.subplot(4, 1, 4)
	plt.plot(x, dclipx[2], label='clip')
	plt.plot(x, dtanhx[2], label='tanh')
	plt.plot(x, datanx[2], label='atan')
	plt.plot(x, dsigx[2], label='Sigmoid')
	plt.plot(x, dlnx[2], label='ln')
	plt.plot(x, dsqrtx[2], label='sqrt')
	plt.plot(x, dquadx[2], label='Quadratic')
	plt.plot(x, dcubx[2], label='Cubic')
	plt.plot(x, dthx[2], label='3/2')
	plt.plot(x, dsqi[2], label='squarizer')
	plt.ylabel('3rd derivative')
	plt.grid()
	plt.ylim([-2.1, 6.1])

	plt.figure()
	x = np.linspace(-3., 3., 201)
	plt.plot(x, x, label='y = x')
	plt.plot(x, three_halfs_drive(x, bias=0.1, clip_top=3), label='Bias 0.1, clip 3')
	plt.plot(x, three_halfs_drive(x, bias=1, clip_top=4), label='Bias 1, clip 4')
	plt.plot(x, three_halfs_drive(x, bias=2), label='Bias 2')
	plt.legend()
	plt.grid()
	plt.title('3/2 drive detail')
	plt.xlabel('x')
	plt.ylabel('y')  # wow such descriptive

	plt.show()


def main(args):
	from matplotlib import pyplot as plt
	from analysis.distortion import plot_distortion

	asym_clip = np.vectorize(lambda x: clip(x + 0.25))
	asym_hardness = np.vectorize(lambda x: clip(x) if x < 0 else tanh(x))

	funcs = [
		(clip, 'clip'),
		(tanh, 'tanh'),
		(atan, 'atan'),
		(sigmoid, 'sigmoid'),
		(ln_drive, 'ln'),
		(sqrt_drive, 'sqrt'),
		(quadratic_drive, 'quadratic'),
		(cubic_drive, 'cubic'),
		(three_halfs_drive, '3/2 power'),
		(asym_clip, 'Biased clip'),
		(asym_hardness, 'Asymmetric hard/tanh'),
		(squarize, 'Squarize'),
	]

	for func, name in funcs:
		print('Processing %s' % name)
		plot_distortion(func, title=name)

	plt.show()
