#!/usr/bin/env python3


import numpy as np
from typing import Union, Optional, Iterable, List
from filters.iir_filters import IIRFilter, ButterworthLowpass
from filters.filter_base import FilterBase


class UpsamplerBase:
	def process_sample(self, sample: float) -> Iterable[float]:
		raise NotImplementedError('process_sample() to be implemented by the child class!')

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		raise NotImplementedError('process_vector() to be implemented by the child class!')

	def process(self, input: Union[float, np.ndarray]) -> np.ndarray:
		"""Calls process_sample or process_vector depending on type of input"""
		if np.isscalar(input):
			return self.process_sample(input)
		else:
			return self.process_vector(input)

	def reset(self) -> None:
		"""Reset Processor state (but not parameters)"""
		pass

	def __call__(self, *args, **kwargs):
		return self.process(*args, **kwargs)


class DownsamplerBase:
	def process_sample(self, sample: float) -> Optional[float]:
		raise NotImplementedError('process_sample() to be implemented by the child class!')

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		raise NotImplementedError('process_vector() to be implemented by the child class!')

	def process(self, input: Union[float, np.ndarray]) -> Union[None, float, np.ndarray]:
		"""Calls process_sample or process_vector depending on type of input"""
		if np.isscalar(input):
			return self.process_sample(input)
		else:
			return self.process_vector(input)

	def reset(self) -> None:
		"""Reset Processor state (but not parameters)"""
		pass

	def __call__(self, *args, **kwargs):
		return self.process(*args, **kwargs)


class ZeroStuffingUpsampler(UpsamplerBase):
	def __init__(self, ratio: int):
		if ratio <= 1:
			raise ValueError('Upsampler ratio must be > 1')
		self.ratio = ratio

	def process_sample(self, sample: float) -> List[float]:
		return [sample * self.ratio] + [0.0] * (self.ratio - 1)

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		# TODO: numpy vectorize this
		y = np.zeros(len(vec) * self.ratio)
		for n, x in enumerate(vec):
			y[n*self.ratio] = x * self.ratio
		return y


class ZeroOrderHoldUpsampler(UpsamplerBase):
	def __init__(self, ratio: int):
		if ratio <= 1:
			raise ValueError('Upsampler ratio must be > 1')
		self.ratio = ratio

	def process_sample(self, sample: float) -> List[float]:
		return [sample] * self.ratio

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		return np.repeat(vec, self.ratio)


class LerpUpsampler(UpsamplerBase):
	"""
	First-order upsampler (i.e. linear interpolation)
	"""

	def __init__(self, ratio: int):
		if ratio <= 1:
			raise ValueError('Upsampler ratio must be > 1')
		self.ratio = ratio
		self.x1 = 0.0
		self.lerp_vals_curr = np.linspace(0.0, 1.0, self.ratio, endpoint=False)
		self.lerp_vals_prev = 1.0 - self.lerp_vals_curr

	def process_sample(self, x: float) -> np.ndarray:
		y = self.lerp_vals_prev * self.x1 + self.lerp_vals_curr * x
		self.x1 = x
		return y

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		# TODO: numpy vectorize this
		y = np.zeros(len(vec) * self.ratio)
		for n, x in enumerate(vec):
			start = n * self.ratio
			end = start + self.ratio
			y[start:end] = self.process_sample(x)
		return y


class FilterUpsampler(UpsamplerBase):
	"""
	A basic upsampler using a given filter
	Doesn't do any polyphase filtering or anything, so won't be too efficient
	"""

	def __init__(self, ratio: int, lpf: FilterBase, zero_order_hold=True):
		"""

		:param ratio: Upsampling ratio - must be integer > 1
		:param lpf: lowpass filter - ideally, cutoff frequency should be 0.5 / ratio
		:param zero_order_hold: if true, will use zero-order-hold (as opposed to zero stuffing)
		"""
		self.resampler = ZeroOrderHoldUpsampler(ratio) if zero_order_hold else ZeroStuffingUpsampler(ratio)
		self.lpf = lpf

	def reset(self):
		self.lpf.reset()

	def process_sample(self, sample: float) -> List[float]:
		y = self.resampler.process_sample(sample)
		for n in range(len(y)):
			y[n] = self.lpf.process_sample(y[n])
		return y

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		return self.lpf.process_vector(self.resampler.process_vector(vec))


class IIRUpsampler(FilterUpsampler):
	"""
	Upsampler using an elliptic (Cauer) filter
	Uses IIR filter, so not linear-phase
	Not as efficient as a properly-implemented polyphase FIR filter
	"""

	def __init__(self, ratio: int, order=12, wc=None, zero_order_hold=True, passband_ripple_dB=0.5, stopband_ripple_dB=120):
		"""
		:param ratio: Upsampling ratio - must be integer > 1
		:param order: Elliptic filter order
		:param wc: Normalized cutoff frequency relative to original sample rate, typically 0.5
		:param zero_order_hold: if true, will use zero-order-hold (as opposed to zero stuffing)
		:param passband_ripple_dB:
		:param stopband_ripple_dB:
		"""
		if ratio <= 1:
			raise ValueError('Upsampler ratio must be > 1')
		if wc is None:
			wc = 0.5

		wc /= ratio
		super().__init__(
			ratio,
			IIRFilter(wc, order=order, passband_ripple_dB=passband_ripple_dB, stopband_ripple_dB=stopband_ripple_dB),
			zero_order_hold=zero_order_hold)


class ButterworthUpsampler(FilterUpsampler):
	"""
	Upsampler using a Butterworth filter
	Not very efficient, but it does the job
	Uses IIR filter, so not linear-phase
	"""

	def __init__(self, ratio: int, order=16, wc=None, zero_order_hold=True):
		"""
		:param ratio: Upsampling ratio - must be integer > 1
		:param order: Butterworth filter order
		:param wc: Normalized cutoff frequency relative to original sample rate, typically 0.5
		:param zero_order_hold: if true, will use zero-order-hold (as opposed to zero stuffing)
		"""
		if ratio <= 1:
			raise ValueError('Upsampler ratio must be > 1')
		if wc is None:
			wc = 0.5

		wc /= ratio
		super().__init__(ratio, ButterworthLowpass(wc, order=order), zero_order_hold=zero_order_hold)


class PolyphaseIirAllpassUpsampler(UpsamplerBase):
	"""
	Upsampler based on polyphase first-order IIR allpass interpolators
	Performance is not great (filters are only first-order, and not really meant for upsampling)
	But makes for a simple polyphase experiment
	"""

	class _IirAllpassInterpolator:
		# https://ccrma.stanford.edu/~jos/pasp/First_Order_Allpass_Interpolation.html
		def __init__(self, interp_coeff):
			self.eta = interp_coeff
			self.x1 = self.y1 = 0.0

		def reset(self):
			self.x1 = self.y1 = 0.0

		def process_sample(self, x: float) -> float:
			y = self.eta * (x - self.y1) + self.x1
			self.x1 = x
			self.y1 = y
			return y

	def __init__(self, ratio: int):
		if ratio <= 1:
			raise ValueError('Upsampler ratio must be > 1')

		self.ratio = ratio
		interp_vals = np.linspace(0.0, 1.0, self.ratio, endpoint=False)[1:]
		self.interpolators = [self._IirAllpassInterpolator(val) for val in interp_vals]

	def reset(self) -> None:
		for i in self.interpolators:
			i.reset()

	def process_sample(self, x: float) -> List[float]:
		return [i.process_sample(x) for i in self.interpolators] + [x]

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		n_samp_out = len(vec) * self.ratio
		y = np.zeros(n_samp_out)

		for n, x in enumerate(vec):
			start = self.ratio * n
			end = start + self.ratio
			y[start:end] = self.process_sample(x)

		return y


class NaiveDownsampler(DownsamplerBase):
	"""
	Naive downsampler
	i.e. downsample just by picking every Nth sample
	"""

	def __init__(self, ratio: int):
		if ratio <= 1:
			raise ValueError('Downsampler ratio must be > 1')
		self.ratio = ratio
		self.counter = 0

	def reset(self):
		self.counter = 0

	def process_sample(self, x: float) -> Optional[float]:
		self.counter = (self.counter + 1) % self.ratio
		return x if (self.counter == 0) else None

	def process_vector(self, vec: np.ndarray):
		# TODO: support these
		if self.counter != 0:
			raise NotImplementedError('NaiveDownsampler does not currently support mixing process_sample and process_vector calls')
		if len(vec) % self.ratio != 0:
			raise NotImplementedError('NaiveDownsampler does not currently support input vector where length is not multiple of ratio')

		return vec[(self.ratio - 1)::self.ratio]


class AveragingDownsampler(DownsamplerBase):
	"""Basic downsampler that just averages every N samples"""

	def __init__(self, ratio: int):
		if ratio <= 1:
			raise ValueError('Downsampler ratio must be > 1')
		self.ratio = ratio
		self.counter = 0
		self.vals = np.zeros(self.ratio)

	def reset(self):
		self.counter = 0
		self.vals = np.zeros(self.ratio)

	def process_sample(self, x: float) -> Optional[float]:
		self.vals[self.counter] = x
		self.counter += 1
		if self.counter >= self.ratio:
			avg = np.average(self.vals)
			self.reset()
			return avg
		else:
			return None

	def process_vector(self, vec: np.ndarray):
		# TODO: support these
		if self.counter != 0:
			raise NotImplementedError('AveragingDownsampler does not currently support mixing process_sample and process_vector calls')
		if len(vec) % self.ratio != 0:
			raise NotImplementedError('AveragingDownsampler does not currently support input vector where length is not multiple of ratio')

		n_samp_down = len(vec) // self.ratio
		y = np.zeros(n_samp_down)
		for n in range(n_samp_down):
			start = n*self.ratio
			end = start + self.ratio
			y[n] = np.average(vec[start:end])


class FilterDownsampler(DownsamplerBase):
	"""
	A basic upsampler using a given filter
	Doesn't do any polyphase filtering or anything, so won't be too efficient
	"""

	def __init__(self, ratio: int, lpf: FilterBase):
		"""

		:param ratio: Downsampling ratio - must be integer > 1
		:param lpf: lowpass filter - ideally, cutoff frequency should be 0.5 / ratio
		"""
		self.resampler = NaiveDownsampler(ratio)
		self.lpf = lpf

	def reset(self):
		self.resampler.reset()
		self.lpf.reset()

	def process_sample(self, sample: float) -> Optional[float]:
		y = self.lpf.process_sample(sample)
		return self.resampler.process_sample(y)

	def process_vector(self, vec: np.ndarray) -> np.ndarray:
		return self.resampler.process_vector(self.lpf.process_vector(vec))


class IIRDownsampler(FilterDownsampler):
	"""
	Upsampler using an elliptic (Cauer) filter
	Uses IIR filter, so not linear-phase
	Not as efficient as a properly-implemented polyphase FIR filter
	"""

	def __init__(self, ratio: int, order=12, wc=None, passband_ripple_dB=0.5, stopband_ripple_dB=120):
		"""
		:param ratio: Upsampling ratio - must be integer > 1
		:param order: Elliptic filter order
		:param wc: Normalized cutoff frequency relative to original sample rate, typically 0.5 / ratio
		:param passband_ripple_dB:
		:param stopband_ripple_dB:
		"""
		if ratio <= 1:
			raise ValueError('Upsampler ratio must be > 1')
		if wc is None:
			wc = 0.5 / ratio

		super().__init__(
			ratio,
			IIRFilter(wc, order=order, passband_ripple_dB=passband_ripple_dB, stopband_ripple_dB=stopband_ripple_dB))


class ButterworthDownsampler(FilterDownsampler):
	"""
	Upsampler using a Butterworth filter
	Not very efficient, but it does the job
	Uses IIR filter, so not linear-phase
	"""

	def __init__(self, ratio: int, order=16, wc=None):
		"""
		:param ratio: Upsampling ratio - must be integer > 1
		:param order: Butterworth filter order
		:param wc: Normalized cutoff frequency relative to original sample rate, typically 0.5 / ratio
		"""
		if ratio <= 1:
			raise ValueError('Upsampler ratio must be > 1')
		if wc is None:
			wc = 0.5 / ratio
		super().__init__(ratio, ButterworthLowpass(wc, order=order))


# TODO: Add actual good resamplers (i.e. polyphase FIR)


def _plot_time_domain_sweep(n_samp, ratio):
	from matplotlib import pyplot as plt
	from generation.signal_generation import gen_freq_sweep_sine, sample_time_index

	sr = 48000
	sr_down = sr // ratio
	sr_up = sr * ratio

	n_samp_down = n_samp // ratio
	n_samp_up = n_samp * ratio

	x = gen_freq_sweep_sine(0.0, 0.5, n_samp=n_samp, log=False)

	ds = IIRDownsampler(ratio=ratio)
	us = IIRUpsampler(ratio=ratio)

	y_down = ds.process_vector(x)
	y_up = us.process_vector(x)

	t = sample_time_index(n_samp, sr)
	t_down = sample_time_index(n_samp_down, sr_down)
	t_up = sample_time_index(n_samp_up, sr_up)

	plt.figure()
	plt.plot(t_up, y_up, '.-', label='Upsampled')
	plt.plot(t, x, '.-', label='Original')
	plt.plot(t_down, y_down, '.-', label='Downsampled')
	plt.grid()
	plt.legend()
	plt.title('Resampling, ratio %i' % ratio)


def _plot_upsampling():
	from matplotlib import pyplot as plt
	from generation.signal_generation import gen_sine, gen_noise, sample_time_index
	from utils import utils
	from utils.plot_utils import plot_fft

	class IIRNonZOHUpsampler(IIRUpsampler):
		def __init__(self, ratio):
			super().__init__(ratio=ratio, zero_order_hold=False)

	class IIRZOHUpsampler(IIRUpsampler):
		def __init__(self, ratio):
			super().__init__(ratio=ratio, zero_order_hold=True)

	upsampler_classes = [
		ZeroStuffingUpsampler,
		ZeroOrderHoldUpsampler,
		LerpUpsampler,
		IIRNonZOHUpsampler,
		IIRZOHUpsampler,
		ButterworthUpsampler,
		PolyphaseIirAllpassUpsampler,
	]

	ratio = 8
	periods = 16

	period_samp = 64
	period_samp_up = period_samp * ratio

	n_samp = period_samp * periods
	n_samp_up = n_samp * ratio

	sr = 48000
	sr_up = sr * ratio

	t = sample_time_index(n_samp, sr)
	t_up = sample_time_index(n_samp_up, sr_up)

	x = gen_sine(freq_norm=(1.0 / period_samp), n_samp=n_samp)

	# I'm not sure why this is needed - in theory we should be fine with rectangular since freq is exactly 1/64
	# But without it, ButterworthUpsampler aliased components disappear
	window = np.blackman

	plt.figure()

	print('Input RMS: %.2f dB' % utils.to_dB(utils.rms(x)))

	# Technically it's wrong to display this with dots connected, but it's impossible to find in the plot otherwise
	# (and plt.stem() doesn't play nice with multiple plots on same axes)
	plt.subplot(1, 2, 1)
	plt.plot(t, x, '.-', label='Input')

	plt.subplot(1, 2, 2)
	plot_fft(x, sr, fmt='.-', window=window, log=False)

	for upsampler_class in upsampler_classes:
		name = upsampler_class.__name__
		us = upsampler_class(ratio=ratio)
		y = us.process_vector(x)
		print('%s RMS: %.2f dB' % (name, utils.to_dB(utils.rms(y))))

		plt.subplot(1, 2, 1)
		plt.plot(t_up, y, '.', label=name)

		plt.subplot(1, 2, 2)
		plot_fft(y / ratio, sr_up, fmt='.-', window=window, log=False)

	plt.subplot(1, 2, 1)
	plt.xlim([0, t[period_samp - 1]])
	plt.grid()
	plt.legend()
	plt.title('Upsamplers, ratio %i' % ratio)

	plt.subplot(1, 2, 2)
	plt.grid()
	plt.xlim([0.0, 0.5*sr_up])


def _plot_downsampling(n_samp=4096):
	from matplotlib import pyplot as plt
	from generation.signal_generation import gen_sine
	from utils import plot_utils, utils

	sample_rate = 192000.0
	ratio = 4

	# Very similar to freq_response.get_freq_response

	freqs = [
		10.0,
		100.0,
		1000.0, 5000.0,
		10000.0, 15000.0,
		20000.0, 22000.0, 24000.0, 25000.0,
		30000.0, 35000.0,
		40000.0, 45000.0, 48000.0,
		50000.0, 55000.0,
		60000.0, 65000.0,
		70000.0, 75000.0,
		80000.0, 85000.0,
		90000.0, 95000.0
	]

	ds4 = IIRDownsampler(ratio=ratio, order=4)
	ds8 = IIRDownsampler(ratio=ratio, order=8)
	ds16 = IIRDownsampler(ratio=ratio, order=16)

	amp4 = np.zeros_like(freqs)
	amp8 = np.zeros_like(freqs)
	amp16 = np.zeros_like(freqs)

	for n, f in enumerate(freqs):

		ds4.reset()
		ds8.reset()
		ds16.reset()

		wc = f / sample_rate

		x = gen_sine(wc, n_samp)

		y4 = ds4.process_vector(x)
		y8 = ds8.process_vector(x)
		y16 = ds16.process_vector(x)

		rmsx = utils.rms(x)
		amp4[n] = utils.rms(y4) / rmsx
		amp8[n] = utils.rms(y8) / rmsx
		amp16[n] = utils.rms(y16) / rmsx

	amp4_dB = utils.to_dB(amp4)
	amp8_dB = utils.to_dB(amp8)
	amp16_dB = utils.to_dB(amp16)

	plt.figure()
	plt.plot(freqs, amp4_dB, '.-', label='order 4')
	plt.plot(freqs, amp8_dB, '.-', label='order 8')
	plt.plot(freqs, amp16_dB, '.-', label='order 16')
	plt.grid()
	plt.legend()
	plt.title('192k-48k Downsampler frequency response')


def _plot_freq_domain_sweep(ratio):
	from matplotlib import pyplot as plt
	from generation.signal_generation import gen_freq_sweep_sine
	from utils import plot_utils

	n_samp = 2**15

	sr = 48000
	sr_down = sr // ratio
	sr_up = sr * ratio

	nfft = 1024
	nfft_down = nfft // ratio
	nfft_up = nfft * ratio

	x = gen_freq_sweep_sine(0.0, 0.5, n_samp=n_samp, log=False)

	ds = IIRDownsampler(ratio=4)
	us = IIRUpsampler(ratio=4)

	y_down = ds.process_vector(x)
	y_up = us.process_vector(x)

	plt.figure()

	plt.subplot(3, 1, 1)
	plot_utils.plot_spectrogram(x, sample_rate=sr, nfft=nfft, log=False)
	plt.title('Original sweep')
	plt.grid()

	plt.subplot(3, 1, 2)
	plot_utils.plot_spectrogram(y_down, sample_rate=sr_down, nfft=nfft_down, log=False)
	plt.title('Downsampled')
	plt.grid()

	plt.subplot(3, 1, 3)
	plot_utils.plot_spectrogram(y_up, sample_rate=sr_up, nfft=nfft_up, log=False)
	plt.title('Upsampled')
	plt.grid()


def plot(args):
	from matplotlib import pyplot as plt
	_plot_time_domain_sweep(n_samp=512, ratio=4)
	_plot_upsampling()
	_plot_downsampling()
	_plot_freq_domain_sweep(ratio=4)
	plt.show()


def main(args):
	plot(args)
