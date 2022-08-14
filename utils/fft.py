#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, Union, Callable

import numpy as np

from utils.utils import to_dB


@dataclass
class FftResult:

	n: int
	sample_rate: float
	full_complex_result: np.ndarray
	bin_size: float
	bin_edges: np.ndarray
	bin_centers: np.ndarray
	normalized: bool

	_magnitude: Optional[np.ndarray] = None
	_magnitude_dB: Optional[np.ndarray] = None
	_phase_radians: Optional[np.ndarray] = None

	@property
	def complex(self):
		return self.full_complex_result[:len(self.full_complex_result)//2]

	@property
	def magnitude(self):
		if self._magnitude is None:
			self._magnitude = np.abs(self.complex)
		return self._magnitude

	@property
	def magnitude_dB(self):
		if self._magnitude_dB is None:
			self._magnitude_dB = to_dB(self.magnitude)
		return self._magnitude_dB

	def set_magnitude_dB(self, min_dB: Optional[float]):
		self._magnitude_dB = to_dB(self.magnitude, min_dB=min_dB)

	@property
	def phase_radians(self):
		if self._phase_radians is None:
			self._phase_radians = np.angle(self.complex)
		return self._phase_radians

	@property
	def phase_degrees(self):
		return np.degrees(self.phase_radians)


def do_fft(
		data,
		sample_rate,
		nfft: Optional[int] = None,
		window: Union[bool, Callable] = True,
		min_dB: Optional[float] = None,
		normalize = False,
		scale = 1.0,
		) -> FftResult:

	if window is True:
		data = data * np.hamming(len(data))
	elif window:
		data = data * window(len(data))

	fft_kwargs = dict()
	if nfft is not None:
		fft_kwargs['n'] = nfft
	if normalize:
		fft_kwargs['norm'] = 'forward'

	fft_result_raw = np.fft.fft(data, **fft_kwargs)

	fft_result_raw *= scale

	fft_len = len(fft_result_raw)

	# Slightly weird logic here in case of odd-size FFT
	bin_edges = np.linspace(0, sample_rate, fft_len, endpoint=False)[:fft_len//2]
	bin_size = bin_edges[1] - bin_edges[0]
	bin_centers = bin_edges + 0.5*bin_size

	result = FftResult(
		n=fft_len,
		sample_rate=sample_rate,
		full_complex_result=fft_result_raw,
		bin_size=bin_size,
		bin_edges=bin_edges,
		bin_centers=bin_centers,
		normalized=normalize,
	)

	if min_dB is not None:
		result.set_magnitude_dB(min_dB)

	return result
