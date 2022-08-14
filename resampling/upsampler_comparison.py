#!/usr/bin/env python3

import argparse
from math import floor, sin, cos, pi, sqrt, isclose
from typing import Callable, Tuple, Optional, Iterable

from matplotlib import pyplot as plt
import numpy as np
import scipy.interpolate

#from analysis.freq_response import single_freq_dft
from filters.allpass import FractionalDelayAllpass
from generation.signal_generation import gen_sine
#from resampling.resamplers import UpsamplerBase
from utils.utils import lerp, quadratic_coeffs
from utils.fft import do_fft


def interpolated_t(num_samples: int, ratio: int) -> np.ndarray:
	num_samples_upsampled = (num_samples - 1) * ratio + 1
	return np.linspace(0, num_samples - 1, num_samples_upsampled, endpoint=True)


def safe_access(x: Iterable, n: int):
	return x[n] if (0 <= n < len(x)) else 0.0


def upsample_fft(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:
	f = np.fft.fft(x)

	f_first_half = f[:len(f)//2]
	f_last_half = f[len(f)//2:]

	f_up = [f_first_half] + ([np.zeros_like(f)] * (ratio - 1)) + [f_last_half]
	f_up = np.concatenate(f_up)

	y = np.fft.ifft(f_up)
	y = np.real(y)

	y *= ratio

	t_out = np.arange(len(y), dtype=np.float) * 1.0 / ratio
	return y, t_out


def upsample_nearest(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	y = np.zeros(len(t_out))

	for idx, t in enumerate(t_out):
		t_nearest = int(round(t))
		y[idx] = x[t_nearest]

	return y, t_out


def upsample_lin(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	y = np.zeros(len(t_out))

	for idx, t in enumerate(t_out):
		t0 = int(floor(t))
		t1 = t0 + 1
		lerp_idx = t - t0
		assert 0.0 <= lerp_idx < 1.0

		x0 = safe_access(x, t0)
		x1 = safe_access(x, t1)

		y[idx] = lerp((x0, x1), lerp_idx)

	return y, t_out


def upsample_quadratic(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	y = np.zeros(len(t_out))

	for idx, t in enumerate(t_out):
		t0 = int(floor(t))
		tn1 = t0 - 1
		t1 = t0 + 1
		lerp_idx = t - t0
		assert 0.0 <= lerp_idx < 1.0

		xn1 = safe_access(x, tn1)
		x0 = safe_access(x, t0)
		x1 = safe_access(x, t1)

		a, b, c = quadratic_coeffs(xn1, x0, x1)

		y[idx] = a * lerp_idx * lerp_idx + b * lerp_idx + c

	return y, t_out


def upsample_cos(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	y = np.zeros(len(t_out))

	for idx, t in enumerate(t_out):
		t0 = int(floor(t))
		t1 = t0 + 1
		lerp_idx_lin = t - t0
		assert 0.0 <= lerp_idx_lin < 1.0
		
		lerp_idx = cos(pi * (1.0 - lerp_idx_lin)) * 0.5 + 0.5
		assert 0.0 <= lerp_idx < 1.0

		x0 = x[t0] if 0 <= t0 < len(x) else 0.0
		x1 = x[t1] if 0 <= t1 < len(x) else 0.0

		y[idx] = lerp((x0, x1), lerp_idx)

	return y, t_out


# FIXME: For both of these STFT methods the impulse response looks great, but any other shape looks bad
# Windowing might be part of the problem, but not the whole story
# I suspect I have an error in the manual reconstruction from sines


def upsample_stft4(x: np.ndarray, ratio: int, window=np.hamming) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	y = np.zeros(len(t_out))

	size_up = 4 * ratio

	w = window(4)
	window_scale = 2.0 / (w[1] + w[2])

	f_t0 = None
	f = None

	for idx, t in enumerate(t_out):
		t0 = int(floor(t)) - 1
		t1 = t0 + 1
		t2 = t0 + 2
		t3 = t0 + 3

		# In theroy we could iterate 4 samples at a time here
		# However, the goal I have in mind for this is an interpolator at arbitrary points
		# Writing the code this way is closer to that use case (and we can still reuse FFT results)

		lerp_idx = t - t0
		assert 1.0 <= lerp_idx < 2.0

		t_arr = [t0, t1, t2, t3]
		x_arr = [safe_access(x, t) for t in t_arr]

		x_arr = np.array(x_arr) * w

		# Reuse previous FFT result if it hasn't changed
		if (f is None) or (f_t0 != t0):
			f = np.fft.fft(x_arr)
			f_t0 = t0

		assert len(f) == 4

		mag = np.abs(f)
		phase = np.angle(f)

		y0 = mag[0] * np.cos(2 * pi * 0 * lerp_idx + phase[0])
		y1 = mag[1] * np.cos(2 * pi * 0.25 * lerp_idx + phase[1])
		y2 = mag[2] * np.cos(2 * pi * 0.5 * lerp_idx + phase[2])
		y3 = mag[3] * np.cos(2 * pi * 0.75 * lerp_idx + phase[3])

		#window_scale_this_sample = 1.0  # TODO

		# TODO: roll this gain factor into window
		y[idx] = 0.25 * window_scale * (y0 + y1 + y2 + y3)

	return y, t_out


def upsample_stft(x: np.ndarray, ratio: int, size=8, window=np.hamming, use_ifft=True) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	y = np.zeros(len(t_out))

	size_up = size * ratio

	w = window(size)
	window_scale = 2.0 / (w[size//2 - 1] + w[size//2])

	# TODO: handle divide by zero
	inv_window = 1.0 / window(size * ratio)

	f_t0 = None
	f = None

	for idx_out, t in enumerate(t_out):
		t0 = int(floor(t)) - (size//2 - 1)
		lerp_idx = t - t0
		assert (size//2 - 1) <= lerp_idx < size//2

		t_arr = [t0 + idx for idx in range(size)]
		x_arr = [safe_access(x, t) for t in t_arr]

		x_arr = np.array(x_arr) * w

		if (f is None) or (f_t0 != t0):
			f = np.fft.fft(x_arr)

		assert len(f) == size

		#window_scale_this_sample = 1.0  # TODO

		if use_ifft:

			f_first_half = f[:len(f)//2]
			f_last_half = f[len(f)//2:]
			f_up = [f_first_half] + ([np.zeros_like(f)] * (ratio - 1)) + [f_last_half]
			f_up = np.concatenate(f_up)

			# TODO: also cache this, just like f
			y_full = np.fft.ifft(f_up)
			y_full = np.real(y_full)
			y_full *= ratio
			y_full *= inv_window

			subsample_idx = idx_out % ratio
			y_full_idx = ratio * (size//2 - 1) + subsample_idx

			yn = y_full[y_full_idx]

		else:
			mag = np.abs(f)
			phase = np.angle(f)

			# TODO: roll this gain factor into window
			yn = 1/size * window_scale * sum(
				mag[idx] * np.cos(2 * pi * idx/size * lerp_idx + phase[idx])
				for idx in range(size)
			)

		y[idx_out] = yn

	return y, t_out



def upsample_scipy(x: np.ndarray, ratio: int, interpolator) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	interpolator = interpolator(np.arange(len(x)), x)
	y = interpolator(t_out)
	return y, t_out


def upsample_pchip(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:
	return upsample_scipy(x, ratio, scipy.interpolate.PchipInterpolator)


def upsample_spline(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:
	return upsample_scipy(x, ratio, scipy.interpolate.CubicSpline)


def upsample_akima(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:
	return upsample_scipy(x, ratio, scipy.interpolate.Akima1DInterpolator)


def upsample_lagrange(x: np.ndarray, ratio: int, order: int) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	y = np.zeros(len(t_out))

	for idx, t in enumerate(t_out):

		floor_t = int(floor(t))

		t_arr = [
			floor_t - (order // 2) + idx + 1
			for idx
			in range(order)
		]

		x_arr = [(x[tt] if 0 <= tt < len(x) else 0.0) for tt in t_arr]
		poly = scipy.interpolate.lagrange(t_arr, x_arr)
		y[idx] = poly(t)

	return y, t_out


def upsample_one_pole_allpass_polyphase(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:

	t_out = np.arange(len(x) * ratio, dtype=np.float) * 1.0 / ratio
	y = np.zeros(len(x) * ratio)

	delay_times = [idx / ratio for idx in range(ratio)]
	# TODO: figure out why this reversed() is necessary - same as in resamplers.py
	delays = [FractionalDelayAllpass(time) for time in reversed(delay_times)]
	y_polyphase = [delay(x) for delay in delays]

	for x_idx, _ in enumerate(x):
		y_idx_base = x_idx * ratio
		for polyphase_idx, polyphase_vec in enumerate(y_polyphase):
			y[y_idx_base + polyphase_idx] = polyphase_vec[x_idx]

	y = y[(ratio - 1):]
	t_out = t_out[:(1 - ratio)]

	return y, t_out


def upsample_one_pole_allpass_polyphase_then_lerp(x: np.ndarray, ratio: int, phases: int) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	y_polyphase, t_polyphase = upsample_one_pole_allpass_polyphase(x, phases)
	interpolator = scipy.interpolate.interp1d(t_polyphase, y_polyphase, kind='linear')
	y = interpolator(t_out)
	return y, t_out


def upsample_sin_fit(x: np.ndarray, ratio: int) -> Tuple[np.ndarray, np.ndarray]:
	t_out = interpolated_t(len(x), ratio)
	y = np.zeros(len(t_out))

	for idx, t in enumerate(t_out):
		t0 = int(floor(t)) - 1
		t1 = t0
		t2 = t0 + 1
		t_idx = t - t0
		assert 1 <= t_idx < 2

		x0 = safe_access(x, t0)
		x1 = safe_access(x, t1)
		x2 = safe_access(x, t2)

		if not any([x0, x1, x2]):
			y[idx] = 0.0
			continue

		# Sine from 3 points
		# https://math.stackexchange.com/questions/609424/sine-function-description-using-three-points

		# FIXME:
		# The top answer there is right - this is unsolvable if x1 = 0; we need 4 points, which is even harder to solve

		w = np.arccos((x0 + x2) / (2 * x1))

		A_sin_phi = x0

		# A * cos(phi) * sin(w) + x0 * cos(w) = x1
		A_cos_phi = (x1 - x0 * cos(w)) / sin(w)

		A = sqrt(A_cos_phi*A_cos_phi + A_sin_phi*A_sin_phi)
		phi = np.arcsin(x0 / A)

		y[idx] = A * np.sin(w * t_idx + phi)

	return y, t_out


def upsample_windowed_sinc(
		x: np.ndarray,
		ratio: int,
		window_half_width_samples: int,
		window: Callable,
		sinc_scale=1.0) -> Tuple[np.ndarray, np.ndarray]:

	window_size = (2 * window_half_width_samples * ratio) + 1

	t_out = interpolated_t(len(x), ratio)
	y = np.zeros(len(t_out))

	sinc_t = np.linspace(-window_half_width_samples, window_half_width_samples, window_size, endpoint=True)
	sinc_t *= sinc_scale
	assert isclose(sinc_t[1] - sinc_t[0], sinc_scale / ratio)
	k = np.sinc(sinc_t) * window(window_size)
	k *= sinc_scale

	#print(k)  # DEBUG

	# TODO: there's probably a better way to do this
	y_zero_stuffed = []
	for n, xn in enumerate(x):
		if n > 0:
			y_zero_stuffed += [0.0] * (ratio - 1)
		#y_zero_stuffed += [xn * ratio]
		y_zero_stuffed += [xn]
	y_zero_stuffed = np.array(y_zero_stuffed)

	y = scipy.signal.convolve(y_zero_stuffed, k, mode='same')

	assert len(y) == len(t_out)

	return y, t_out


def upsample_lanczos(x: np.ndarray, ratio: int, window_half_width_samples: int, sinc_scale=1.0) -> Tuple[np.ndarray, np.ndarray]:
	def window(n):
		return np.sinc(np.linspace(-1, 1, n, endpoint=True))
	return upsample_windowed_sinc(x, ratio, window_half_width_samples, window, sinc_scale=sinc_scale)


def plot_upsampler(interpolation_func: Callable, title: str, num_samples=16, ratio=8):

	print(f'Plotting upsampler "{title}"')

	#num_samples = 128  # DEBUG
	#num_samples_impulse = num_samples
	num_samples_impulse = max(num_samples, 128)

	# TODO: more complex signals, with multiple harmonic & inharmonic components, plus noise
	# TODO: sawtooth

	fig, subplots = plt.subplots(3, 4)
	fig.suptitle(f'Interpolator: {title}, ratio {ratio}')

	ax_ir = subplots[0][0]
	ax_ir_baseband  = subplots[1][0]
	ax_ir_upband  = subplots[2][0]

	ax_sr = subplots[0][1]
	ax_sr_baseband  = subplots[1][1]
	ax_sr_upband  = subplots[2][1]

	ax_sin = subplots[0][2]
	ax_sin_baseband = subplots[1][2]
	ax_sin_upband = subplots[2][2]

	ax_linearity_check = subplots[0][3]

	# Linearity check

	x = np.zeros(num_samples)
	x[num_samples // 2:] = 1.0

	y1, t_up_1 = interpolation_func(x, 4)
	
	y2, _ = interpolation_func(x, 2)
	y2, t_up_2 = interpolation_func(y2, 2)
	t_up_2 *= 0.5

	assert len(t_up_1) == len(t_up_2)
	assert all(isclose(v1, v2) for (v1, v2) in zip(t_up_1, t_up_2))
	t_up = t_up_1

	err = np.abs(y2 - y1)

	ax_linearity_check.set_title('Linearity check')
	ax_linearity_check.plot(t_up, y1, '.-', label='2 -> 2')
	ax_linearity_check.plot(t_up, y2, '.-', label='4')
	ax_linearity_check.plot(t_up, err, 'r-', label='Error', zorder=-1)
	ax_linearity_check.grid()
	ax_linearity_check.legend()

	# Impulse

	x = np.zeros(num_samples_impulse - 1)
	x[((num_samples_impulse - 1) // 2)] = 1.0

	y, t_up = interpolation_func(x, ratio)

	ax_ir.set_title('Impulse response')
	ax_ir.plot(x, '.', label='Original', zorder=2)
	ax_ir.plot(t_up, y, '.-', label='Upsampled', zorder=1)
	ax_ir.grid()

	fft = do_fft(y, sample_rate=ratio, min_dB=-200, window=False, normalize=False, scale=1/ratio)
	ax_ir_baseband.plot(fft.bin_centers, fft.magnitude_dB)
	ax_ir_baseband.grid()
	ax_ir_baseband.set_xlim([0., 0.5])
	ax_ir_baseband.set_ylim([-12, 3])
	ax_ir_baseband.set_yticks([-12, -9, -6, -3, 0, 3])
	ax_ir_baseband.set_title('Baseband')

	ax_ir_upband.plot(fft.bin_centers, fft.magnitude_dB)

	# Step

	x = np.zeros(num_samples)
	x[num_samples // 2:] = 1.0

	y, t_up = interpolation_func(x, ratio)

	ax_sr.set_title('Step response')
	ax_sr.plot(x, '.', label='Original', zorder=2)
	ax_sr.plot(t_up, y, '.-', label='Upsampled', zorder=1)
	ax_sr.grid()

	fft = do_fft(y, sample_rate=ratio, min_dB=-200, window=False, normalize=False, scale=2/(ratio * num_samples))
	ax_sr_baseband.plot(fft.bin_centers, fft.magnitude_dB)
	ax_sr_baseband.grid()
	ax_sr_baseband.set_xlim([0., 0.5])
	ax_sr_baseband.set_ylim([-24, 0])
	ax_sr_baseband.set_title('Baseband')

	ax_sr_upband.plot(fft.bin_centers, fft.magnitude_dB)

	# Sine

	x = gen_sine(1/7, n_samp=num_samples*7)
	ax_sin.plot(x, '.', label='Original 1/7', zorder=4)
	y, t_up = interpolation_func(x, ratio)
	fft = do_fft(y, sample_rate=ratio, min_dB=-200, window=np.hanning, normalize=True, scale=2*np.sqrt(2))
	line, = ax_sin.plot(t_up, y, '-', label='Upsampled 1/7', zorder=2)
	ax_sin_baseband.plot(fft.bin_centers, fft.magnitude_dB, label='Upsampled 1/7', color=line.get_color())
	ax_sin_upband.plot(fft.bin_centers, fft.magnitude_dB, label='Upsampled 1/7', color=line.get_color())

	x = gen_sine(0.25, n_samp=num_samples*4)
	ax_sin.plot(x, '.', label='Original 1/4', zorder=3)
	y, t_up = interpolation_func(x, ratio)
	fft = do_fft(y, sample_rate=ratio, min_dB=-200, window=np.hanning, normalize=True, scale=2*np.sqrt(2))
	line, = ax_sin.plot(t_up, y, '-', label='Upsampled 1/4', zorder=1)
	ax_sin_baseband.plot(fft.bin_centers, fft.magnitude_dB, label='Upsampled 1/4', color=line.get_color())
	ax_sin_upband.plot(fft.bin_centers, fft.magnitude_dB, label='Upsampled 1/4', color=line.get_color())

	ax_sin.grid()
	ax_sin.set_title('Sine response')
	ax_sin.set_xlim([0, 7*2])

	ax_sin_baseband.grid()
	ax_sin_baseband.set_xlim([0.0, 0.5])
	ax_sin_baseband.set_ylim([-24, 0])
	ax_sin_baseband.set_title('Baseband')
	ax_sin_baseband.legend()

	for ax in [ax_ir_upband, ax_sr_upband, ax_sin_upband]:
		ax.grid()
		ax.set_xlim([0.5, 0.5*ratio])
		ax.set_ylim([-160, 0])
		ax.set_title('Upsampled band')


def plot(args):
	plot_upsampler(upsample_fft, 'FFT', num_samples=32)

	if args.stft:
		plot_upsampler(lambda x, r: upsample_stft4(x, r, np.hamming), 'STFT4, hamming', num_samples=16)
		plot_upsampler(lambda x, r: upsample_stft4(x, r, np.ones), 'STFT4, rect', num_samples=16)
		plot_upsampler(lambda x, r: upsample_stft(x, r, 6), 'STFT6', num_samples=32)
		plot_upsampler(lambda x, r: upsample_stft(x, r, 8, np.hamming), 'STFT8, hamming', num_samples=32)
		plot_upsampler(lambda x, r: upsample_stft(x, r, 8, np.ones), 'STFT8, rect', num_samples=32)
		plot_upsampler(lambda x, r: upsample_stft(x, r, 16, np.hamming), 'STFT16, hamming', num_samples=64)
		plot_upsampler(lambda x, r: upsample_stft(x, r, 16, np.ones), 'STFT16, rect', num_samples=64)
	elif args.sinc:
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 1, np.ones), 'Boxcar 1', num_samples=16)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 2, np.ones), 'Boxcar 2', num_samples=16)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 4, np.ones), 'Boxcar 4', num_samples=16)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 6, np.ones), 'Boxcar 6', num_samples=32)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, np.ones), 'Boxcar 8', num_samples=32)
		#plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 1, np.hamming), 'Hamming 1', num_samples=16)
		#plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 2, np.hamming), 'Hamming 2', num_samples=16)
		#plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 4, np.hamming), 'Hamming 4', num_samples=16)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, np.hamming), 'Hamming 8', num_samples=32)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, np.hanning), 'Hann 8', num_samples=32)
		#plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 1, np.blackman), 'Blackman 1', num_samples=16)
		#plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 2, np.blackman), 'Blackman 2', num_samples=16)
		#plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 4, np.blackman), 'Blackman 4', num_samples=16)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, np.blackman), 'Blackman 8', num_samples=32)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, scipy.signal.windows.flattop), 'Flat top 8', num_samples=32)
		#plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, scipy.signal.windows.flattop, sinc_scale=2.0), 'Flat top 8, scale 2', num_samples=32)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, scipy.signal.windows.flattop, sinc_scale=0.5), 'Flat top 8, scale 0.5', num_samples=32)
		#plot_upsampler(lambda x, r: upsample_lanczos(x, r, 1), 'Lanczos 1', num_samples=16)
		plot_upsampler(lambda x, r: upsample_lanczos(x, r, 2), 'Lanczos 2', num_samples=16)
		plot_upsampler(lambda x, r: upsample_lanczos(x, r, 4), 'Lanczos 4', num_samples=16)
		plot_upsampler(lambda x, r: upsample_lanczos(x, r, 6), 'Lanczos 6', num_samples=32)
		plot_upsampler(lambda x, r: upsample_lanczos(x, r, 8), 'Lanczos 8', num_samples=32)
		plot_upsampler(lambda x, r: upsample_lanczos(x, r, 8, sinc_scale=0.5), 'Lanczos 8, scale 0.5', num_samples=32)
	else:
		plot_upsampler(upsample_nearest, 'Nearest', num_samples=8)
		plot_upsampler(upsample_lin, 'Linear', num_samples=8)
		plot_upsampler(upsample_cos, 'Cosine', num_samples=8)
		plot_upsampler(upsample_quadratic, 'Quadratic', num_samples=8)
		plot_upsampler(upsample_spline, 'Spline', num_samples=16)
		plot_upsampler(upsample_pchip, 'PCHIP', num_samples=8)
		plot_upsampler(upsample_akima, 'Akima', num_samples=8)
		plot_upsampler(lambda x, ratio: upsample_lagrange(x, ratio, 4), 'Lagrange 4', num_samples=16)
		plot_upsampler(lambda x, ratio: upsample_lagrange(x, ratio, 6), 'Lagrange 6', num_samples=16)
		plot_upsampler(lambda x, ratio: upsample_lagrange(x, ratio, 8), 'Lagrange 8', num_samples=32)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, np.ones), 'Boxcar 8', num_samples=32)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, np.hamming), 'Hamming 8', num_samples=32)
		plot_upsampler(lambda x, r: upsample_windowed_sinc(x, r, 8, scipy.signal.windows.flattop), 'Flat top 8', num_samples=32)
		plot_upsampler(lambda x, r: upsample_lanczos(x, r, 8), 'Lanczos 8', num_samples=32)
		plot_upsampler(upsample_one_pole_allpass_polyphase, '1-pole allpass polyphase', num_samples=16)
		plot_upsampler(lambda x, r: upsample_one_pole_allpass_polyphase_then_lerp(x, r, 2), 'Allpass polyphase 2 then lerp', num_samples=16)
		plot_upsampler(lambda x, r: upsample_one_pole_allpass_polyphase_then_lerp(x, r, 4), 'Allpass polyphase 4 then lerp', num_samples=16)
		#plot_upsampler(upsample_sin_fit, '3-point sinusoid fit', num_samples=8)

	plt.show()


def main(args):
	plot(args)


def get_parser():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('--sinc', action='store_true', help='Windows sinc interpolators')
	parser.add_argument('--stft', action='store_true', help='STFT based interpolators')
	return parser
