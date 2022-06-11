#!/usr/bin/env python3

import datetime
from multiprocessing import Pool
import time
from typing import Optional, Callable, Tuple, Union, Iterable

import numpy as np
import scipy.signal

from filters.filter_base import FilterBase, ResonantFilterBase
from generation import signal_generation
from utils import wavfile
from utils.utils import print_timestamped


"""
TODO: more test cases

Test cases to determine best algorithms/approximations/estimates:

Sawtooth

Combination of sines
	One paper I found used 110 Hz + 155 Hz, which seems good (IM is at 75/200, HD2 at 220/310)

Variety of gain levels

Variety of input frequencies

Variety cutoff frequencies

Instant transitions vs bandlimited

Square waves
	good case because they have fast transitions and are always at one end or the
	other (in heavy distortion region), yet the distortion wouldn't affect the
	wave if it weren't for the lowpass filtering

Different resonance levels, including self-osc

Audio-rate FM
"""


def _downsample(x: np.ndarray, downsample_factor: int, max_fft_size=16384) -> np.ndarray:

	# TODO: figure out optimal max_fft_size for performance (this isn't really a performance bottleneck here though)

	# TODO: handle some of these cases (and throw better exceptions for the ones not handling)
	assert downsample_factor < max_fft_size
	assert max_fft_size % downsample_factor == 0
	assert len(x) % downsample_factor == 0

	num_samp_out = len(x) // downsample_factor

	y = np.zeros(num_samp_out, dtype=x.dtype)

	max_fft_size_down = max_fft_size // downsample_factor

	start_idx = 0
	while start_idx < len(x):

		end_idx = min(start_idx + max_fft_size, len(x))

		y_start_idx = start_idx // downsample_factor
		y_end_idx = min(y_start_idx + max_fft_size_down, len(y))

		#print(f'{downsample_factor=} {start_idx=} {end_idx=} {y_start_idx=} {y_end_idx=}'

		xw = x[start_idx:end_idx]
		yw = scipy.signal.resample(xw, y_end_idx - y_start_idx)
		y[y_start_idx:y_end_idx] = yw

		start_idx += max_fft_size

	return y


def generate_test_signal(num_samp: int, freq_Hz: float, sample_rate_Hz: float, square=False) -> np.ndarray:

	if square:
		return signal_generation.gen_square(freq_norm=(freq_Hz / sample_rate_Hz), n_samp=num_samp)

	detune_Hz = 0.5

	audible_noise_ampl = 0.1

	# TODO: more realistic noise that better simulates real noise - e.g. white + 1/f + "oscillator drift" sim
	# Even just filtering it with filters.one_pole.LeakyIntegrator might be close enough to realistic drift?
	freq_white_noise_ampl_Hz = 1.0

	# But for now, just simulate drift by summing some slow, non harmonically related sine waves
	drift_1 = \
		0.3 * signal_generation.gen_sine(0.87 / sample_rate_Hz, start_phase=0.11, n_samp=num_samp) + \
		0.2 * signal_generation.gen_sine(1.03 / sample_rate_Hz, start_phase=0.35, n_samp=num_samp) + \
		0.1 * signal_generation.gen_sine(1.31 / sample_rate_Hz, start_phase=0.72, n_samp=num_samp)
	drift_2 = \
		0.3 * signal_generation.gen_sine(0.71 / sample_rate_Hz, start_phase=0.71, n_samp=num_samp) + \
		0.2 * signal_generation.gen_sine(1.20 / sample_rate_Hz, start_phase=0.67, n_samp=num_samp) + \
		0.1 * signal_generation.gen_sine(1.47 / sample_rate_Hz, start_phase=0.89, n_samp=num_samp)

	# TODO: use deterministic noise so this is the same every time
	white_noise_1 = signal_generation.gen_noise(num_samp, gaussian=True, amp=freq_white_noise_ampl_Hz)
	white_noise_2 = signal_generation.gen_noise(num_samp, gaussian=True, amp=freq_white_noise_ampl_Hz)

	base_freq_1 = freq_Hz - 0.5*detune_Hz
	base_freq_2 = freq_Hz + 0.5*detune_Hz

	freq_1 = np.ones(num_samp) * base_freq_1 + white_noise_1 + drift_1
	freq_2 = np.ones(num_samp) * base_freq_2 + white_noise_2 + drift_2

	freq_1 /= sample_rate_Hz
	freq_2 /= sample_rate_Hz

	phase_1 = signal_generation.gen_phase(freq_1, n_samp=num_samp, start_phase=0)
	phase_2 = signal_generation.gen_phase(freq_2, n_samp=num_samp, start_phase=0.11)

	# TODO: use PolyBLEPs
	# TODO: while not using polyBLEPs, pre-filter out some high frequency content
	saw_1 = phase_1 - 0.5
	saw_2 = phase_2 - 0.5

	x = saw_1 + saw_2 + signal_generation.gen_noise(num_samp, gaussian=True, amp=audible_noise_ampl)

	return x


def _test_filter_at_gain_res(
		filter_constructor: Callable[..., Union[FilterBase, ResonantFilterBase]],
		x: np.ndarray,
		wc_start: float,
		wc_end: float,
		gain = 1.0,
		resonance: Optional[float] = None,
		normalize_gain = True,
		) -> Tuple[np.ndarray, float]:

	assert gain != 0.0

	xg = x * gain

	if resonance is None:
		filter = filter_constructor(wc_start)
	else:
		filter = filter_constructor(wc_start, resonance=resonance)
		filter.set_resonance(resonance)

	start = time.monotonic()

	y = filter.process_freq_sweep(xg, wc_start=wc_start, wc_end=wc_end, log=True)

	duration_seconds = time.monotonic() - start

	# TODO: not just a sweep, also try audio-rate modulation

	if normalize_gain:
		y /= gain

	# TODO: also apply very brief envelope to start & end to prevent clicks

	return y, duration_seconds


def _test_filter(
		filter_constructor: Callable[..., Union[FilterBase, ResonantFilterBase]],
		filename: str,
		resonant: bool,
		gain_resonance_pairs: Iterable[Tuple[float, Optional[float]]],
		sample_rate_out=48000,
		oversampling=4,
		sig_freq=55.0,
		fc_start=20000.,
		fc_end=20.0,
		name: Optional[str]=None,
		sweep_gain=True,
		pool: Optional[Pool]=None,
		) -> None:

	if name is None:
		name = filename

	internal_sample_rate = sample_rate_out * oversampling
	time_per_gain_seconds = 5.0
	num_samp_per_gain = int(round(time_per_gain_seconds * internal_sample_rate))

	gain_resonance_square_pairs = [
		(gain, resonance, False) for gain, resonance in gain_resonance_pairs
	] + [
		(gain, resonance, True) for gain, resonance in gain_resonance_pairs
	]

	num_samp = num_samp_per_gain * len(gain_resonance_square_pairs)
	normalize_gain = True

	x_saw = generate_test_signal(num_samp=num_samp_per_gain, freq_Hz=sig_freq, sample_rate_Hz=internal_sample_rate, square=False)
	x_squ = generate_test_signal(num_samp=num_samp_per_gain, freq_Hz=sig_freq, sample_rate_Hz=internal_sample_rate, square=True)
	y = np.zeros(num_samp)

	if pool is None:

		for gain_idx, (gain, resonance, square) in enumerate(gain_resonance_square_pairs):

			square_saw_name = 'square' if square else 'saw'

			assert gain != 0.0

			if resonance is not None:
				assert resonance >= 0.0
				print_timestamped(f'Processing "{name}" ({square_saw_name}) at gain {gain}, resonance {resonance}...')
			else:
				print_timestamped(f'Processing "{name}" ({square_saw_name}) at gain {gain}...')

			yg, duration_seconds = _test_filter_at_gain_res(
				filter_constructor=filter_constructor,
				x=(x_squ if square else x_saw),
				wc_start=fc_start / internal_sample_rate,
				wc_end=fc_end / internal_sample_rate,
				resonance=resonance,
				gain=gain,
				normalize_gain=normalize_gain,
			)

			length_seconds = len(yg) / internal_sample_rate
			real_time_scale = duration_seconds / length_seconds
			print_timestamped(
				f'Processing {length_seconds:.3f} seconds '
				f'at {internal_sample_rate / 1000:g} kHz '
				f'took {duration_seconds:.3f} seconds '
				f'= {real_time_scale:.2f}x real-time')

			start_idx = gain_idx * num_samp_per_gain
			end_idx = start_idx + num_samp_per_gain

			y[start_idx:end_idx] = yg

	else:
		async_results = []

		for gain, resonance, square in gain_resonance_square_pairs:

			assert gain != 0.0

			square_saw_name = 'square' if square else 'saw'

			if resonance is not None:
				assert resonance >= 0.0
				print_timestamped(f'Starting processing "{name}" ({square_saw_name}) at gain {gain}, resonance {resonance}...')
			else:
				print_timestamped(f'Starting processing "{name}" ({square_saw_name}) at gain {gain}...')

			async_result = pool.apply_async(
				_test_filter_at_gain_res,
				kwds=dict(
					filter_constructor=filter_constructor,
					x=(x_squ if square else x_saw),
					wc_start=fc_start / internal_sample_rate,
					wc_end=fc_end / internal_sample_rate,
					resonance=resonance,
					gain=gain,
					normalize_gain=normalize_gain,
				)
			)
			async_results.append(async_result)

		for gain_idx, (async_result, (gain, resonance, square)) in enumerate(zip(async_results, gain_resonance_square_pairs)):
			yg, duration_seconds = async_result.get()

			square_saw_name = 'square' if square else 'saw'

			length_seconds = len(yg) / internal_sample_rate
			real_time_scale = duration_seconds / length_seconds
			if resonance is not None:
				print_timestamped(
					f'Processing "{name}" ({square_saw_name}) '
					f'at gain {gain}, '
					f'resonance {resonance} '
					f'for {length_seconds:.3f} seconds '
					f'at {internal_sample_rate / 1000:g} kHz '
					f'took {duration_seconds:.3f} seconds '
					f'= {real_time_scale:.2f}x real-time'
				)
			else:
				print_timestamped(
					f'Processing "{name}" ({square_saw_name}) '
					f'at gain {gain} '
					f'for {length_seconds:.3f} seconds '
					f'at {internal_sample_rate / 1000:g} kHz '
					f'took {duration_seconds:.3f} seconds '
					f'= {real_time_scale:.2f}x real-time'
				)

			start_idx = gain_idx * num_samp_per_gain
			end_idx = start_idx + num_samp_per_gain

			y[start_idx:end_idx] = yg

	print_timestamped(f'Downsampling {name}...')

	y = _downsample(y, oversampling)

	y_peak = max(np.amax(y), -np.amin(y))

	assert y_peak >= 0
	if y_peak == 0:
		raise Exception('Filter did not return any signal!')

	y /= y_peak

	print_timestamped(f'Saving {filename}')
	wavfile.export_wavfile(y, sample_rate=sample_rate_out, filename=filename, allow_overwrite=True)


def test_non_resonant_filter(
		filter_constructor: Callable[..., FilterBase],
		filename: str,
		sample_rate_out=48000,
		oversampling=4,
		name: Optional[str]=None,
		sweep_gain=True,
		pool: Optional[Pool]=None,
		) -> None:

	if sweep_gain:
		gain_resonance_pairs = [
			(0.1, None),
			(1.0, None),
			(10.0, None),
		]
	else:
		gain_resonance_pairs = [1.0, None]

	_test_filter(
		resonant=False,
		filter_constructor=filter_constructor,
		name=name,
		filename=filename,
		gain_resonance_pairs=gain_resonance_pairs,
		sample_rate_out=sample_rate_out,
		oversampling=oversampling,
		fc_start=32000.,
		fc_end=16.125,
		sweep_gain=sweep_gain,
		pool=pool,
	)


def test_resonant_filter(
		filter_constructor: Callable[..., ResonantFilterBase],
		filename: str,
		sample_rate_out=48000,
		oversampling=4,
		self_oscillation=False,
		name: Optional[str]=None,
		sweep_gain=True,
		pool: Optional[Pool]=None,
		) -> None:

	if not sweep_gain:
		gain_resonance_pairs = [
			(1.0, 0.0),
			(1.0, 0.25),
			(1.0, 0.5),
			(1.0, 0.75),
			(1.0, 0.95),
			(1.0, 1.01) if self_oscillation else None,
		]
	else:
		# If sweeping gain, then presumably this must be a nonlinear filter
		# So prioritize low gains, which should be more linear
		gain_resonance_pairs = [
			(0.1, 0.0),
			(0.1, 0.25),
			(0.1, 0.5),
			(0.1, 0.75),
			(0.1, 0.95),
			(0.1, 1.01) if self_oscillation else None,
			(1.0, 0.0),
			(1.0, 0.5),
			(1.0, 1.01) if self_oscillation else None,
			(10.0, 0.0),
			(10.0, 0.5),
			(10.0, 1.01) if self_oscillation else None,
		]
	gain_resonance_pairs = [pair for pair in gain_resonance_pairs if pair is not None]

	_test_filter(
		resonant=True,
		filter_constructor=filter_constructor,
		name=name,
		filename=filename,
		gain_resonance_pairs=gain_resonance_pairs,
		sample_rate_out=sample_rate_out,
		oversampling=oversampling,
		fc_start=32000.,
		fc_end=31.25,
		sweep_gain=sweep_gain,
		pool=pool,
	)
