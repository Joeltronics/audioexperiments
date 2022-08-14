#!/usr/bin/env python3

"""
Partially based on public-domain code by Daniel Werner

https://www.experimentalscene.com/articles/minbleps.php
mirror: https://gist.github.com/endolith/2654862

See also:
https://www.katjaas.nl/minimumphase/minimumphase.html
"""


import numpy as np
from matplotlib import pyplot as plt

from analysis.cepstrum import get_real_cepstrum_from_fft
from utils.fft import do_fft


def minimum_phase(real_cepstrum: np.ndarray) -> np.ndarray:
	n = len(real_cepstrum)
	nd2 = n // 2

	real_time = np.zeros(n)

	real_time[0] = real_cepstrum[0]
	real_time[1:nd2] = 2 * real_cepstrum[1:nd2]
	if (n % 2) == 0:
		real_time[nd2] = real_cepstrum[nd2]

	complex_freq = np.fft.fft(real_time)
	complex_freq = np.exp(complex_freq)
	return np.real(np.fft.ifft(complex_freq))


def do_plot(size = 256, t0 = 8):

	t = np.linspace(-t0, t0, size, endpoint=True)
	x = np.sinc(t)
	w = np.hamming(size)
	x *= w

	fft_x = do_fft(x, window=False)
	cx = get_real_cepstrum_from_fft(fft_x.full_complex_result)

	y = minimum_phase(cx)
	fft_y = do_fft(y, window=False)
	cy = get_real_cepstrum_from_fft(fft_y.full_complex_result)

	fig, subplots = plt.subplots(3, 1)

	ax_ir, ax_mag, ax_cepstrum = subplots
	ax_phase = ax_mag.twinx()

	ax_ir.plot(x, label='Linear phase')
	ax_ir.plot(y, label='Minimum phase')

	ax_mag.semilogx(fft_x.bin_centers, fft_x.magnitude_dB)
	ax_mag.semilogx(fft_y.bin_centers, fft_y.magnitude_dB)

	ax_phase.semilogx(fft_x.bin_centers, fft_x.phase_degrees, '--')
	ax_phase.semilogx(fft_y.bin_centers, fft_y.phase_degrees, '--')

	ax_cepstrum.plot(fft_x.bin_centers, cx[:size//2])
	ax_cepstrum.plot(fft_y.bin_centers, cy[:size//2])

	ax_ir.legend()
	ax_ir.set_ylabel('IR')
	ax_mag.set_ylabel('FFT')
	ax_cepstrum.set_ylabel('Cepstrum')

	ax_mag.set_ylim([-72, 24])
	ax_mag.set_yticks([-72, -48, -24, 0, 24])

	ax_phase.set_ylim([-180, 180])
	ax_phase.set_yticks([-180, -90, 0, 90, 180])

	for ax in subplots:
		ax.grid()


def plot(args):
	do_plot(256, 8)
	do_plot(255, 8)
	do_plot(16, 4)
	do_plot(17, 4)
	plt.show()


def main(args):
	plot(args)
