#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

from utils import *

def plot_fft(data, sample_rate, nfft=None, log=True, freq_range=(20., 20000.), label=None):

	if nfft is not None:
		data_fft = np.fft.fft(data, n=nfft)
	else:
		data_fft = np.fft.fft(data)

	fft_len = len(data_fft)
	data_len = fft_len // 2
	data_fft = data_fft[0:data_len]

	data_fft = to_dB(np.abs(data_fft))

	f = np.linspace(0, sample_rate/2.0, num=data_len, endpoint=False)
	
	# Make plot based on bin center instead of edge
	f += (f[1] - f[0]) / 2.0

	if log:
		plt.semilogx(f, data_fft, label=label)
	else:
		plt.plot(f, data_fft, label=label)

	plt.xlim(freq_range)
	plt.grid()
