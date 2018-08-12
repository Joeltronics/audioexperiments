#!/usr/bin/env python3


import numpy as np
import scipy.io.wavfile
import os.path
from typing import Tuple


def import_wavfile(filename) -> Tuple[np.ndarray, int]:
	"""Import wav file and normalize to float values in range [-1, 1)

	:param filename:
	:return: data, sample rate (Hz)
	"""
	if not os.path.exists(filename):
		raise FileNotFoundError(filename)

	sample_rate, data = scipy.io.wavfile.read(filename)

	# Convert to range (-1,1)

	if data.dtype == np.dtype('int8'):
		data = data.astype('float') / 128.0
	elif data.dtype == np.dtype('uint8'):
		data = (data.astype('float') - 128.0) / 128.0
	elif data.dtype == np.dtype('int16'):
		data = data.astype('float') / float(2 ** 15)
	elif data.dtype == np.dtype('int32'):
		data = data.astype('float') / float(2 ** 31)
	elif data.dtype == np.dtype('float'):
		pass
	else:
		raise ValueError('Unknown data type: %s' % data.dtype)

	return data, sample_rate


def export_wavfile(data: np.ndarray, sample_rate: int, filename, allow_overwrite=False) -> None:
	"""Save float array to 16-bit wav file. No dithering.

	:param data: np.array of type float, in range [-1, 1]
	:param sample_rate: in Hz
	:param filename:
	"""

	if data.dtype != np.dtype('float'):
		raise ValueError('Array data type must be float')

	if (not allow_overwrite) and os.path.exists(filename):
		raise FileExistsError(filename)

	data_clipped = np.clip(data, -1.0, 1.0)

	# It appears numpy rounds data toward zero
	gain = 2 ** 15 - 1
	data_16 = np.array(data_clipped * gain, dtype=np.int16)

	scipy.io.wavfile.write(filename, sample_rate, data_16)
