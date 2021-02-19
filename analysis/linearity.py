#!/usr/bin/env python3

import math
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from analysis import time_domain_response
from utils import utils
from unit_test import unit_test
from processor import ProcessorBase
from generation import signal_generation


DEFAULT_CHECK_LINEAR_EPS = 0.00001


_unit_tests = []


def _check_linear_homogeneity(system: ProcessorBase, n_samp: int, eps=DEFAULT_CHECK_LINEAR_EPS) -> bool:

	# f(a*x) = a*f(x)

	start_freq = 1 / 96000
	end_freq = 0.5
	x = signal_generation.gen_freq_sweep_sine(start_freq, end_freq, n_samp=n_samp, log=True, start_phase=0.25)

	system.reset()
	fx = system(x)

	for ampl in [0.0001, 1000.0]:

		system.reset()
		fax = system(ampl * x)  # f(a * x)

		afx = ampl * fx  # a * f(x)

		if not utils.approx_equal(afx, fax, eps_abs=(eps * ampl)):
			return False

	return True


def _check_linear_additivity(system: ProcessorBase, n_samp: int, eps=DEFAULT_CHECK_LINEAR_EPS) -> bool:

	# f(x1 + x2) = f(x1) + f(x2)

	start_freq = 1 / 96000
	end_freq = 0.5
	x1 = signal_generation.gen_freq_sweep_sine(start_freq, end_freq, n_samp=n_samp, log=True, start_phase=0.25)
	x2 = signal_generation.gen_freq_sweep_sine(end_freq, start_freq, n_samp=n_samp, log=True, start_phase=0.25)

	system.reset()
	fx1 = system(x1)
	system.reset()
	fx2 = system(x2)
	fx1_fx2 = fx1 + fx2

	x1x2 = x1 + x2
	system.reset()
	f_x1x2 = system(x1x2)

	return utils.approx_equal(fx1_fx2, f_x1x2, eps_abs=eps)


def check_linear_superposition(system: ProcessorBase, n_samp: int, eps=DEFAULT_CHECK_LINEAR_EPS) -> bool:
	return \
		_check_linear_homogeneity(system, n_samp=n_samp, eps=eps) and \
		_check_linear_additivity(system, n_samp=n_samp, eps=eps)


def check_linear_impulse_step_ramp(system: ProcessorBase, n_samp: int, amplitude=10.0, eps=DEFAULT_CHECK_LINEAR_EPS, verbose=False) -> bool:
	"""
	Check if system is linear by comparing impulse, step, and ramp responses

	:param system:
	:param n_samp:
	:param amplitude:
	:param eps:
	:param verbose:
	:return:
	"""

	# Step response here is actually -f(-step); if system is linear, this should be identical to f(step)
	# This way, we catch systems where positive & negative have different response - e.g. y = abs(x)
	ir = time_domain_response.get_impulse_response(system, n_samp, amplitude, reset=True)
	sr = -time_domain_response.get_step_response(system, n_samp, amplitude, reset=True, negative=True)
	rr = time_domain_response.get_ramp_response(system, n_samp, slope=amplitude / n_samp, reset=True) * n_samp

	dsr = np.diff(sr)
	drr = np.diff(rr)
	ddrr = np.diff(drr)

	impulse_step_err = dsr - ir[1:]
	impulse_ramp_err = ddrr - ir[2:]
	step_ramp_err = drr - sr[1:]

	impulse_step_err = np.amax(np.abs(impulse_step_err))
	impulse_ramp_err = np.amax(np.abs(impulse_ramp_err))
	step_ramp_err = np.amax(np.abs(step_ramp_err))

	linear_step_impulse = impulse_step_err < eps
	linear_ramp_impulse = impulse_ramp_err < eps
	linear_ramp_step = step_ramp_err < eps

	if verbose:
		print('Max errors: dstep-impulse %f, dramp-step %f, d2ramp-impulse %f' % (
			impulse_step_err,
			step_ramp_err,
			impulse_ramp_err,
		))

	return linear_step_impulse and linear_ramp_impulse and linear_ramp_step


def check_linear(system: ProcessorBase, n_samp: int, eps=DEFAULT_CHECK_LINEAR_EPS, full=False, verbose=False) -> bool:
	return check_linear_superposition(system, n_samp=n_samp, eps=eps) and (
		check_linear_impulse_step_ramp(system, n_samp=n_samp, eps=eps, verbose=verbose) if full else True
	)


def _test_check_linear():
	from filters import one_pole
	from filters import biquad
	from overdrive import overdrive
	from processor import GainWrapper, CascadedProcessors, GainProcessor

	wc = 1000. / 96000.
	Q = 1.5
	n_samp = 16384

	# TODO: test abs(x)
	# TODO: test other cases where positive is linear but negative isn't

	unit_test.test_equal(check_linear(CascadedProcessors([]), n_samp, full=True), True)
	unit_test.test_equal(check_linear(one_pole.BasicOnePole(wc=wc), n_samp, full=True), True)
	unit_test.test_equal(check_linear(one_pole.BasicOnePole(wc=wc), n_samp, full=True), True)
	unit_test.test_equal(check_linear(one_pole.BasicOnePoleHighpass(wc=wc), n_samp, full=True), True)
	unit_test.test_equal(check_linear(biquad.BiquadLowpass(wc=wc, Q=Q), n_samp, full=True), True)

	unit_test.test_equal(check_linear(overdrive.TanhProcessor(), n_samp, full=True), False)
	unit_test.test_equal(check_linear(GainWrapper(overdrive.TanhProcessor(), 10.), n_samp, full=True), False)
	unit_test.test_equal(check_linear(GainWrapper(overdrive.TanhProcessor(), 0.1), n_samp, full=True), False)
	unit_test.test_equal(check_linear(overdrive.Squarizer(), n_samp, full=True), False)
	unit_test.test_equal(check_linear(CascadedProcessors([overdrive.Squarizer(), GainProcessor(0.1)]), n_samp, full=True), False)
	unit_test.test_equal(check_linear(GainWrapper(biquad.Rossum92Biquad(wc=wc, Q=Q), 10.), n_samp, full=True), False)


_unit_tests.append(_test_check_linear)


def test(verbose=False, long=False):
	return unit_test.run_unit_tests(_unit_tests, verbose=verbose)


def _do_main(verbose=False):
	from matplotlib import pyplot as plt
	from filters import one_pole
	from filters import biquad
	from overdrive import overdrive
	from processor import GainWrapper, CascadedProcessors, GainProcessor
	from utils.utils import to_dB

	sample_rate = 96000
	cutoff = 1000
	Q = 2.0

	n_samp = 16384  # Number of samples for IR/Step/Ramp stuff; not used for sine sweep. Must be at least 9600 for 10 Hz @ 96 kHz

	wc = cutoff / sample_rate

	filters = [
		("pass-through processor", CascadedProcessors([])),

		("Basic one pole", one_pole.BasicOnePole(wc=wc)),
		("Trapz one pole", one_pole.TrapzOnePole(wc=wc)),
		("Basic one pole highpass", one_pole.BasicOnePoleHighpass(wc=wc)),
		("Biquad, Q=%g" % Q, biquad.BiquadLowpass(wc=wc, Q=Q)),

		("tanh overdrive", overdrive.TanhProcessor()),
		("tanh overdrive, 20 dB gain", GainWrapper(overdrive.TanhProcessor(), 10.)),
		("tanh overdrive, -20 dB gain", GainWrapper(overdrive.TanhProcessor(), 0.1)),
		("Squarizer", overdrive.Squarizer()),
		("Squarizer -20 dB", CascadedProcessors([overdrive.Squarizer(), GainProcessor(0.1)])),
		("One pole then tanh", CascadedProcessors([one_pole.BasicOnePole(wc=wc), overdrive.TanhProcessor(gain=2)])),
		("tanh then one pole", CascadedProcessors([overdrive.TanhProcessor(gain=2), one_pole.BasicOnePole(wc=wc)])),
		("Biquad, Q=%g, then hard clip at 1.1" % Q, CascadedProcessors([biquad.BiquadLowpass(wc=wc, Q=Q), overdrive.Clipper(gain=1.0/1.1)])),
		("Biquad, Q=%g, then hard clip at 1" % Q, CascadedProcessors([biquad.BiquadLowpass(wc=wc, Q=Q), overdrive.Clipper()])),
		("Rossum 92 Nonlinear Biquad, Q=%g, gain 10" % Q, GainWrapper(biquad.Rossum92Biquad(wc=wc, Q=Q), 10.)),
	]

	n = np.arange(n_samp)
	t = n / sample_rate

	impulse = time_domain_response.generate_impulse(n_samp, amplitude=1.0)
	step = time_domain_response.generate_step(n_samp, amplitude=1.0)
	ramp = time_domain_response.generate_ramp(n_samp, slope=1.0)

	for filter_name, filter in filters:

		print('Processing filter "%s"' % filter_name)

		linear = check_linear(filter, n_samp=n_samp)
		if verbose:
			print('Linear: %s' % linear)

		ir = time_domain_response.get_impulse_response(filter, n_samp, amplitude=1.0, reset=True)
		sr = time_domain_response.get_step_response(filter, n_samp, amplitude=1.0, reset=True)
		rr = time_domain_response.get_ramp_response(filter, n_samp, slope=1.0, reset=True)

		nir = -time_domain_response.get_impulse_response(filter, n_samp, amplitude=1.0, reset=True, negative=True)
		nsr = -time_domain_response.get_step_response(filter, n_samp, amplitude=1.0, reset=True, negative=True)
		nrr = -time_domain_response.get_ramp_response(filter, n_samp, slope=1.0, reset=True, negative=True)

		dsr = np.diff(sr)
		drr = np.diff(rr)
		ddrr = np.diff(drr)

		n_dsr = n[1:]
		n_ddrr = n[2:]

		n_impulse_step_err = n_dsr
		impulse_step_err = dsr - ir[1:]

		n_impulse_ramp_err = n_ddrr
		impulse_ramp_err = ddrr - ir[2:]

		n_step_ramp_err = n_dsr
		step_ramp_err = drr - sr[1:]

		if verbose:
			print('Max errors: dstep-impulse %f, dramp-step %f, d2ramp-impulse %f' % (
				np.amax(np.abs(impulse_step_err)),
				np.amax(np.abs(impulse_ramp_err)),
				np.amax(np.abs(step_ramp_err)),
			))

		#
		# Plotting
		#

		fig = plt.figure()
		fig.suptitle(filter_name)

		plt.subplot(4, 1, 1)
		plt.plot(n, ir, label='f(impulse)')
		plt.plot(n, nir, label='-f(-impulse)')
		plt.plot(n_dsr, dsr, label="f'(step)")
		plt.plot(n_ddrr, ddrr, label="f''(ramp)")
		plt.grid()
		plt.legend()

		plt.subplot(4, 1, 2)
		plt.plot(n, sr, label='f(step)')
		plt.plot(n, nsr, label='-f(-step)')
		plt.plot(n_dsr, drr, label="f'(ramp)")
		plt.grid()
		plt.legend()

		plt.subplot(4, 1, 3)
		#plt.plot(n, ramp, label='ramp')
		plt.plot(n, rr, label='f(ramp)')
		plt.plot(n, nrr, label='-f(-ramp)')
		plt.grid()
		plt.legend()

		plt.subplot(4, 1, 4)
		plt.semilogy(n_impulse_step_err, np.maximum(np.abs(impulse_step_err), 1e-12), label="Impulse vs step' error")
		plt.semilogy(n_impulse_ramp_err, np.maximum(np.abs(impulse_ramp_err), 1e-12), label="Impulse vs ramp'' error")
		plt.semilogy(n_step_ramp_err, np.maximum(np.abs(step_ramp_err), 1e-12), label="Step vs ramp' error")
		plt.axhline(DEFAULT_CHECK_LINEAR_EPS, color='r', label='Default linearity threshold')
		plt.grid()
		plt.legend()

	print('Showing plots')
	plt.show()


def plot(args):
	_do_main(verbose=False)


def main(args):
	_do_main(verbose=True)
