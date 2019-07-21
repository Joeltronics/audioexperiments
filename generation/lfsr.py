#!/usr/bin/env python3


from typing import List, Union, Optional
import numpy as np


def wrap_bits(val, num_bits):
	mask = 2 ** num_bits - 1
	return val & mask


def parity(val) -> np.uint32:
	"""
	XOR all bits together, i.e. determine if odd number of bits is set
	:param val: must be unsigned type, or positive value of signed type
	:return: 1 if odd number of bits set, 0 if even
	"""

	# Algorithm based on "bit twiddling hacks" algo:
	# http://www.graphics.stanford.edu/~seander/bithacks.html#ParityParallel

	if val < 0:
		raise ValueError('Must give positive value if using signed type')

	# TODO: more than 32-bit
	if val > (2**32 - 1):
		raise NotImplementedError('Parity not implemented for values > 32 bits')

	val = np.uint32(val)
	val = val ^ (val >> 16)
	val = val ^ (val >> 8)
	val = val ^ (val >> 4)
	val = val & 0xF
	return (np.uint32(0x6996) >> val) & np.uint32(0x1)


assert parity(np.uint32(0b0011)) == 0
assert parity(np.uint32(0b0110)) == 0
assert parity(np.uint32(0b0111)) == 1

assert parity(np.uint32(0x10000000)) == 1
assert parity(np.uint32(0x10000001)) == 0


class LFSR:
	internal_type = np.uint32
	max_num_bits = 8 * internal_type(0).nbytes

	def __init__(self, poly_terms: Optional[List[int]], seed: int=1, num_bits: Optional[int]=None, verbose=False):
		"""
		:param poly_terms: List of powers in LFSR polynomial (power 0 can be omitted)
		:param seed:
		:param verbose:
		"""

		seed = self.internal_type(seed)
		self.num_bits = max(poly_terms) if (num_bits is None and poly_terms is not None) else num_bits

		if self.num_bits > self.max_num_bits:
			raise TypeError('Max number of bits is %u' % self.max_num_bits)

		if poly_terms is None:
			poly_terms = []

		mask = self.internal_type(0)
		one = self.internal_type(1)
		for poly_term in poly_terms:
			if poly_term == 0:
				# Ignore 0
				continue
			mask = mask | (one << (poly_term - 1))

		self.mask = self.internal_type(mask)

		self.state = None
		self.set_state(seed)

		if verbose:
			print('LFSR, {} bits, poly: {}, mask {:b}'.format(self.num_bits, sorted(poly_terms, reverse=True), self.mask))

	def __call__(self) -> bool:
		fb = parity(self.state & self.mask)
		self.state = wrap_bits(self.state << 1 | fb, self.num_bits)
		return bool(fb)

	def get_state(self) -> int:
		return self.state

	def get_state_str(self):
		return '{0:015b}'.format(self.state)

	def set_state(self, state: int):
		state = self.internal_type(state)
		self.state = wrap_bits(state, self.num_bits)


def get_maximal_lfsr(num_bits, seed=1, verbose=False, alt=False) -> Optional[LFSR]:
	"""
	:param num_bits: Must be at least 2; currently only up to 25 is implemented
	:param seed: optional seed value
	:param verbose:
	:return: LFSR with period 2^n - 1
	"""

	n = num_bits

	if n < 2:
		raise ValueError('Must be at least 2 bits!')

	elif n in [2, 3, 4, 6, 7, 15, 22]:
		# e.g. [7, 6] or [7, 1] (either form works)
		poly = [n, 1] if alt else [n, n-1]

	elif n in [5, 11, 21]:
		# 5, 3 or 5, 2
		# 11, 9 or 11, 2
		# 21, 19 or 21, 2
		poly = [n, 2] if alt else [n, n-2]

	elif n in [10, 17, 20, 25]:
		# 10, 7 or 10, 3
		# 17, 14 or 17, 3
		# 20, 17 or 20, 3
		poly = [n, 3] if alt else [n, n-3]

	elif n == 9:
		# 9, 5
		poly = [n, 4] if alt else [n, n-4]

	elif n == 23:
		# 23, 18
		poly = [n, 5] if alt else [n, n-5]

	elif n == 18:
		# 18, 11
		poly = [n, 7] if alt else [n, n-7]

	else:
		if alt:
			return None

		if n == 8:
			poly = [8, 6, 5, 4]

		elif n == 12:
			poly = [12, 11, 10, 4]

		elif n == 16:
			poly = [16, 15, 13, 4]

		elif n == 13:
			poly = [13, 12, 11, 8]

		elif n == 14:
			poly = [14, 13, 12, 2]

		elif n == 19:
			poly = [19, 6, 2, 1]

		elif n == 24:
			poly = [24, 23, 22, 17]

		else:
			raise NotImplementedError('No maximal LFSR implemented for %u bits' % num_bits)

	lfsr = LFSR(poly, seed=seed, verbose=verbose)
	assert lfsr.num_bits == num_bits
	return lfsr


class GBLFSR(LFSR):
	"""
	LFSR based on Game Boy noise channel
	"""

	def __init__(self, short=False, seed=1, verbose=False):
		self.short = short

		"""
		In long mode, game boy LFSR is just maximal 15-bit
		
		In short mode, LFSR isn't exactly a Fibonacci LFSR
		Feedback is still taken from bits 15 & 14, but in addition to being put into bit 1 it's also put into bit 6
		(post-shift). This effectively makes its behavior identical to a 7-bit LFSR, using the upper bits.
		Note that LFSR period finder won't catch this though, as bits below 6 still contribute to self.state (even if
		they won't actually affect output)	
		"""

		if self.short:
			# polynomial doesn't really apply here, but it's effectively this
			poly_terms = [7, 6]
		else:
			poly_terms = [15, 14]

		super().__init__(poly_terms, seed=seed, num_bits=15, verbose=False)

		if verbose and self.short:
			print('GB LFSR, short, {} bits'.format(self.num_bits, self.mask))
		elif verbose:
			print('GB LFSR, {} bits, poly: {}, mask {:b}'.format(self.num_bits, sorted(poly_terms, reverse=True), self.mask))

		if self.short:
			self.__call__ = self._call_short

	def _call_short(self):
		# Code based on MAME gb.cpp:
		# https://github.com/mamedev/mame/blob/ec3caa98bdcab04b0cb90b3cf1c1eb740433dfd6/src/devices/sound/gb.cpp

		s = self.state

		fb = ((s >> 1) ^ (s)) & 1
		s = (s >> 1) | (fb << 14)
		s = (s & ~(1 << 6)) | (fb << 6)

		self.state = s

		assert 0 <= self.state < 32768

		return bool(1 - (self.state & 1))


class NESLFSR(LFSR):
	"""
	LFSR based on NES noise channel
	"""

	def __init__(self, short=False, extra_short=False, seed=None, verbose=False):
		"""
		:param short: if True, will have length 93 or 31 (depending on seed; most seeds give 93 including default) instead of 32767
		:param extra_short: Uses specific seed that will give length 31 instead of 93
		:param seed: manually seed (cannot be used if extra_short is set, as extra_short overrides seed)
		:param verbose:
		"""

		if extra_short:
			short = True

		if seed is None:
			if extra_short:
				# This number was found empirically - I'm sure it's possible to derive a list of numbers that satisfy
				# this condition, but having 1 empirical number is good enough
				seed = 1847
			else:
				seed = 1
		elif extra_short:
			raise ValueError('Cannot set extra_short if providing seed!')

		self.short = short

		if self.short:
			# Equivalent to 7-bit LFSR in short mode
			poly_terms = [15, 6]
		else:
			poly_terms = [15, 1]

		super().__init__(poly_terms, seed=seed, num_bits=None, verbose=False)

		if verbose:
			print('NES LFSR, {} bits, poly: {}, mask {:b}'.format(self.num_bits, sorted(poly_terms, reverse=True), self.mask))


def find_lfsr_period(lfsr: LFSR, verbose=False, return_full_state=False):
	"""Slow, uses a lot of memory, but probably the only way to test an LFSR"""

	n_max = 2 ** lfsr.num_bits - 1

	first_state = lfsr.get_state()
	states_seen = [first_state]  # TODO: see if it would be worth preallocating to n_max
	ret_vals = [] if return_full_state else None

	for n in range(n_max):
		state_prev_str = lfsr.get_state_str()
		state_prev = lfsr.get_state()

		val = lfsr()

		state_new_str = lfsr.get_state_str()
		state_new = lfsr.get_state()

		if verbose and (n < 5 or n >= (n_max - 1)):
			#print('%i: %i, LFSR state %s -> %s' % (n, val, state_prev_str, state_new_str))
			print('%i: %i, LFSR state %i -> %i' % (n, val, state_prev, state_new))

		if ret_vals is not None:
			ret_vals.append(val)

		if state_new == first_state:
			loop_start = 0
			period = n + 1
			if verbose:
				print('%i: State is identical to first; period %i; exiting loop' % (n, period))
			break

		elif state_new in states_seen:
			# This case can happen with non-maximal LFSRs
			last = states_seen.index(state_new)
			loop_start = last
			period = n - last + 1
			if verbose:
				print('%i: State has been seen before (%i); period %i; exiting loop' % (n, last, period))
			break

		else:
			states_seen.append(state_new)
	else:
		loop_start = None
		period = None

	if not return_full_state:
		return period

	return dict(
		period=period,
		loop_start=loop_start,
		states_seen=states_seen,
		ret_vals=ret_vals,
	)


def is_maximal_lfsr(lfsr: LFSR):
	expected_period = (2 ** lfsr.num_bits) - 1
	initial_state = lfsr.get_state()

	# FIXME: this only works if expected_period is a Mersenne prime
	# Otherwise, this would fail if true period is factor of expected_period

	for _ in range(expected_period):
		lfsr()

	return lfsr.get_state() == initial_state


def analyze_lfsr(lfsr: LFSR):
	state = find_lfsr_period(lfsr, verbose=False, return_full_state=True)

	period = state['period']
	loop_start = state['loop_start']

	if loop_start:
		looped_part = state['ret_vals'][loop_start:]
		print('Period: %u, but looped region is %u - %u' % (
			period, loop_start, loop_start + period))

	else:
		looped_part = state['ret_vals']
		print('Period: %u' % period)

	assert len(looped_part) == period

	if period < 128:
		print('Full dump: ' + ''.join(['1' if val else '0' for val in looped_part]))

	assert all([val in [0, 1] for val in looped_part])

	num_one = sum(looped_part)
	num_zero = period - num_one

	print('Number of 0: %u (%.1f%%)' % (num_zero, num_zero/period*100))
	print('Number of 1: %u (%.1f%%)' % (num_one, num_one/period*100))

	vals_wrapped = looped_part + [looped_part[0]]

	zero_one_transitions = sum([vals_wrapped[n+1] and not vals_wrapped[n] for n in range(len(vals_wrapped) - 1)])
	print('0->1 and 1->0 transitions: %u of each' % zero_one_transitions)

	print('')


def test(verbose=False, long=False, max_len=None):
	from unit_test import unit_test

	if max_len is None:
		max_len = 25 if long else 13

	tests = []

	# Use lambdas so that LFSRs aren't constructed until running the test
	tests += [
		lambda: unit_test.test_equal(find_lfsr_period(NESLFSR(short=True, verbose=verbose)), 93),
		lambda: unit_test.test_equal(find_lfsr_period(NESLFSR(extra_short=True, verbose=verbose)), 31),
		lambda: unit_test.test_equal(find_lfsr_period(NESLFSR(short=False, verbose=verbose)), 32767),

		lambda: unit_test.test_equal(find_lfsr_period(GBLFSR(short=True, verbose=verbose)), 127),
		lambda: unit_test.test_equal(find_lfsr_period(GBLFSR(short=False, verbose=verbose)), 32767),
	]

	def _test_lfsr_len(n):
		lfsr = get_maximal_lfsr(n, verbose=verbose)
		return is_maximal_lfsr(lfsr)

	def _test_alt_lfsr_len(n):
		lfsr = get_maximal_lfsr(n, verbose=verbose, alt=True)
		assert lfsr is not None
		return is_maximal_lfsr(lfsr)

	for n in range(2, max_len + 1):
		tests.append(lambda: _test_lfsr_len(n))
		if get_maximal_lfsr(n, verbose=False, alt=True) is not None:
			tests.append(lambda: _test_alt_lfsr_len(n))

	return unit_test.run_unit_tests(tests, verbose=verbose)


def main(args):

	for n in range(2, 14):
		print('Analyzing maximal LFSR n=%u' % n)
		lfsr = get_maximal_lfsr(n, verbose=True, alt=False)
		analyze_lfsr(lfsr)

		alt_lfsr = get_maximal_lfsr(n, verbose=True, alt=True)
		if alt_lfsr is not None:
			print('Analyzing alternate maximal LFSR n=%u' % n)
			analyze_lfsr(lfsr)

	print('Analyzing GB short LFSR')
	analyze_lfsr(GBLFSR(short=True, verbose=True))

	print('Analyzing NES short LFSR')
	analyze_lfsr(NESLFSR(short=True, verbose=True))

	print('Analyzing NES short LFSR with extra-short seed')
	analyze_lfsr(NESLFSR(extra_short=True, verbose=True))

	print('Analyzing GB long LFSR (may take a little while)')
	analyze_lfsr(GBLFSR(short=False, verbose=True))

	print('Analyzing NES long LFSR (may take a little while)')
	analyze_lfsr(GBLFSR(short=False, verbose=True))
