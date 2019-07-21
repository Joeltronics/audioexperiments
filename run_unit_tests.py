#!/usr/bin/env python3

import importlib
import argparse
import inspect


# TODO: just use pytest for this


UNIT_TEST_MODULES = [
	'utils.approx_equal',  # Test approx_equal first because unit test framework depends on it
	'utils.utils',
	'filters.biquad',
	'filters.one_pole',
	'filters.pink',
	'filters.peak_filters',
	'delay_reverb.delay_line',
	'generation.lfsr',
]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', action='store_true', help='Print details of every test, even on success')
	parser.add_argument('-l', '--long', action='store_true', help='Run long tests')
	args = parser.parse_args()

	num_failures = 0
	num_tests = len(UNIT_TEST_MODULES)
	failed_suites = []

	separator = '=' * 40

	print('Running %i unit test suites...' % num_tests)

	for mod_name in UNIT_TEST_MODULES:
		print('')
		print(separator)
		print("Unit test suite: %s" % mod_name)
		print(separator)

		mod = importlib.import_module(mod_name)

		kwargs = dict()

		test_args = inspect.getfullargspec(mod.test).args

		if 'verbose' in test_args:
			test_args.remove('verbose')
			kwargs['verbose'] = args.verbose

		if 'long' in test_args:
			test_args.remove('long')
			kwargs['long'] = args.long

		if test_args:
			print('WARNING: unexpected args in test() function: %s' % test_args)

		result = mod.test(**kwargs)

		if result is None:
			print('WARNING: test() returned None')
		if not result:
			num_failures += 1
			failed_suites.append(mod_name)

	print('')
	print(separator)
	print('Results')
	print(separator)

	if num_failures:
		print('%i/%i test suites failed: %s' % (num_failures, num_tests, ', '.join(failed_suites)))
	else:
		print('All %i test suites passed!' % num_tests)

	print('')

	exit(num_failures)


if __name__ == "__main__":
	main()
else:
	raise ImportError('run_unit_tests.py must be run as main!')
