#!/usr/bin/env python3

import importlib
import argparse
import inspect


# TODO: just use pytest for this


# If any of these fail, don't continue
CRITICAL_UNIT_TEST_MODULES = [
	'utils.approx_equal',  # Test approx_equal first because unit test framework depends on it
	'utils.utils',
	'analysis.linearity',  # Also used by later unit tests
	'analysis.freq_response',  # Also used by later unit tests
]

UNIT_TEST_MODULES = [
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

	test_modules = \
		[(module, True) for module in CRITICAL_UNIT_TEST_MODULES] + \
		[(module, False) for module in UNIT_TEST_MODULES]

	num_failures = 0
	num_tests = len(test_modules)
	failed_suites = []
	critical_failure = False

	separator = '=' * 40

	print('Running %i unit test suites...' % num_tests)

	for mod_name, is_critical in test_modules:
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
			if is_critical:
				critical_failure = True
				break

	print('')
	print(separator)
	print('Results')
	print(separator)

	if critical_failure:
		assert len(failed_suites) == 1
		print('Critical test suite failed: %s' % failed_suites[0])
	elif num_failures:
		print('%i/%i test suites failed: %s' % (num_failures, num_tests, ', '.join(failed_suites)))
	else:
		print('All %i test suites passed!' % num_tests)

	print('')

	exit(num_failures)


if __name__ == "__main__":
	main()
else:
	raise ImportError('run_unit_tests.py must be run as main!')
