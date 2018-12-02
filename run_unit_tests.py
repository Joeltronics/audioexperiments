#!/usr/bin/env python3

import importlib
import argparse


if __name__ != "__main__":
	raise ImportError('run_unit_tests.py must be run as main!')


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='Print details of every test, even on success')

args = parser.parse_args()

unit_test_modules = [
	# Test approx_equal first because unit test framework depends on it
	'utils.approx_equal',
	'utils.utils',
	'filters.biquad',
	'filters.one_pole',
	'filters.pink',
	'filters.peak_filters',
	'delay_reverb.delay_line',
]

# blatantly copied from unit_test.run_unit_tests but with "suites" added

num_failures = 0
num_tests = len(unit_test_modules)
failed_suites = []

separator = '=' * 40

print('Running %i unit test suites...' % num_tests)

for mod_name in unit_test_modules:
	print('')
	print(separator)
	print("Unit test suite: %s" % mod_name)
	print(separator)
	mod = importlib.import_module(mod_name)
	result = mod.test(verbose=args.verbose)
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
