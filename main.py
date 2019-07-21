#!/usr/bin/env python3

import argparse
import importlib
import inspect


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('module')
	parser.add_argument('-t', '--test', action='store_true', help="run module's unit tests")
	parser.add_argument('-p', '--plot', action='store_true', help="run module's plot() function")
	parser.add_argument('-v', '--verbose', action='store_true', help="verbose unit tests")
	parser.add_argument('--long', action='store_true', help="long unit tests")
	parser.add_argument('--mh', action='store_true', help="run module's --help")

	return parser.parse_known_args()


def test(mod, module_name, verbose, long):

	if hasattr(mod, 'test'):
		kwargs = dict()

		test_args = inspect.getfullargspec(mod.test).args

		if 'verbose' in test_args:
			test_args.remove('verbose')
			kwargs['verbose'] = verbose

		if 'long' in test_args:
			test_args.remove('long')
			kwargs['long'] = long

		if test_args:
			print('WARNING: unexpected args in test() function: %s' % test_args)

		mod.test(**kwargs)
	else:
		print("Error: module '%s' has no test()" % module_name)
		return -1


def plot(mod, module_name, module_args):
	if hasattr(mod, 'plot'):
		mod.plot(module_args)
	else:
		print("Error: module '%s' has no plot()" % module_name)
		return -1


def module_main(mod, module_name, module_args):
	if hasattr(mod, 'main'):
		ret = mod.main(module_args)
		if ret:
			print('Returned %s' % ret)
		return ret
	else:
		print("Error: module '%s' has no main()" % module_name)
		return -1


def main():
	main_args, module_args = parse_args()

	if main_args.long:
		main_args.test = True

	if main_args.test and main_args.plot:
		print('ERROR: cannot give both --test and --plot')
		return -1

	mod = importlib.import_module(main_args.module)

	if main_args.mh:
		module_args.append('--help')

	if main_args.test:
		return test(mod, module_name=main_args.module, verbose=main_args.verbose, long=main_args.long)

	elif main_args.plot:
		return plot(mod, module_name=main_args.module, module_args=module_args)

	else:
		return module_main(mod, module_name=main_args.module, module_args=module_args)


if __name__ == "__main__":
	ret = main()
	exit(ret)
else:
	raise ImportError('main.py must be run as main!')
