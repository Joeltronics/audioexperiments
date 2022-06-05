#!/usr/bin/env python3

import argparse
import importlib
import inspect
import sys


MODULES = [
	'analysis.distortion',
	'analysis.freq_response',
	'analysis.linearity',
	'analysis.sine_to_phase',
	'approx.cheby',
	'approx.trig',
	'compression.basic_compressors',
	'delay_reverb.dattorro_reverb',
	'delay_reverb.delay_line',
	'delay_reverb.multitap_comb',
	'filters.allpass',
	'filters.biquad',
	'filters.cascade',
	'filters.crossover',
	'filters.filter_base',
	'filters.iir_filters',
	'filters.one_pole',
	'filters.peak_filters',
	'filters.pink',
	'filters.zdf.cascade',
	'filters.zdf.onepole',
	'filters.zdf.svf',
	'filters.zdf.tanh_lin_approx',
	'filters.zdf.zdf_mystran',
	'generation.additive',
	'generation.lfsr',
	'generation.multi_lfsr',
	'generation.polyblep',
	'generation.signal_generation',
	'overdrive.diode_clip',
	'overdrive.overdrive',
	'overdrive.tanh_fb',
	'resampling.resamplers',
	'solvers.solvers',
	'utils.approx_equal',
	'utils.utils',
]


def parse_args():
	parser = argparse.ArgumentParser()

	test_parser = argparse.ArgumentParser(add_help=False)
	test_parser.add_argument('-t', '--test', action='store_true', help="run unit tests")
	test_parser.add_argument('-v', '--verbose', action='store_true', help="verbose unit tests")
	test_parser.add_argument('--long', action='store_true', help="long unit tests")

	plot_parser = argparse.ArgumentParser(add_help=False)
	plot_parser.add_argument('-p', '--plot', action='store_true', help="run plot() function")

	subparsers = parser.add_subparsers(dest='module_name')

	for module_name in MODULES:

		# Importing every single possible module is slow.
		# If a module name isn't present in argv, then we know its subparser details won't be used anyway,
		# so just add an empty subparser instead of importing the module
		if module_name not in sys.argv:
			subparsers.add_parser(module_name)
			continue

		module = importlib.import_module(module_name)

		parents = []

		if hasattr(module, 'test'):
			parents.append(test_parser)

		if hasattr(module, 'plot'):
			parents.append(plot_parser)

		if hasattr(module, 'get_parser'):
			parents.append(module.get_parser())

		subparsers.add_parser(module_name, parents=parents)

	return parser.parse_args()


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
	args = parse_args()

	if 'long' not in args:
		args.long = False

	if 'test' not in args:
		args.test = False

	if 'plot' not in args:
		args.plot = False

	if args.long:
		args.test = True

	if args.test and args.plot:
		print('ERROR: cannot give both --test and --plot')
		return -1

	mod = importlib.import_module(args.module_name)

	if args.test:
		return test(mod, module_name=args.module_name, verbose=args.verbose, long=args.long)
	elif args.plot:
		return plot(mod, module_name=args.module_name, module_args=args)
	else:
		return module_main(mod, module_name=args.module_name, module_args=args)


if __name__ == "__main__":
	ret = main()
	exit(ret)
