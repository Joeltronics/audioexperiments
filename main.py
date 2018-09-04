#!/usr/bin/env python3

import argparse
import importlib

if __name__ != "__main__":
	raise ImportError('main.py must be run as main!')

parser = argparse.ArgumentParser()

parser.add_argument('module')
parser.add_argument('-t', '--test', action='store_true', help="run module's unit tests")
parser.add_argument('-p', '--plot', action='store_true', help="run module's plot() function")
parser.add_argument('-v', '--verbose', action='store_true', help="verbose unit tests")

args, remaining_args = parser.parse_known_args()

mod = importlib.import_module(args.module)

if args.test:
	if hasattr(mod, 'test'):
		mod.test(verbose=args.verbose)
	else:
		print("Error: module '%s' has no test()" % args.module)

if args.plot:
	if hasattr(mod, 'plot'):
		mod.plot(remaining_args)
	else:
		print("Error: module '%s' has no plot()" % args.module)

if not args.test and not args.plot:
	if hasattr(mod, 'main'):
		mod.main(remaining_args)
	else:
		print("Error: module '%s' has no main()" % args.module)
