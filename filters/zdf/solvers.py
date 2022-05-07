#!/usr/bin/env python3

import numpy as np

max_n_iter = 20
eps = 1e-6

def newton_raphson(f, df, initial_estimate=None, max_n_iter=max_n_iter, eps=eps, stats=None):
	
	if initial_estimate is None:
		initial_estimate = s
	
	y = initial_estimate
	
	errs = []
	
	success = False
	prev_err = None
	for iter in range(max_n_iter):
		
		fy = f(y)
		
		err = abs(fy)
		
		errs += [err]
		
		if err <= eps:
			success = True
			break
		
		if (prev_err is not None) and (err >= prev_err):
			print('Warning: failed to converge! Falling back to initial estimate')
			y = initial_estimate
			break
		
		dfy = df(y)
		
		# Prevent divide-by-zero, or very shallow slopes
		if dfy < eps:
			# this shouldn't be possible with the functions we're actually using (derivative of tanh)
			print("Warning: d/dy f(y=%f) = 0.0, can't solve Newton-Raphson" % y)
			break
			
		y = y - fy/dfy
		
		prev_err = err
	
	if stats is not None:
		stats.add(
			success=success,
			est=initial_estimate,
			n_iter=iter+1,
			final=y,
			err=errs)
	
	return y


def iterative(f_zero, initial_estimate=None, max_n_iter=max_n_iter, eps=eps, stats=None):
	
	if initial_estimate is None:
		initial_estimate = 0.0
	
	y = initial_estimate
	
	errs = []
	
	success = False
	prev_abs_err = None
	for iter in range(max_n_iter):
		
		err = f_zero(y)
		
		abs_err = abs(err)
		errs += [abs_err]
		
		if abs_err <= eps:
			success = True
			break
		
		if (prev_abs_err is not None) and (abs_err >= prev_abs_err):
			print('Warning: failed to converge! Falling back to initial estimate')
			#return initial_estimate
			y = initial_estimate
			break
		
		y = y - err
		
		prev_abs_err = abs_err
	
	if stats is not None:
		stats.add(
			success=success,
			est=initial_estimate,
			n_iter=iter+1,
			final=y,
			err=errs)
	
	return y

