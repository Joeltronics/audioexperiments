#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

# TODO: currently there is no way to separate stats from each (outer) iteration
# (e.g. first outer iteration should maybe have separate stats)


class IterStats:
	def __init__(self, title=None):
		self.stats = []
		self.title = title
	
	def add(self, success, n_iter, est, final, err, input=None):
		
		if len(err) != n_iter:
			print('len(err) != n_iter (%i != %i)' % (len(err), n_iter))
	
		self.stats += [{
			'success': success,
			'n_iter': n_iter,
			'est': est, # initial estimate
			'final': final,
			'err': err,
			'init_err': err[0],
			'final_err': err[-1],
			'input': input
		}]
	
	def output(self, bPrint=True, plot_iter=True, plot_est=False, plot_err = False, new_fig=True):
		
		n_total = len(self.stats)
		
		# TODO: count successes & failures
		successes = [i['success'] for i in self.stats]
		success_counter = Counter(successes)
		
		# Number of iterations
		
		n_iter = [i['n_iter'] for i in self.stats]
		iter_counter = Counter(n_iter)
		
		# Initial error
		
		init_errs = [i['init_err'] for i in self.stats]
		init_err_range = (min(init_errs), sum(init_errs) / n_total, max(init_errs))
		#init_diff = [abs(i['final'] - i['est']) for i in self.stats]
		#init_diff_range = (min(init_errs), sum(init_errs) / n_total, max(init_errs))
		
		if bPrint:
			print('')
			print('Stats for ' + self.title + ':')
			succ_counts = list(success_counter.items())
			succ_counts = [(i[0], i[1]/n_total) for i in succ_counts]
			print('Successes: ' + repr(succ_counts))
			iter_counts = list(iter_counter.items())
			iter_counts = [(i[0], i[1]/n_total) for i in iter_counts]
			print('Convergence: ' + repr(iter_counts))
			print('Initial Estimate Error from zero: average %.3f, range (%.3f, %.3f)' % (init_err_range[1], init_err_range[0], init_err_range[2]))
		
		if plot_iter:
			
			max_n_iter = max(n_iter)
			
			x = np.arange(1, max_n_iter+2)
			y = np.zeros_like(x)
			
			for n in range(max_n_iter):
				y[n] = iter_counter[n+1]
			
			if new_fig:
				plt.figure()
			
			plt.plot(x, y / n_total, 'o-')
			
			if new_fig:
				plt.title('Convergence of ' + self.title)
				plt.xlabel('# iterations')
				plt.grid()

		if plot_est:

			x = [i['est'] for i in self.stats]
			y = [i['final'] for i in self.stats]

			if new_fig:
				plt.figure()

			plt.plot(x, y, '.')

			if new_fig:
				plt.title('Estimate of ' + self.title)
				plt.xlabel('Estimate')
				plt.ylabel('Final')
				plt.grid()

		if plot_err:
			y = [i['final'] - i['est'] for i in self.stats]

			if new_fig:
				plt.figure()

			plt.semilogy(y, '.')

			if new_fig:
				plt.title('Estimate error of ' + self.title)
				plt.xlabel('Estimate')
				plt.ylabel('Final')
				plt.grid()
