

def SolveLadderY(g, s, initialEstimate=None, stats=stats_Ladder):
	#y = s - g*tanh(y)
	# solves for y (iteratively)
	
	if initialEstimate is None:
		initialEstimate = s
	
	y = initialEstimate
	
	#y = s - g*tanh(y)
	#y + g*tanh(y) = s
	#y + g*tanh(y) - s = 0
	
	errs = []
	
	bSuccess = False
	prevErr = None
	for iter in range(maxNIter):
		
		# Calculate it!
		y = s - g*math.tanh(y)
		
		err = y + g*math.tanh(y) - s
		
		err = abs(err)
		errs += [err]
		
		if err <= eps:
			bSuccess = True
			break
		
		if (prevErr is not None) and (err >= prevErr):
			print('Warning: failed to converge! Falling back to initial estimate')
			#return initialEstimate
			y = initialEstimate
			break
		
		prevErr = err
	
	stats.Add(
		bSuccess=bSuccess,
		est=initialEstimate,
		nIter=iter+1,
		final=y,
		err=errs)
	
	return y

def SolveOtaY(g, x, s, initialEstimate=None, stats=stats_Ota):
	# y = s + g*tanh(x - y)
	# solves for y (iteratively)
	
	if initialEstimate is None:
		initialEstimate = s
	
	y = initialEstimate
	
	# y = s + g*tanh(x - y)
	# y + g*tanh(y - x) = s
	# y + g*tanh(y - x) - s = 0
	
	errs = []
	
	bSuccess = False
	prevErr = None
	for iter in range(maxNIter):
		
		# Calculate it!
		y = s + g*math.tanh(x - y)
		
		err = y + g*math.tanh(y - x) - s
		
		err = abs(err)
		errs += [err]
		
		if err <= eps:
			bSuccess = True
			break
		
		if (prevErr is not None) and (err >= prevErr):
			print('Warning: failed to converge! Falling back to initial estimate')
			y = initialEstimate
			break
			#return initialEstimate
		
		prevErr = err
	
	stats.Add(
		bSuccess=bSuccess,
		est=initialEstimate,
		nIter=iter+1,
		final=y,
		err=errs)
	
	return y