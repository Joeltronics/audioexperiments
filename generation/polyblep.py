
from math import pi, cos, sin, atan2, sqrt, floor, ceil
import numpy as np
import matplotlib.pyplot as plt

# For optimized implementations, could assume input is bounded [-1.0,1.0]
def PolyBlep(t):
	if (t > 1.0):
		return 1.0
	elif (t < -1.0):
		return 0.0
	elif (t > 0.0):
		return -0.5*(t*t) + t + 0.5
	else:
		return  0.5*(t*t) + t + 0.5		

# For optimized implementations, could assume input is bounded [-1.0,1.0]
def DiffPolyBlep(t):
	if (t > 1.0) or (t < -1.0):
		return 0.0
	if (t > 0.0):
		return -0.5*(t*t) + t - 0.5
	else:
		return  0.5*(t*t) + t + 0.5

def IntegralPolyBlep(t, C=1.0):
	if (t < -1.0):
		return 0.5*t + C
	elif (t > 1.0):
		return -0.5*t + C
	elif (t > 0.0):
		return (1.0/6.0)*(t*t*t) - 0.5*(t*t) - 1.0/6.0 + C
	else:
		return -(1.0/6.0)*(t*t*t) - 0.5*(t*t) - 1.0/6.0 + C

def IntegralDiffPolyBlep(t):
	if (t > 1.0) or (t < -1.0):
		return 0.0
	elif (t > 0.0):
		return -(1.0/6.0)*(t*t*t) + 0.5*(t*t) - 0.5*t + 1.0/6.0
	else:
		return  (1.0/6.0)*(t*t*t) + 0.5*(t*t) + 0.5*t + 1.0/6.0
	
def GenPhase(freq, nSamp, startPhase=0.0):
	
	if (freq <= 0.0) or (freq >= 0.5):
		print("Warning: freq", freq)

	# This could be vectorized, but that's hard to do without internet access right now ;)
	ph = np.zeros(nSamp)
	phase = startPhase
	for n in range(nSamp):
		phase += freq
		ph[n] = phase
	
	ph = np.mod(ph, 1.0)
	
	return ph

def Saw(freq, nSamp, startPhase=0.0, oversamp=1, bAdvancedPoly=False):
	"""
	Advanced Polyblep:
	Scales PolyBlep amount according to phase
	Only applies to saw
	Effect is negligible except at very high frequencies or lots of oversampling
	Probably not worth it
	"""
	y = np.zeros(nSamp)
	#pbs = np.zeros(nSamp)
	phaseVec = GenPhase(freq, nSamp, startPhase)
	
	fOver = freq * oversamp
	
	for n, phase in enumerate(phaseVec):
		
		pb = 0.0
		if phase < fOver:
			pb = -DiffPolyBlep(phase / fOver)
			if bAdvancedPoly:
				pb *= (1.0 - phase)
			
		elif phase > (1.0-fOver):
			pb = -DiffPolyBlep((phase-1.0) / fOver)
			if bAdvancedPoly:
				pb *= phase
			
		y[n] = phase - 0.5 + pb
		#pbs[n] = pb
	
	y *= 2.0
	
	return y, phaseVec

def Squ(freq, nSamp, startPhase=0.0, dutyCycle=0.5):
	y = np.zeros(nSamp)
	phaseVec = GenPhase(freq, nSamp, startPhase)
	
	prevPh = startPhase
	for n, ph in enumerate(phaseVec):
		
		# TODO: polyblep is 2 samples, this only deals with 1
		
		if (ph < prevPh):
			# 1 => -1 
			
			#y[n] = 0.0
			
			# Find zero crossing location between samples (value 0.0-1.0)
			zeroxLoc = (dutyCycle - prevPh) / (ph - prevPh + 1)
			
			y[n] = PolyBlep(zeroxLoc)
			
		elif (ph >= dutyCycle) and (prevPh < dutyCycle):
			# -1 => 1

			#y[n] = 0.0
			
			# Find zero crossing location between samples (value 0.0-1.0)
			zeroxLoc = (dutyCycle - prevPh) / (ph - prevPh)
			
			y[n] = -PolyBlep(zeroxLoc)
			
		else:
			y[n] = 1.0 if (ph >= dutyCycle) else -1.0
			
		prevPh = ph
	
	return y



	

def PlotPolyBlep(npts=201):
	xx = np.linspace(-2.0, 2.0, npts)
	d = np.zeros(npts)
	y = np.zeros(npts)
	i = np.zeros(npts)
	id = np.zeros(npts)
	
	for n, x in enumerate(xx):
		d[n] = DiffPolyBlep(x)
		y[n] = PolyBlep(x)
		i[n] = IntegralPolyBlep(x)
		id[n] = IntegralDiffPolyBlep(x)
	
	#id = 1.0 - 0.5*np.abs(xx) - i
	
	plt.figure()
	plt.plot(xx, y, xx, d, xx, i, xx, id)
	plt.legend(['Step','DiffStep','Integral','IntegralDiff'])
	plt.grid()
	#plt.show()

def PlotFull():
	fs = 44100
	f = 440
	nSamp = 1024 * 4
	oversamp = 2.0
	#t = np.linspace(0.0, 0.01, fs)
	t = np.arange(nSamp)
	w = f / fs
	
	#y = Squ(w, nSamp)
	y, ph = Saw(w, nSamp, oversamp=oversamp)
	yAdv, _ = Saw(w, nSamp, oversamp=oversamp, bAdvancedPoly=True)
	
	ph = ph * 2.0 - 1.0
	
	Y = np.fft.fft(y * np.hamming(nSamp))
	Y = 20*np.log10(abs(Y))
	Y -= np.max(Y)
	
	YADV = np.fft.fft(yAdv* np.hamming(nSamp))
	YADV = 20*np.log10(abs(YADV))
	YADV -= np.max(YADV)
	
	PH = np.fft.fft(ph * np.hamming(nSamp))
	PH = 20*np.log10(abs(PH))
	PH -= np.max(PH)
	
	f = np.arange(nSamp) / nSamp
	
	plt.figure()
	
	plt.subplot(211)
	plt.plot(t, ph, '.-', t, y, '.-')
	plt.legend(['Naive','PolyBlep, oversamp=%.0f' % oversamp])
	#plt.plot(t, y, '.-', t, yAdv, '.-')
	#plt.legend(['PolyBlep, oversamp=%.0f' % oversamp, 'Advanced PolyBlep'])
	plt.grid()
	plt.xlim([0, 512])
	plt.ylim([-1.1, 1.1])
	
	plt.subplot(212)

	fNonAlias = np.arange(0.5*nSamp/oversamp) / nSamp
	fAlias = np.arange(0.5*nSamp/oversamp, nSamp) / nSamp
	YNonAlias = Y[0:(0.5*nSamp/oversamp)]
	YAlias = Y[(0.5*nSamp/oversamp):nSamp]
	
	print(len(fNonAlias), len(YNonAlias), len(fAlias), len(YAlias))
	print(np.min(fNonAlias), np.max(fNonAlias), np.min(fAlias), np.max(fAlias))
	
	plt.plot(f, PH, fNonAlias, YNonAlias, fAlias, YAlias)
	plt.legend(['Naive','PolyBlep', 'Aliased'])
	plt.xlim([0, 0.5])
	plt.xticks([1/32.0, 1/16.0, 1/8.0, 1/4.0, 1/2.0])
	
	#plt.plot(f, PH, '.-', f, Y, '.-', f, YADV, '.-')
	#plt.legend(['Naive','PolyBlep','Advanced'])
	#plt.xlim([0, 1/4.0])
	#plt.xticks([1/32.0, 1/16.0, 1/8.0, 1/4.0])

	#plt.plot(f, Y, '.', f, YADV, '.')
	#plt.legend(['PolyBlep','Advanced'])
	#plt.xlim([0, 0.5])
	#plt.xticks([1/32.0, 1/16.0, 1/8.0, 1/4.0, 1/2.0])
	
	plt.grid()
	plt.ylim([-80, 0])
	
	#plt.show()

def PlotCycle(fftsize=512):
	
	oversamp = 4
	
	actuallen = fftsize/oversamp
	
	incr = 2.0 / fftsize
	
	x = np.arange(0, fftsize)
	x2 = np.arange(0, fftsize, oversamp)
	y = x / fftsize + 0.5 + (incr / 2.0)
	y = np.mod(y, 1.0) * 2.0 - 1.0
	
	y2 = y[x2]
	
	prev = -1.0
	for n, val in enumerate(y):
		if (val < prev):
			zeroxloc = n
			# TODO: value between samples
			break
		prev = val
	
	print("Num samples:", len(y))
	print("Zero crossing loc:", zeroxloc)
	print("Range:", np.min(y), np.max(y))
	
	N = np.array(range(fftsize))
	Y = np.fft.fft(y)
	theta = np.angle(Y)
	Y = 20*np.log10(abs(Y))
	Y = Y - np.max(Y)
	
	
	plt.figure()
	
	plt.subplot(211)
	#plt.subplot(311)
	plt.plot(x, y, '-', x2, y2, '.')
	plt.xlim([0, fftsize-1])
	plt.ylim([-1.1, 1.1])
	plt.grid()
	
	plt.subplot(212)
	#plt.subplot(312)
	plt.plot(x, Y, '.')
	plt.xlim([0, fftsize/2-1])
	plt.ylim([-100,0])
	plt.xlabel('Harmonic')
	plt.ylabel('Magnitude (dB)')
	plt.grid()
	
	#plt.subplot(313)
	#plt.plot(x, theta, '.')
	#plt.xlim([0, fftsize/2-1])
	#plt.ylabel('Phase')
	#plt.grid()
	
	#plt.show()

def TriFromSawSqu():
	fs = 44100
	f = 440
	nSamp = 1024 * 4
	oversamp = 2.0
	#t = np.linspace(0.0, 0.01, fs)
	t = np.arange(nSamp)
	w = f / fs
	
	ySaw, ph = Saw(w, nSamp, oversamp=oversamp)
	#ySqu = Squ(w, nSamp)
	ySqu = np.array([(val >= 0.5) * 2 - 1 for val in ph]) # naive square
	y = ySaw * ySqu
	
	ph = ph * 2.0 - 1.0
	
	Y = np.fft.fft(y * np.hamming(nSamp))
	Y = 20*np.log10(abs(Y))
	Y -= np.max(Y)
	
	PH = np.fft.fft(ph * np.hamming(nSamp))
	PH = 20*np.log10(abs(PH))
	PH -= np.max(PH)
	
	f = np.arange(nSamp) / nSamp
	
	plt.figure()
	
	#plt.subplot(211)
	#plt.plot(t, ph, '.-', t, y, '.-')
	#plt.legend(['Naive','PolyBlep, oversamp=%.0f' % oversamp])
	plt.plot(t, ySqu, '.-', t, ySaw, '.-', t, y, '.-')
	plt.grid()
	plt.xlim([0, 512])
	plt.ylim([-1.1, 1.1])
	
	"""
	plt.subplot(212)

	fNonAlias = np.arange(0.5*nSamp/oversamp) / nSamp
	fAlias = np.arange(0.5*nSamp/oversamp, nSamp) / nSamp
	YNonAlias = Y[0:(0.5*nSamp/oversamp)]
	YAlias = Y[(0.5*nSamp/oversamp):nSamp]
	
	print(len(fNonAlias), len(YNonAlias), len(fAlias), len(YAlias))
	print(np.min(fNonAlias), np.max(fNonAlias), np.min(fAlias), np.max(fAlias))
	
	plt.plot(f, PH, fNonAlias, YNonAlias, fAlias, YAlias)
	plt.legend(['Naive','PolyBlep', 'Aliased'])
	plt.xlim([0, 0.5])
	plt.xticks([1/32.0, 1/16.0, 1/8.0, 1/4.0, 1/2.0])
	
	plt.grid()
	plt.ylim([-80, 0])
	"""
	
	#plt.show()
	
if __name__ == "__main__":
	
	#PlotPolyBlep()
	#PlotFull()
	#PlotCycle()
	TriFromSawSqu()
	
	plt.show()



