
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import struct

Exp = lambda x: np.exp(x)
Ln = lambda x: np.log(x)
Log2 = lambda x: np.log2(x)
Exp2 = lambda x: np.power(2.0, x)
InvSqrt = lambda x: 1.0 / np.sqrt(x)
Sqrt = lambda x: np.sqrt(x)
Tanh = lambda x: np.tanh(x)

log2e = Log2(math.e)

##### Derivative #####

def Deriv(x, y):
	dx = x[1] - x[0]
	dydx = np.diff(y) / dx
	xx = x[:-1] + 0.5*dx
	return xx, dydx

##### Error #####

def SquRelErr(yApprox, yExact):
	err = 1.0 - yApprox[yExact != 0] / yExact[yExact != 0]
	return np.average(np.square(err))

def SquAbsErr(yApprox, yExact):
	err = yApprox - yExact
	return np.average(np.square(err))

def MaxAbsErr(yApprox, yExact):
	err = yApprox - yExact
	err = np.abs(err)
	return np.amax(err)

def MaxDerivErr(yApprox, yExact):
	return np.amax(np.diff(yApprox) - np.diff(yExact))

##### Type Punning #####

def PunFloatToInt(x):
	fx = float(x)
	return struct.unpack("@i", struct.pack("@f", fx))[0]

def PunIntToFloat(x):
	return struct.unpack("@f", struct.pack("@i", x))[0]

##### The approximations #####

def InvSqrtApprox(x, order=1, c1=0x5F375A86):
	xhalf = 0.5*x
	
	i = PunFloatToInt(x)
	i = c1 - ( i >> 1 )
	x = PunIntToFloat(i)
	
	for _ in range(order):
		# Newton-Raphson step - repeat for more accuracy
		x = x*(1.5 - xhalf*x*x )
	
	return x

vInvSqrtApprox = np.vectorize(InvSqrtApprox)

def SqrtApprox(x, order=1):
	i = PunFloatToInt(x)
	i -= (1 << 23)
	i >>= 1
	i += (1 << 29)
	y = PunIntToFloat(i)
	
	for _ in range(order):
		# Newton-Raphson step - repeat for more accuracy
		y = ( y*y + x ) / ( 2*y )
	
	return y

vSqrtApprox = np.vectorize(SqrtApprox)

def Log2Approx(x, c1=0x3f800000, c2=0x00800000):
	# 0x00800000 = 0x40000000 - 0x3f800000
	return float(PunFloatToInt(x) - c1) / float(c2)

vLog2Approx = np.vectorize(Log2Approx)

def LnApprox(x, c1=0x3f800000, c2=0x00800000):
	c2 = float(c2) * log2e
	y = float(PunFloatToInt(x) - c1) / c2
	return y

vLnApprox = np.vectorize(LnApprox)

def Exp2Approx(x, c1=0x3f800000, c2=0x00800000):
	# 0x00800000 = 0x40000000 - 0x3f800000
	y_int = int(x*float(c2) + float(c1))
	return PunIntToFloat(y_int)

vExp2Approx = np.vectorize(Exp2Approx)

def ExpApprox(x, c1=0x3f800000, c2=0x00800000):
	# 0x00800000 = 0x40000000 - 0x3f800000
	y_int = int(x*float(c2)*log2e + float(c1))
	y = PunIntToFloat(y_int)
	return y

vExpApprox = np.vectorize(ExpApprox)

def TanhApprox(x, c1=0x3f800000, c2=0x00800000):
	expX = vExpApprox(x, c1=c1, c2=c2)
	expNegX = vExpApprox(-x, c1=c1, c2=c2)
	return (expX - expNegX) / (expX + expNegX)

vTanhApprox = np.vectorize(TanhApprox)

"""
def InvSqrtApprox(x):
	return _DoOperation(x, _InvSqrtApprox)

def SqrtApprox(x):
	return _DoOperation(x, _SqrtApprox)

def Log2Approx(x):
	return _DoOperation(x, _Log2Approx)

def Exp2Approx(x):
	return _DoOperation(x, _Exp2Approx)

def ExpApprox(x):
	return _DoOperation(x, _ExpApprox)

def LnApprox(x):
	return _DoOperation(x, _LnApprox)

def TanhApprox(x):
	return _DoOperation(x, _TanhApprox)
"""


##### Here's the stuff #####


def Sweep1D(x, fExact, fApprox, fErr):

	c1_init = 0x3f800000
	c2 = 0x00800000

	yExact = fExact(x)
	
	# These have to be array so we can access by reference in UpdateErr
	c1_minErr = [c1_init]
	minErr = [None]
	
	def UpdateErr(c1, c2, err):
		if not np.isnan(err):
			if (not minErr[0]) or (err < minErr[0]):
				minErr[0] = err
				c1_minErr[0] = c1
				print('c1 = 0x%08x, c2 = 0x%08x, err^2 = %.6e <-- new minimum' % (c1, c2, err))
			else:
				#if (err < 0.9):
				print('c1 = 0x%08x, c2 = 0x%08x, err^2 = %.6e' % (c1, c2, err))
	
	# A bisection search would be way faster, but whatever
	
	min  = 0x3D000000
	max  = 0x3FF00000
	step = 0x00100000
	
	while (step >= 1):
	
		print('0x%08x to 0x%08x, step: 0x%08x' % (min, max, step))
	
		prevErr = None
		for c1 in np.arange(min, max+1, step):
			yApprox = fApprox(x, c1, c2)
			err = fErr(yApprox, yExact)
			UpdateErr(c1, c2, err)
			
			# Error should have a single minimum - so if it ever increases, can break loop early
			if prevErr and (err > prevErr):
				break
			prevErr = err
		
		min = c1_minErr[0] - step 
		max = c1_minErr[0] + step
		step //= 16
		
	yApprox = fApprox(x, c1_minErr[0], c2)
	
	return yApprox, c1_minErr[0], c2, minErr[0]

def Sweep2Dplot(x, fExact, fApprox, fErr, title=None):

	yExact = fExact(x)

	c1_init = 0x3f800000
	c2_init = 0x00800000
	
	#c1_sweep = np.arange(0x3D000000, 0x3FF00000+1, 0x00100000)
	c1_sweep = np.arange(0x3F000000, 0x3FF00000+1, 0x00100000)
	
	#c2_sweep = np.arange(0x00000000, 0x00F00000+1, 0x00100000)
	c2_sweep = np.arange(0x00700000, 0x00900000+1, 0x00010000)
	#c2_sweep = np.arange(0x00800000, 0x00800000+1, 0x00010000)
	#c2_sweep = 0x40000000 - c1_sweep
	
	nVals = len(c1_sweep) * len(c2_sweep)
	
	pltC1  = np.zeros(nVals)
	pltC2  = np.zeros(nVals)
	pltErr = np.zeros(nVals)
	
	n = 0
	
	for c1 in c1_sweep:
		print('c1 = 0x%08x' % c1)
		for c2 in c2_sweep:
			
			yApprox = fApprox(x, c1, c2)
			err = fErr(yApprox, yExact)
			
			pltC1[n] = c1
			pltC2[n] = c2
			pltErr[n] = err
			
			n += 1
			
	pltC1 = np.array(pltC1)
	pltC2 = np.array(pltC2)
	pltErr = np.array(pltErr)
	
	pltErr = np.log10(pltErr)
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	#ax.plot_surface(pltC1, pltC2, pltErr)
	#ax.plot_wireframe(pltC1, pltC2, pltErr)
	ax.plot_trisurf(pltC1, pltC2, pltErr)
	#ax.scatter(pltC1, pltC2, pltErr)
	
	plt.xlabel('C1')
	plt.ylabel('C2')
	ax.set_zlabel('log10(err)')
	
	if title:
		plt.title(title)

x = np.linspace(-5, 5, 5000)

Sweep2Dplot(x, Exp, vExpApprox, SquRelErr, title='Exp')
Sweep2Dplot(x, Tanh, vTanhApprox, SquAbsErr, title='Tanh')
#Sweep2Dplot(x, Tanh, vTanhApprox, MaxAbsErr, title='Tanh')
#Sweep2Dplot(x, Tanh, vTanhApprox, MaxDerivErr, title='Tanh')

"""
print('')
print('exp(x)')
print('')

yExpApprox, c1_minExpErr, c2_minExpErr, minExpErr = Sweep1D(x, Exp, vExpApprox, SquRelErr)

print('')
print('tanh(x)')
print('')

yTanhApprox, c1_minTanhErr, c2_minTanhErr, minTanhErr = Sweep1D(x, Tanh, vTanhApprox, MaxAbsErr)

print('Minimum exp error: c1 = 0x%08x, c2 = 0x%08x, err^2 = %.12f' % (c1_minExpErr, c2_minExpErr, minExpErr))
print('Minimum tanh error: c1 = 0x%08x, c2 = 0x%08x, err^2 = %.12f' % (c1_minTanhErr, c2_minTanhErr, minTanhErr))

yExact = Exp(x)
yApprox = vExpApprox(x, c1_minExpErr, c2_minExpErr)
err = yApprox[yExact != 0] / yExact[yExact != 0] - 1.0

plt.figure()

plt.subplot(211)
plt.plot(x, yExact, x, yApprox)
plt.grid()
plt.title('exp')

plt.subplot(212)
plt.plot(x, err)
plt.grid()


yExact = Tanh(x)
yApprox = vTanhApprox(x, c1_minTanhErr, c2_minTanhErr)
err = yApprox - yExact

plt.figure()

plt.subplot(211)
plt.plot(x, yExact, x, yApprox)
plt.grid()
plt.title('tanh')

plt.subplot(212)
plt.plot(x, err)
plt.grid()
"""



plt.draw()
plt.show()