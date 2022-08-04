
import numpy as np
from matplotlib import pyplot as plt
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

def PunFloatToInt(x):
	fx = float(x)
	return struct.unpack("@i", struct.pack("@f", fx))[0]

def PunIntToFloat(x):
	return struct.unpack("@f", struct.pack("@i", x))[0]

def _IsSingleValue(x):
	try:
		len(x)
		return False
	except TypeError:
		return True

def _DoOperation(x, f):
	if _IsSingleValue(x):
		return f(x)
	else:
		y = np.zeros_like(x)
		for n, xx in enumerate(x):
			y[n] = f(xx)
		return y

def _InvSqrtApprox(x, order=1):
	xhalf = 0.5*x
	
	i = PunFloatToInt(x)
	i = 0x5F375A86 - ( i >> 1 )
	x = PunIntToFloat(i)
	
	for _ in range(order):
		# Newton-Raphson step - repeat for more accuracy
		x = x*(1.5 - xhalf*x*x )
	
	return x

def _SqrtApprox(x, order=1):
	i = PunFloatToInt(x)
	i -= (1 << 23)
	i >>= 1
	i += (1 << 29)
	y = PunIntToFloat(i)
	
	for _ in range(order):
		# Newton-Raphson step - repeat for more accuracy
		y = ( y*y + x ) / ( 2*y )
	
	return y

def _Log2Approx(x):
	# 0x00800000 = 0x40000000 - 0x3f800000
	return float(PunFloatToInt(x) - 0x3f800000) / float(0x00800000)

def _LnApprox(x):
	c1 = float(0x00800000) * log2e
	#c2 = 0x3f800000
	c2 = 0x3f78acfb
	y = float(PunFloatToInt(x) - c2) / (c1)
	#y *= 1.03 # optional, improves accuracy slightly
	return y

def _Exp2Approx(x):
	# TODO: round to int instead?
	c1 = float(0x00800000)
	#c2 = float(0x3f800000)
	c2 = float(0x3f78acfb)
	y_int = int(x*c1 + c2)
	return PunIntToFloat(y_int)

def _ExpApprox(x):
	c1 = float(0x00800000)*log2e
	#c2 = float(0x3f800000)
	c2 = float(0x3f78acfb)
	y_int = int(x*c1 + c2)
	y = PunIntToFloat(y_int)
	#y *= 0.96 # optional, improves accuracy slightly
	
	# Newton-Raphson - these don't help because LnApprox is just as imprecise
	# Might work better once I find better "magic numbers" though
	#y = y - y*_LnApprox(y) + x*y
	
	return y

def _ExpApproxMulti(x):
	
	c1 = 0.5*Ln(2.0)
	c2 = 0.25*Ln(2.0)
	expc1 = Exp(c1)
	expc2 = Exp(c2)
	
	expX1 = _ExpApprox(x)
	expX2 = _ExpApprox(x + c1) / expc1
	expX3 = _ExpApprox(x - c1) * expc1
	expX4 = _ExpApprox(x + c2) / expc2
	expX5 = _ExpApprox(x - c2) * expc2
	
	y = (expX1 + expX2 + expX3 + expX4 + expX5) / 5.0
	
	# 0.96 is just from "eyeballing" it
	#y *= 0.96
	
	return y
	
	

def _TanhApprox(x):
	if True:
		expX = _ExpApprox(x)
		expNegX = _ExpApprox(-x)
		#expNegX = 1.0 / expX # actually less precise
	elif False:
		expX    = 0.5*(_ExpApprox(x) + 1.0/_ExpApprox(-x))
		expNegX = 0.5*(_ExpApprox(-x) + 1.0/_ExpApprox(x))
	else:
		expX = _ExpApproxMulti(x)
		expNegX = _ExpApproxMulti(-x)
	
	y = (expX - expNegX) / (expX + expNegX)
	
	# Newton Raphson - doesn't seem to help
	#y = y - (1.0 - y*y) *  (0.5*_LnApprox((1+y)/(1-y)) - x)
	
	return y

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
	#return Log2Approx(x) / log2e
	return _DoOperation(x, _LnApprox)

def TanhApprox(x):
	return _DoOperation(x, _TanhApprox)

def SanityCheck():
	x = math.pi

	x_int = PunFloatToInt(x)

	print('x = %f' % x)
	print('x_int = %i' % x_int)
	print('')
	print('Log2(x) = %f' % (math.log(x)/math.log(2)))
	print('Log2Approx(x) = %f' % Log2Approx(x))
	print('Exp2(x) = %f' % (2.0**x))
	print('Exp2Approx(x) = %f' % Exp2Approx(x))
	print('')
	print('Log2Approx(0) = %f' % Log2Approx(0))
	print('LnApprox(0) = %f' % LnApprox(0))
	print('LnApprox(e) = %f' % LnApprox(math.e))
	print('Exp2Approx(0) = %f' % Exp2Approx(0))
	print('Exp2Approx(-1) = %f' % Exp2Approx(-1))

#SanityCheck()

def Deriv(x, y):
	dx = x[1] - x[0]
	dydx = np.diff(y) / dx
	xx = x[:-1] + 0.5*dx
	return xx, dydx

def PlotLog2(nPts=5000):
	x = np.linspace(0.01, 20, nPts)

	yExact = Log2(x)
	yApprox = Log2Approx(x)
	plt.plot(x, yExact, x, yApprox)
	plt.grid()
	plt.title('Log2(x)')

def PlotExp2(nPts=5000):
	x = np.linspace(-5, 5, nPts)
	#x = np.linspace(-20, 20, nPts)

	yExact = Exp2(x)
	yApprox = Exp2Approx(x)
	plt.plot(x, yExact, x, yApprox)
	plt.grid()
	plt.title('2^x')

def PlotExp(nPts=5000):
	x = np.linspace(-5, 5, nPts)
	#x = np.linspace(-20, 20, nPts)

	yExact = Exp(x)
	yApprox = ExpApprox(x)
	plt.plot(x, yExact, x, yApprox)
	plt.grid()
	plt.title('e^x')

def PlotLn(nPts = 5000):
	x = np.linspace(0.01, 20, nPts)

	yExact = Ln(x)
	yApprox = LnApprox(x)
	plt.plot(x, yExact, x, yApprox)
	plt.grid()
	plt.title('Ln(x)')

def PlotExpLn(nPts = 5000):
	x = np.linspace(0.01, 20, nPts)

	err = ExpApprox(LnApprox(x)) - x
	plt.plot(x, err)
	plt.grid()
	plt.title('Exp(Ln(x)) - x')

def PlotLnExp(nPts=5000):
	x = np.linspace(-5, 5, nPts)
	#x = np.linspace(-20, 20, nPts)

	err = LnApprox(ExpApprox(x)) - x
	plt.plot(x, err)
	plt.grid()
	plt.title('Ln(Exp(x)) - x')

def PlotInvSqrt(nPts = 5000):
	x = np.linspace(0.1, 20, nPts)

	yExact = InvSqrt(x)
	yApprox = InvSqrtApprox(x)
	plt.plot(x, yExact, x, yApprox)
	plt.grid()
	plt.title('1/sqrt(x)')

def PlotSqrt(nPts = 5000):
	x = np.linspace(0, 20, nPts)

	yExact = Sqrt(x)
	yApprox = SqrtApprox(x)
	plt.plot(x, yExact, x, yApprox)
	plt.grid()
	plt.title('sqrt(x)')

	
def PlotTanh(nPts = 5000):
	x = np.linspace(-5, 5, nPts)

	yExact = Tanh(x)
	yApprox = TanhApprox(x)
	plt.plot(x, yExact, x, yApprox)
	plt.grid()
	plt.title('tanh(x)')

def PlotLnDeriv(nPts = 5000):
	x = np.linspace(0.1, 20, nPts)

	yExact = Ln(x)
	yApprox = LnApprox(x)
	
	xp, dyExact = Deriv(x, yExact)
	__, dyApprox = Deriv(x, yApprox)
	
	plt.plot(xp, dyExact, xp, dyApprox)
	plt.grid()
	plt.title('d/dx ln(x)')

def PlotExpDeriv(nPts = 5000):
	x = np.linspace(-5, 5, nPts)

	yExact = Exp(x)
	yApprox = ExpApprox(x)
	
	xp, dyExact = Deriv(x, yExact)
	__, dyApprox = Deriv(x, yApprox)
	
	plt.plot(xp, dyExact, xp, dyApprox)
	plt.grid()
	plt.title('d/dx exp(x)')

def PlotTanhDeriv(nPts = 5000):
	x = np.linspace(-5, 5, nPts)

	yExact = Tanh(x)
	yApprox = TanhApprox(x)
	
	xp, dyExact = Deriv(x, yExact)
	__, dyApprox = Deriv(x, yApprox)
	
	plt.plot(xp, dyExact, xp, dyApprox)
	plt.grid()
	plt.title('d/dx tanh(x)')

def PlotInvSqrtDeriv(nPts = 5000):
	x = np.linspace(0.2, 20, nPts)

	yExact = InvSqrt(x)
	yApprox = InvSqrtApprox(x)
	
	xp, dyExact = Deriv(x, yExact)
	__, dyApprox = Deriv(x, yApprox)
	
	plt.plot(xp, dyExact, xp, dyApprox)
	plt.grid()
	plt.title('d/dx 1/sqrt(x)')

def PlotSqrtDeriv(nPts = 5000):
	x = np.linspace(0.1, 20, nPts)

	yExact = Sqrt(x)
	yApprox = SqrtApprox(x)
	
	xp, dyExact = Deriv(x, yExact)
	__, dyApprox = Deriv(x, yApprox)
	
	plt.plot(xp, dyExact, xp, dyApprox)
	plt.grid()
	plt.title('d/dx sqrt(x)')

plt.figure()

plt.subplot(221)
#PlotLog2()
PlotLn()
plt.subplot(222)
#PlotExp2()
PlotExp()
plt.subplot(223)
PlotLnDeriv()
plt.subplot(224)
PlotExpDeriv()


plt.figure()
plt.subplot(221)
PlotInvSqrt()
plt.subplot(222)
PlotSqrt()
plt.subplot(223)
PlotInvSqrtDeriv()
plt.subplot(224)
PlotSqrtDeriv()

plt.figure()
plt.subplot(211)
PlotLnExp()
plt.subplot(212)
PlotExpLn()

plt.figure()

plt.subplot(211)
PlotTanh()

plt.subplot(212)
PlotTanhDeriv()

plt.draw()
plt.show()

