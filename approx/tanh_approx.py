
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.misc
import struct

# http://math.stackexchange.com/questions/107292/rapid-approximation-of-tanhx

order = 7
xRange = 20

log2e = math.log2(math.e)

def punFloatToInt(x):
	fx = float(x)
	return struct.unpack("@i", struct.pack("@f", fx))[0]

def punIntToFloat(x):
	ix = int(x)
	return struct.unpack("@f", struct.pack("@i", ix))[0]

def exp2Approx(x):
	c1 = float(0x00800000)
	c2 = float(0x3f800000)
	#c2 = float(0x3f78acfb)
	y_int = int(x*c1 + c2)
	return punIntToFloat(y_int)

def expApprox(x):
	c1 = float(0x00800000)*log2e
	#c2 = float(0x3f800000)
	c2 = float(0x3f78acfb)
	y_int = int(x*c1 + c2)
	return punIntToFloat(y_int)

def tanhApproxTypePun(x):
	
	y = np.zeros_like(x)
	for n, xx in enumerate(x):
		expX = expApprox(xx)
		expNegX = expApprox(-xx)
		yy = (expX - expNegX) / (expX + expNegX)
		y[n] = yy
	
	return y
	

def nCr(n, r):
	return scipy.misc.comb(n, r)

def tanhApproxContinuedFraction5(x):
	# 2 multiply
	# 5 divide
	# 6 add
	
	x2 = np.square(x) # 1 mult

	y = 9.0 + x2 / 11.0 # 1 add, 1 mult
	y = 7.0 + x2 / y # 1 add, 1 div
	y = 5.0 + x2 / y # 1 add, 1 div
	y = 3.0 + x2 / y # 1 add, 1 div
	y = 1.0 + x2 / y # 1 add, 1 div
	y = x / y # 1 div
	
	return y

def tanhApproxContinuedFraction3(x):
	# 2 multiply
	# 3 divide
	# 3 add
	# 1 mult + add can be saved with MAC
	
	x2 = np.square(x) # 1 mult

	y = 5.0 + x2 / 7.0 # 1 add, 1 mult / MAC
	y = 3.0 + x2 / y # 1 add, 1 div
	y = 1.0 + x2 / y # 1 add, 1 div
	y = x / y # 1 div
	
	return y
	
def tanhApproxContinuedFraction(x, order):
	
	x2 = np.square(x)

	#coeffs = [1, 3, 5, 7, 9, 11, ...]
	coeffs = range(1, 2*order, 2)
	y = coeffs[-1]
	
	for n in range(order, 0, -1):
		y = coeffs[n-1] + x2 / y
	
	y = x / y
	
	return y


def tanhApproxPade3(x):

	print('tanhApproxPade3')

	if False:
		
		# Requires a lot of extra allocation
		# 1 divide
		# 3 MAC
		# 5 multiply
		# 2 add
		# (1 div, 10 mult/add/MAC)
		
		x2 = np.square(x) # 1 mult
		x4 = np.square(x2) # 1 mult
		x6 = x2 * x4 # 1 mult
		
		num = x * (10.0 + x2) * (60.0 + x2) # 2 add, 2 mult
		den = 600.0 + 270.0*x2 + 11.0*x4 + x6*(1.0/24.0) # 3 MAC
		
		return num / den # div
		
	elif True:
		
		# Just factor out x
		# 4 vector alloc
		# 1 divide
		# 4 MAC
		# 4 mult
		# 1 add
		# (4 vect, 1 div, 9 mult/add/MAC)
		
		x2 = np.square(x) # 1 mult, 1 alloc
		x4 = np.square(x2) # 1 mult, 1 alloc
		x6 = x2 * x4 # 1 mult
		
		# num = x*(600.0 + 70.0*x2 + x4)
		num = 600.0 + 70.0*x2 # 1 MAC, 1 alloc
		num = num + x4 # 1 add
		num = num * x # 1 mult
		
		# den = 600.0 + 270.0*x2 + 11.0*x4 + x6/24.0
		den = 600.0 + 270.0*x2 # 1 MAC (no alloc: can reuse x)
		den = den + 11.0*x4 # 1 MAC
		den = den + (1.0/24.0)*x6 # 1 MAC
		
		return num / den # div
	
	elif True:
	
		# Fully expanded
		# 4 vector alloc
		# 1 divide
		# 5 MAC
		# 5 mult
		# (4 vect, 1 dif, 10 MAC/mult)
	
		x2 = np.square(x) # 1 mult, 1 alloc
		x4 = np.square(x2) # 1 mult, 1 alloc
		#x6 = np.square(x4) # 1 mult, 1 alloc
		x6 = x2 * x4
		x3 = x*x2 # 1 mult, 1 alloc
		
		#num = 600.0*x + 70.0*x3 + x5
		num = x4*x # 1 mult, 1 alloc
		num = num + 70.0*x3 # 1 MAC
		num = num + 600.0*x # 1 MAC
		
		# den = 600.0 + 270.0*x2 + 11.0*x4 + x6/24.0
		den = 600.0 + 270.0*x2 # 1 MAC (no alloc: can reuse x)
		den = den + 11.0*x4 # 1 MAC
		den = den + (1.0/24.0)*x6 # 1 MAC
		
		return num / den # div
	
	elif True:
		
		# Fully factored, with square term
		# slightly less efficient, but less allocation
		
		x2 = np.square(x)
		
		num = x * (10.0 + x2) * (60.0 + x2)
		den = 24. * (x2 + 1.82075) * (x2 - 2.96199*x + 3.70548) * (x2 + 2.91699*x + 2.70548)
		
		return num / den
	
	else:
		
		# Fully-fully factored
		# 3 vector allocations
		# 1 divide
		# 7 MAC
		# 5 mult
		# (3 vect, 1 div, 12 mult/MAC)
		
		# num = x * (10.0 + x*x) * (60.0 + x*x)
		
		term = x*x + 60.0 # MAC, 1 alloc
		num = x * term # mult, 1 alloc
		term = x*x + 10.0 # MAC
		num = num * term # mult
		
		# den = 24. * (x*x + 1.82075) * (x*x - 2.96199*x + 3.70548) * (x*x + 2.91699*x + 2.70548)
		
		term = x*x + 1.82075 # MAC
		den = 24.0 * term  # mult, 1 alloc
		
		term = -2.96199*x + 3.70548 # MAC
		term = x*x + term # MAC
		den = den * term # mult
		
		term = 2.91699*x + 2.70548 # MAC
		term = x*x + term # MAC
		den = den * term # mult
		
		return num / den # DIV
		

def padeN(x, n):
	
	y = 0.0
	for j in range(n):
		
		num = float(nCr(n, j)) * np.power(x,j)
		den = float(nCr(2*n, j) * math.factorial(j))
		
		y += num / den
	
	return y

def tanhApproxPade(x, order):
	pn = padeN(x, order)
	pnNeg = padeN(-x, order)
	
	pn2 = np.square(pn)
	pnNeg2 = np.square(pnNeg)
	
	num = pn2 - pnNeg2
	den = pn2 + pnNeg2
	
	return num/den

def tanhApproxTaylor(x, order):
	
	y = 1.0*x
	
	if order >= 3:
		x2 = x*x
		x3 = x2*x
		
		y -= (1.0/3.0)*x3
	
	if order >= 5:
		x5 = x3*x2
		y += (2.0/15.0)*x5
	
	if order >= 7:
		x7 = x5*x2
		y -= (17.0/315.0)*x7
	
	return y
	

x = np.linspace(-xRange, xRange, 1024)

yactual = np.tanh(x)

ycf = tanhApproxContinuedFraction(x, order)

if order == 3:
	yp = tanhApproxPade3(x)
else:
	yp = tanhApproxPade(x, order)

#yt = tanhApproxTaylor(x, order)
yt = tanhApproxTypePun(x)

if (order % 2 == 1):
	# if odd only
	nMin = np.argmin(ycf)
	nMax = np.argmax(ycf)
	
	print('Continued fraction:')
	print('Min: %.6f, %.6f' % (x[nMin], ycf[nMin]))
	print('Max: %.6f, %.6f' % (x[nMax], ycf[nMax]))

	nMin = np.argmin(yp)
	nMax = np.argmax(yp)
	
	print('Pade:')
	print('Min: %.12f, %.12f' % (x[nMin], yp[nMin]))
	print('Max: %.12f, %.12f' % (x[nMax], yp[nMax]))

ep = yp - yactual
ecf = ycf - yactual
et = yt - yactual

#ep = 1.0 - yp/yactual
#ecf = 1.0 - ycf/yactual
#et = 1.0 - yt/yactual

#ep = np.abs(ep)
#ecf = np.abs(ecf)
#et = np.abs(et)

plt.figure()

plt.subplot(311)
plt.plot(x, yp, label='Pade')
plt.plot(x, yp, label='CF')
plt.plot(x, yp, label='Type pun')
plt.plot(x, yp, label='Actual')
plt.grid()
plt.legend(loc=4)
plt.xlim([-2., 2.])
plt.ylim([-1.1, 1.1])

plt.subplot(312)
plt.plot(x, yp, label='Pade')
plt.plot(x, ycf, label='CF')
plt.plot(x, yt, label='Type pun')
plt.plot(x, yactual, label='Actual')
plt.grid()
plt.legend(loc=4)
plt.ylim([-1.1, 1.1])

plt.subplot(313)
plt.plot(x, ep, label='Pade')
plt.plot(x, ecf, label='CF')
plt.plot(x, et, label='Type pun')
plt.xlim([-4.7, 4.7])
#plt.ylim([0, 0.001])
plt.ylim([-0.01, 0.01])
plt.grid()
plt.title('Error')
plt.legend(loc=4)

plt.show()

