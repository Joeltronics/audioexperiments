# import scipy
from scipy import special as sp
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

# Is = Diode current parameter
# n = Diode ideality (1-2)
# Rs = Series resistance before diode
# Rp = Resistance parallel to diode
# T = temperature (celsius)

# Sweeping T isn't supported until I find these values (they're probably in SciPy already?)
# kB =
# q =

ymax = 2  # 1.5
xmax = 3  # 2
# sweep = {'Type':1, 'n':0, 'Rs':0, 'Is':0, 'T':0}
sweep = {'Type': 0, 'n': 0, 'Rs': 1, 'Is': 0, 'T': 0}
nplot = 10000

params = []

# Measured diodes from:
# http://www.bentongue.com/xtalset/16MeaDio/16MeaDio.html

if sweep['Type']:
	# TODO: find actual numbers!
	# params.append({ 'name':'Silicon, N=1', 'Is':1E-12, 'n':1, 'Rs':1000, 'T':27})
	# params.append({ 'name':'Silicon, N=1.5', 'Is':1E-12, 'n':1.5, 'Rs':1000, 'T':27})
	# params.append({ 'name':'Silicon, N=2', 'Is':1E-12, 'n':2, 'Rs':1000, 'T':27})
	# params.append({ 'name':'Germanium, N=1', 'Is':1E-5, 'n':1, 'Rs':1000, 'T':27})
	# params.append({ 'name':'Germanium, N=1.5', 'Is':1E-5, 'n':1.5, 'Rs':1000, 'T':27})
	# params.append({ 'name':'Germanium, N=2', 'Is':1E-5, 'n':2, 'Rs':1000, 'T':27})
	# params.append({ 'name':'LED, N=1', 'Is':1E-20, 'n':1, 'Rs':1000, 'T':27})
	# params.append({ 'name':'LED, N=1.5', 'Is':1E-20, 'n':1.5, 'Rs':1000, 'T':27})
	# params.append({ 'name':'LED, N=2', 'Is':1E-20, 'n':2, 'Rs':1000, 'T':27})

	params.append({'name': '1N4148 min, n=1', 'Is': 4E-12, 'n': 1, 'Rs': 1000, 'T': 27})
	params.append({'name': '1N4148 max, n=1', 'Is': 7E-12, 'n': 1, 'Rs': 1000, 'T': 27})
	params.append({'name': '1N4148 min, n=2', 'Is': 4E-12, 'n': 2, 'Rs': 1000, 'T': 27})
	params.append({'name': '1N4148 max, n=2', 'Is': 7E-12, 'n': 2, 'Rs': 1000, 'T': 27})

	# params.append({ 'name':'1N4148 estimated', 'Is':5E-12, 'n':1.8, 'Rs':4700, 'T':27})
	# params.append({ 'name':'TS-808 min gain', 'Is':5E-12, 'n':1.8, 'Rs':4700, 'Rp':51000, 'T':27})
	# params.append({ 'name':'TS-808 max gain', 'Is':5E-12, 'n':1.8, 'Rs':4700, 'Rp':551000, 'T':27})

	params.append({'name': 'Measured 1N4148', 'Is': 1.23E-12, 'n': 1.73, 'Rs': 1000, 'T': 27})  # 710k
	params.append({'name': 'Measured 1N4148', 'Is': 3.10E-12, 'n': 1.89, 'Rs': 1000, 'T': 27})  # 11k
	params.append({'name': 'Measured 1N4148', 'Is': 6.70E-12, 'n': 2.18, 'Rs': 1000, 'T': 27})  # 170

	params.append({'name': '1N4148, n=1.65', 'Is': 5.5E-12, 'n': 1.65, 'Rs': 1000, 'T': 27})

# params.append({ 'name':'Measured 1N34A 1k', 'Is':3500E-12, 'n':1.71, 'Rs':1000, 'T':27}) # 710k
# params.append({ 'name':'Measured 1N34A 1', 'Is':1100E-12, 'n':1.28, 'Rs':1000, 'T':27}) # 22k
# params.append({ 'name':'Measured 1N34A 1', 'Is':720E-12, 'n':1.08, 'Rs':1000, 'T':27}) # 170
# params.append({ 'name':'Measured 1N34A 2', 'Is':230E-12, 'n':1.28, 'Rs':1000, 'T':27}) # 47k
# params.append({ 'name':'Measured 1N34A 2', 'Is':160E-12, 'n':1.13, 'Rs':1000, 'T':27}) # 2.8k
# params.append({ 'name':'Measured 1N34A 2', 'Is':160E-12, 'n':1.13, 'Rs':1000, 'T':27}) # 37
# params.append({ 'name':'Measured 1N404A Ge B-E', 'Is':1540E-12, 'n':1.01, 'Rs':1000, 'T':27}) # 56k
# params.append({ 'name':'Measured high Is Schottky', 'Is':265E-12, 'n':1.15, 'Rs':1000, 'T':27}) # 360
# params.append({ 'name':'Measured low Is Schottky', 'Is':103E-12, 'n':1.03, 'Rs':1000, 'T':27}) # 151
# params.append({ 'name':'Measured quad Schottky', 'Is':72E-12, 'n':1.02, 'Rs':1000, 'T':27}) # 117

if sweep['n']:
	for n in [1, 1.25, 1.5, 1.75, 2]:
		params.append({'name': 'n=' + str(n), 'Is': 1E-12, 'n': n, 'Rs': 1000, 'T': 27})

if sweep['Is']:
	for Is in [1E-14, 1E-12, 1E-10, 1E-8, 1E-6]:
		params.append({'name': 'Is=' + str(Is), 'Is': Is, 'n': 1, 'Rs': 1000, 'T': 27})

if sweep['Rs']:
	for R in [1, 10, 100, 1000, 10000, 100000]:
		if (R < 1000):
			params.append({'name': 'R=' + str(R), 'Is': 1E-12, 'n': 1, 'Rs': R, 'T': 27})
		else:
			params.append({'name': 'R=' + str(R / 1000) + 'k', 'Is': 1E-12, 'n': 1, 'Rs': R, 'T': 27})

if sweep['T']:
	for T in [0, 10, 27, 40, 70]:
		params.append({'name': 'T=' + str(T) + ' C', 'Is': 1E-12, 'n': 1, 'Rs': 1000, 'T': T})

# First plot diode IV curves

plt.figure()

x = np.linspace(0, 1, nplot)

for param in params:
	Rs = float(param['Rs'])
	Is = float(param['Is'])
	n = float(param['n'])
	T = float(param['T'] + 273)  # convert Celsius to Kelvin

	Vt = 8.61733238E-5 * T
	# Vt = 0.026 # fixed value for ~300K

	Vd = x
	Id = Is * (np.exp(Vd / (n * Vt)) - 1)
	y = Id

	plt.plot(x, y, label=param['name'])

plt.grid()
plt.legend(loc=2)
plt.title('Diode I-V curves')
plt.ylim([0, 20e-6])
plt.xlabel('V')
plt.ylabel('I')

# Now plot clipped diode

y = np.linspace(0, ymax, nplot)

plt.figure()

plt.plot(y, y, label='y = x')
for param in params:
	Rs = float(param['Rs'])
	Is = float(param['Is'])
	n = float(param['n'])
	T = float(param['T'] + 273)  # convert Celsius to Kelvin

	Vt = 8.61733238E-5 * T
	# Vt = 0.026 # fixed value for ~300K

	Id = Is * (np.exp(y / (n * Vt)) - 1)

	if 'Rp' in param:
		Rp = float(param['Rp'])
		x = y * (1 + Rs / Rp) + Rs * Id
	else:
		x = y + Rs * Id

	plt.plot(x, y, label=param['name'])

plt.grid()
plt.legend()
plt.xlabel('Vin')
plt.ylabel('Vout')
plt.xlim(0, xmax)
plt.ylim(0, ymax)
plt.xticks(np.arange(0, xmax, 0.25))
plt.yticks(np.arange(0, ymax, 0.25))

plt.draw()
plt.show()
