#!/usr/bin/env python3

"""
Solves this circuit:
	          Rs
	Vin ---\/\/\/\/---+-----+--- Vout
	                  |     |
	                  |     >
	                  |     >  Rd
	                  >     >
	              Rp  >     |
	                  >     V  D
	                  |     -
	                  |     |
	                  +--+--+
	                     |
	                    Gnd

according to Shockley Diode Equation
"""

import argparse
from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, Tuple


@dataclass(frozen=True)
class Diode:
	Is: float
	n: float
	T: float = 27.0

	@property
	def Vt(self):
		# Vt = k * T / q
		# Approx 0.026 at 300 K
		temp_K = self.T + 273.15
		return 8.61733238E-5 * temp_K


@dataclass(frozen=True)
class DiodeCircuit:
	name: str
	diode: Diode
	Rs: float = 1000.0
	Rd: float = 0.0
	Rp: Optional[float] = None  # None = open circuit (i.e. infinite resistance)

	@classmethod
	def make(cls, Is: float, n: float, T=27.0, **kwargs):
		return cls(diode=Diode(Is=Is, n=n, T=T), **kwargs)


"""
Shockley diode equation:

Id = Is * exp( Vd / (n * Vt) - 1 )

Id: diode current
Vd: voltage across diode
Is: reverse bias saturation current (or scale current); typically 10^-12 A
Vt: thermal voltage (Vt = kT/q), 0.026 V at room temperature
n:  ideality factor, typically 1-2

The -1 term can be ignored when Vd >> n*Vt

===== Basic case -  no Rp, no Rd =====

Vout = Vd
Vout = Vin - Id * Rs

Vout = Vin - Rs * Is * exp(Vout / (n*Vt) - 1)
Vin = Vout + Rs * Is * exp(Vout / (n*Vt) - 1)

Inverting this involves Lambert W function, which isn't easy to calculate. We'll just calculate the inverse for now
TODO: solve it using scipy.special.lambertw
TODO: solve with Newton-Raphson

Id = Is * exp(Vout / (n*Vt) - 1)
x = y + Rs * Id

===== Full case - with Rp & Rd =====

Vout = Vd + Id * Rd

Vin = Vout + I * Rs
Vin = Vout + (Irp + Id) * Rs
Vin = Vout + (Vout / Rp + Id) * Rs

We can already calculate this if we calculate Vout first, but let's put them together to get Vin as a function of only Vd:
Vin = Vd + Id * Rd + Rs * ((Vd + Id * Rd) / Rp + Id)
Vin = Vd + Id * Rd + (Vd + Id * Rd) * (Rs / Rp) + Rs * Id
Vin = Vd + (Id * Rd) + (Id * Rs) + (Vd * Rs / Rp) + (Id * Rd * Rs / Rp)

TODO: turn these into just Vin as a function of Vout (i.e. without needing Vd as an input)
"""


def calculate_iv_curve(diode: Diode, Vd: np.ndarray) -> np.ndarray:
	Is = diode.Is
	n = diode.n
	Vt = diode.Vt
	return Is * (np.exp(Vd / (n * Vt)) - 1.0)


def calculate_clip_from_vd(circuit: DiodeCircuit, v_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	:returns: v_in, v_out
	"""
	Rs = circuit.Rs
	Is = circuit.diode.Is
	n = circuit.diode.n
	Vt = circuit.diode.Vt
	Rp = circuit.Rp
	Rd = circuit.Rd

	Id = Is * (np.exp(v_d / (n * Vt)) - 1.0)
	Vd = v_d

	if Rp is None:
		Vin = Vd + (Rd * Id) + (Rs * Id)
	else:
		Vin = Vd + (Rd * Id) + (Rs * Id) + (Vd * Rs / Rp) + (Id * Rd * Rs / Rp)

	Vout = Vd + Id * Rd

	return Vin, Vout


def plot(args):

	parser = argparse.ArgumentParser()

	group = parser.add_argument_group('Various diodes to plot')
	group.add_argument('--type', dest='sweep_type', action='store_true', help='Plot various diode types')
	group.add_argument('--silicon', dest='sweep_silicon', action='store_true', help='Plot silicon diode (1N4148) datasheet values')
	group.add_argument('--measured', dest='sweep_measured', action='store_true', help='Plot various measured diodes')
	group.add_argument('--ts808', dest='sweep_ts808', action='store_true', help='Plot diodes in ts808 feedback path')

	group = parser.add_argument_group('Sweep diode parameters')
	group.add_argument('--temp', dest='sweep_temperature', action='store_true', help='Sweep temperature')
	group.add_argument('-n', dest='sweep_n', action='store_true', help='Sweep diode N')
	group.add_argument('--rs', dest='sweep_rs', action='store_true', help='Sweep shunt resistance')
	group.add_argument('--rp', dest='sweep_rp', action='store_true', help='Sweep parallel resistance')
	group.add_argument('--rd', dest='sweep_rd', action='store_true', help='Sweep diode series resistance')
	group.add_argument('--rsrd', dest='sweep_rs_rd', action='store_true', help='Sweep diode shunt & series resistance (with constant total)')
	group.add_argument('--rprd', dest='sweep_rp_rd', action='store_true', help='Sweep various combos of Rp & Rd')
	group.add_argument('--is', dest='sweep_is', action='store_true', help='Sweep diode saturation current')

	group = parser.add_argument_group('Plot parameters')
	group.add_argument('--xrange', dest='x_range', default=3.0, type=float, help='Clip plot X range')
	group.add_argument('--yrange', dest='y_range', default=2.0, type=float, help='Clip plot Y range')

	args = parser.parse_args(args)

	# TODO: can we check the group directly?
	if not any((
			args.sweep_type,
			args.sweep_silicon,
			args.sweep_ts808,
			args.sweep_measured,
			args.sweep_n,
			args.sweep_rs,
			args.sweep_rd,
			args.sweep_rs_rd,
			args.sweep_is,
			args.sweep_temperature,
			args.sweep_rp,
			args.sweep_rp_rd)):
		args.sweep_type = True

	nplot = 10000

	plot_iv = False
	plot_clip = True
	circuits = []

	if args.sweep_type:
		plot_iv = True
		circuits.append(DiodeCircuit.make(name='Silicon, N=1', Is=1E-12, n=1))
		circuits.append(DiodeCircuit.make(name='Silicon, N=1.5', Is=1E-12, n=1.5))
		circuits.append(DiodeCircuit.make(name='Silicon, N=2', Is=1E-12, n=2))
		circuits.append(DiodeCircuit.make(name='Germanium, N=1', Is=1E-5, n=1))
		circuits.append(DiodeCircuit.make(name='Germanium, N=1.5', Is=1E-5, n=1.5))
		circuits.append(DiodeCircuit.make(name='Germanium, N=2', Is=1E-5, n=2))
		circuits.append(DiodeCircuit.make(name='LED, N=1', Is=1E-20, n=1))
		circuits.append(DiodeCircuit.make(name='LED, N=1.5', Is=1E-20, n=1.5))
		circuits.append(DiodeCircuit.make(name='LED, N=2', Is=1E-20, n=2))

	if args.sweep_silicon:
		plot_iv = True
		circuits.append(DiodeCircuit.make(name='1N4148 min, n=1', Is=4E-12, n=1))
		circuits.append(DiodeCircuit.make(name='1N4148 max, n=1', Is=7E-12, n=1))
		circuits.append(DiodeCircuit.make(name='1N4148 min, n=2', Is=4E-12, n=2))
		circuits.append(DiodeCircuit.make(name='1N4148 max, n=2', Is=7E-12, n=2))

	if args.sweep_ts808:
		circuits.append(DiodeCircuit.make(name='1N4148 estimated', Is=5E-12, n=1.8, Rs=4700))
		circuits.append(DiodeCircuit.make(name='TS-808 min gain', Is=5E-12, n=1.8, Rs=4700, Rp=51000))
		circuits.append(DiodeCircuit.make(name='TS-808 max gain', Is=5E-12, n=1.8, Rs=4700, Rp=551000))

	if args.sweep_measured:
		plot_iv = True

		# Measured diodes from:
		# https://web.archive.org/web/20161221074456/http://www.bentongue.com/xtalset/16MeaDio/16MeaDio.html

		circuits.append(DiodeCircuit.make(name='Measured 1N4148 @ 710k', Is=1.23E-12, n=1.73))  # 710k
		circuits.append(DiodeCircuit.make(name='Measured 1N4148 @ 11k', Is=3.10E-12, n=1.89))  # 11k
		circuits.append(DiodeCircuit.make(name='Measured 1N4148 @ 170R', Is=6.70E-12, n=2.18))  # 170

		circuits.append(DiodeCircuit.make(name='1N4148, n=1.65', Is=5.5E-12, n=1.65))

		circuits.append(DiodeCircuit.make(name='Measured 1N34A 1k', Is=3500E-12, n=1.71)) # 710k
		circuits.append(DiodeCircuit.make(name='Measured 1N34A 1', Is=1100E-12, n=1.28)) # 22k
		circuits.append(DiodeCircuit.make(name='Measured 1N34A 1', Is=720E-12, n=1.08)) # 170
		circuits.append(DiodeCircuit.make(name='Measured 1N34A 2', Is=230E-12, n=1.28)) # 47k
		circuits.append(DiodeCircuit.make(name='Measured 1N34A 2', Is=160E-12, n=1.13)) # 2.8k
		circuits.append(DiodeCircuit.make(name='Measured 1N34A 2', Is=160E-12, n=1.13)) # 37

		circuits.append(DiodeCircuit.make(name='Measured 1N404A Ge B-E', Is=1540E-12, n=1.01)) # 56k
		circuits.append(DiodeCircuit.make(name='Measured high Is Schottky', Is=265E-12, n=1.15)) # 360
		circuits.append(DiodeCircuit.make(name='Measured low Is Schottky', Is=103E-12, n=1.03)) # 151
		circuits.append(DiodeCircuit.make(name='Measured quad Schottky', Is=72E-12, n=1.02)) # 117

	if args.sweep_n:
		plot_iv = True
		for n in [1, 1.25, 1.5, 1.75, 2]:
			circuits.append(DiodeCircuit.make(name=f'n={n}', Is=1E-12, n=n))

	if args.sweep_is:
		plot_iv = True
		for Is in [1E-14, 1E-12, 1E-10, 1E-8, 1E-6]:
			circuits.append(DiodeCircuit.make(name=f'Is={Is}', Is=Is, n=1))

	if args.sweep_rs:
		for Rs in [1, 10, 100, 1000, 10000, 100000]:
			if Rs < 1000:
				circuits.append(DiodeCircuit.make(name=f'Rs={Rs}', Is=1E-12, n=1, Rs=Rs))
			else:
				circuits.append(DiodeCircuit.make(name=f'Rs={Rs//1000}k', Is=1E-12, n=1, Rs=Rs))

	if args.sweep_rd:
		for Rd in [0, 10, 100, 1000, 10000, 100000]:
			if Rd < 1000:
				circuits.append(DiodeCircuit.make(name=f'Rs=1k, Rd={Rd}', Is=1E-12, n=1, Rs=1000, Rd=Rd))
			else:
				circuits.append(DiodeCircuit.make(name=f'Rs=1k, Rd={Rd//1000}k', Is=1E-12, n=1, Rs=1000, Rd=Rd))

	if args.sweep_rs_rd:
		for Rd in [0, 100, 250, 500, 750, 900, 1000]:
			Rs = 1000 - Rd
			circuits.append(DiodeCircuit.make(name=f'Rs={Rs}, Rd={Rd}', Is=1E-12, n=1, Rs=Rs, Rd=Rd))

	if args.sweep_rp:
		for Rp in [1, 10, 100, 1000, 10000, 100000]:
			if Rp < 1000:
				circuits.append(DiodeCircuit.make(name=f'Rs=1k, Rp={Rp}', Is=1E-12, n=1, Rp=Rp))
			else:
				circuits.append(DiodeCircuit.make(name=f'Rs=1k, Rp={Rp//1000}k', Is=1E-12, n=1, Rp=Rp))

	if args.sweep_rp_rd:
		circuits.append(DiodeCircuit.make(name=f'Rs=1k, Rp=1k, Rd=0', Is=1E-12, n=1, Rs=1000, Rp=1000, Rd=0))
		circuits.append(DiodeCircuit.make(name=f'Rs=1k, Rp=1k, Rd=100', Is=1E-12, n=1, Rs=1000, Rp=1000, Rd=100))
		circuits.append(DiodeCircuit.make(name=f'Rs=1k, Rp=10k, Rd=100', Is=1E-12, n=1, Rs=1000, Rp=10000, Rd=100))
		circuits.append(DiodeCircuit.make(name=f'Rs=1k, Rp=10k, Rd=1k', Is=1E-12, n=1, Rs=1000, Rp=10000, Rd=1000))

	if args.sweep_temperature:
		plot_iv = True
		for T in [0, 10, 27, 40, 70]:
			circuits.append(DiodeCircuit.make(name=f'T={T} C', Is=1E-12, n=1, T=T))

	if plot_iv:

		plt.figure()

		v = np.linspace(0, 1, nplot)

		for circuit in circuits:
			i = calculate_iv_curve(circuit.diode, v)
			plt.plot(v, i, label=circuit.name)

		plt.grid()
		plt.legend(loc=2)
		plt.title('Diode I-V curves')
		plt.ylim([0, 20e-6])
		plt.xlabel('V')
		plt.ylabel('I')

	if plot_clip:

		xmax = args.x_range
		ymax = args.y_range

		plt.figure()

		v_d = np.linspace(0, ymax, nplot)

		plt.plot(v_d, v_d, label='y = x')
		for circuit in circuits:
			v_in, v_out = calculate_clip_from_vd(circuit, v_d=v_d)
			plt.plot(v_in, v_out, label=circuit.name)

		plt.grid()
		plt.legend()
		plt.xlabel('Vin')
		plt.ylabel('Vout')
		plt.xlim(0, xmax)
		plt.ylim(0, ymax)
		plt.xticks(np.arange(0, xmax, 0.25))
		plt.yticks(np.arange(0, ymax, 0.25))

	plt.show()


def main(args):
	plot(args)
