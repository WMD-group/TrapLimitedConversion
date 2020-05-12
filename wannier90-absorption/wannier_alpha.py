#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eV2Icm = 8.065E3 
hbar = 6.582119569E-16	# [eV*s]
# sigma_si2cgs = 9E11 # [S/cm] -> [1/sec] # 1/(4pi * 8.8541878128E-12) [F/m] 
vac_permittivity = 8.8541878128E-12 
c = 29979245800


def read_kubo(filename = 'wannier90-kubo_S_xx.dat'):
    data = pd.read_csv(filename, sep=r'\s+', names=['E', 's_real', 's_imag'])
    data = data[1:] # remove energy 0
    return data

def calc_eps(opt_conductivity):
    E = opt_conductivity.E           # eV
    w = E / hbar                     # [s-1]
    s_real = opt_conductivity.s_real #* sigma_si2cgs # [S/cm] -> [1/s]
    s_imag = opt_conductivity.s_imag #* sigma_si2cgs # [S/cm] -> [1/s]
    sigma = s_real + s_imag * 1j
    eps = 1 + 1j * np.pi * sigma / w / vac_permittivity * 100 # [si] [cm] -> [m]
    eps_imag = np.imag(np.array(eps))

    # find first occurence of eps_imag < 0 
    # ignore too small values
    indx = np.where(eps_imag < 5E-3)
    eps_imag[:indx[0][-1]] = 0
    eps_imag[np.where(np.abs(eps_imag) < 1E-5)] = 0

    eps = pd.DataFrame(data = {'E': E[1:], 'eps_imag': eps_imag[1:], 'eps_real': np.real(np.array(eps))[1:]})
    return eps


def KK_trans(eps):
    """
    Kramers-Kronig transform
    returns real part of epsilon
    """
    from scipy.integrate import simps
    from scipy.integrate import cumtrapz
    eps_real = []

    cshift = 0.001
    eps_imag = eps.eps_imag
    wp = eps.E / hbar

    for w in wp:
        eps_real.append(1 + 2 / np.pi * simps(eps_imag * wp / (wp**2 - w**2 + 1j*cshift), wp))
    return np.real(eps_real)


def calc_alpha(eps):
    """
    calculate and return absorption coefficient
    """
    w = eps.E / hbar                     # [s-1]
    N = np.sqrt(eps.eps_real + eps.eps_imag*1j)
    k = np.imag(np.array(N))
    alpha = 2 * w / c * k
    alpha = np.array(alpha)
    alpha[np.where(alpha < 1E-10)] = 1E-10
    return alpha


def plot_eps(eps, band_gap):
    ax = plt.subplot(211)

    if band_gap is not None:
        ax.plot([band_gap, band_gap], [-1E100, 1E100], ls='--')

    ax.plot(eps.E, eps.eps_real, label="real")
    ax.plot(eps.E, eps.eps_imag, label="imag")
    
    ax.set_xlim(0, 4)
    ax.set_ylim(-5, 10)
    ax.legend()
    
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Dielectric function")
    

def plot_eps_xyz(eps_xyz, eps_ave, band_gap):
    ax = plt.subplot(211)

    if band_gap is not None:
        ax.plot([band_gap, band_gap], [-1E100, 1E100], ls='--')

    for eps_i, label in zip(eps_xyz, ['x', 'y', 'z']):
        ax.plot(eps_i.E, eps_i.eps_real, ls='--', label="{}_real".format(label))
        ax.plot(eps_i.E, eps_i.eps_imag, label="{}_imag".format(label))        

    ax.plot(eps_ave.E, eps_ave.eps_real, c='k', ls='--', label="real")
    ax.plot(eps_ave.E, eps_ave.eps_imag, c='k', label="imag")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Dielectric function")
    ax.set_xlim(0, 4)
    ax.set_ylim(-0, 16)
    ax.legend()
    

def plot_alpha(E, alpha, band_gap):
    ax = plt.subplot(212)
    if band_gap is not None:
        ax.plot([1240/band_gap, 1240/band_gap], [-1E100, 1E100], ls='--')

    ax.plot(1240/E, np.log10(alpha))
    ax.set_xlabel("Wave length (nm)")
    ax.set_ylabel("log(alpha)")
    ax.set_xlim(200, 1400)
    ax.set_ylim(-0.5, 7)


def get_eps_xyz(dir):
    opt_conductivity = []
    for file in ["wannier90-kubo_S_xx.dat", "wannier90-kubo_S_yy.dat", "wannier90-kubo_S_zz.dat"]:
        opt_conductivity.append(read_kubo("{}/{}".format(dir, file)))
    
    eps = []
    for opt_i in opt_conductivity:
        eps.append(calc_eps(opt_i))

    return eps


def ave_eps(eps_xyz, mode="harmonic"):
    # geometric mean or harmonic mean
    # harmonic mean: why not?
    if mode == "harmonic":
        eps_imag = 3/np.sum([1/eps_i.eps_imag for eps_i in eps_xyz] , axis = 0)
        eps_real = 3/np.sum([1/eps_i.eps_real for eps_i in eps_xyz] , axis = 0)
    else:
        eps_imag = np.sum([eps_i.eps_imag for eps_i in eps_xyz] , axis = 0)/3.
        eps_real = np.sum([eps_i.eps_real for eps_i in eps_xyz] , axis = 0)/3.

    eps = pd.DataFrame(data = {'E': eps_xyz[0].E, 'eps_imag': eps_imag, 'eps_real': eps_real})

    return eps
    
def scissor(alpha, E, band_gap):
    # shift the onset of alpha to the band_gap
    # only for alpha
    # find onset 
    indx_onset = np.where(alpha > 1E-10)[0][0]
    E_onset = E.values[indx_onset]
    E_diff = E_onset - band_gap
    return E.values - E_diff


def save_alpha(E, alpha):
    np.savetxt("alpha.dat", np.array([E, alpha]).T)


def main(dir, band_gap = None):
    # read_kubo(filename = 'wannier90-kubo_S_xx.dat'):
    eps_xyz = get_eps_xyz(dir)
    eps = ave_eps(eps_xyz, mode="geometric")
    alpha = calc_alpha(eps)
    print(eps[:1])

    E_scissor = scissor(alpha, eps.E, band_gap)
    save_alpha(E_scissor, alpha)

    # plot_eps(eps, band_gap)
    plot_eps_xyz(eps_xyz, eps, band_gap)
    plot_alpha(E_scissor, alpha, band_gap)

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculte dielectric functions and absorption coefficient from optical conductivity calculated by Wannier90.')
    parser.add_argument('dir', metavar='D', type=str,
                    help='directory where wannier-kubo_S_{xx,yy,zz}.dat are')
    parser.add_argument('band_gap', metavar='G', type=float,
                    help='an direct band gap of material (eV)')

    args = parser.parse_args()

    main(args.dir, args.band_gap)
