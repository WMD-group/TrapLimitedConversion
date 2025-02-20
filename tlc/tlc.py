import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.constants as scpc
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar, minimize_scalar

from scfermi import Scfermi, run_scfermi_all

kb_in_eV_per_K = scpc.physical_constants["Boltzmann constant in eV/K"][0]  # 8.6173303e-5 eV K-1, Boltzmann constant
sun_power = 100.   # AM1.5G standard irradiance in mW/cm^2; https://www.pveducation.org/pvcdrom/appendices/standard-solar-spectra

# for reference: (but use scipy.constants values for consistency and accuracy)
# k = 1.38064852e-23     # m^2 kg s^-2 K^-1, Boltzmann constant
# h = 6.62607004e-34     # m^2 kg s^-1    , planck constant
# c = 2.99792458e8       # m s^-1         , speed of light
# eV = 1.6021766208e-19  # joule        , eV to joule
# e = 1.6021766208e-19   # C             , elemental charge

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
ref_solar = pd.read_csv(os.path.join(MODULE_DIR, "../data/ASTMG173.csv"), header=1)  #
# http://rredc.nrel.gov/solar/spectra/am1.5 ; Units: nm vs W m^-2 nm^-1
# data range: 280nm to 4000nm, 0.31eV to 4.42857 eV

# WL = 'wavelength' (nm), solar_per_nm = 'solar irradiance' per inverse nanometer (per energy); W m^-2 nm^-1: 
WL, solar_per_nm = ref_solar.iloc[:, 0], ref_solar.iloc[:, 2]

# convert nm to eV as energy units:
def ev_to_or_from_nm(eV_or_nm: float):  
    """
    Converts energy in eV to wavelength in nm, or vice versa.
    
    Copied from https://github.com/SMTG-Bham/sumo/pull/134 from
    @kavanase.

    Args:
        eV_or_nm (float):
            Input wavelength (in nm) or energy (in eV).

    Returns:
        nm_or_eV (float):
            Output energy (in eV) or wavelength (in nm).
    """
    return  1e9*(scpc.h*scpc.c) / (eV_or_nm*scpc.electron_volt)

E = ev_to_or_from_nm(WL)  # eV
solar_per_E = solar_per_nm * (scpc.eV/1e-9) * scpc.h * scpc.c / (scpc.eV*E)**2  # jacobian transformation, converts solar irradiance to: W m^-2 eV^-1
Es = np.arange(0.32, 4.401, 0.002)  # equally-spaced energy spectrum for solar irradiance

# linear interpolation to get an equally spaced spectrum
AM15 = np.interp(Es, E[::-1], solar_per_E[::-1])  # AM15 (standard) solar irradiance in W m^-2 eV^-1
AM15flux = AM15 / (Es*scpc.eV)  # AM15 solar flux; number of incident photons in m^-2 eV^-1 s^-1

# code to parse the AM1.5G spectrum from NREL into solar irradiance and flux has been tempated from
# https://github.com/marcus-cmc/Shockley-Queisser-limit; C. Marcus Chuang 2016

class Trap():
    """
    Class Trap
    """
    def __init__(self, D, E_t1, E_t2, N_t, q1, q2, q3, g, C_p1, C_p2, C_n1, C_n2):
        """
        initialise Trap class

        Args:
        D: defect name
        E_t1: energy level 1 (eV)
        E_t2: energy level 2 (eV)
        N_t: total trap concentration (cm^-3)
        q1: charge state 1
        q2: charge state 2
        q3: charge state 3
        g: degeneracy factor
        C_p1: hole capture coefficient for defect 1
        C_p2: hole capture coefficient for defect 2
        C_n1: electron capture coefficient for defect 1
        C_n2: electron capture coefficient for defect 2
        """
        self.D = D
        self.E_t1 = E_t1
        self.E_t2 = E_t2
        self.N_t = N_t
        self.q1 = q1
        self.q2 = q2
        self.q3 = q3
        self.g = g
        # capture coeff (avoiding div by 0)
        self.C_p1 = C_p1 if C_p1 > 0 else 1E-100
        self.C_n1 = C_n1 if C_n1 > 0 else 1E-100
        self.C_p2 = C_p2 if C_p2 > 0 else 1E-100
        self.C_n2 = C_n2 if C_n2 > 0 else 1E-100
        self.name = "${{{}}} ({}/{}/{})$".format(D, q1, q2, q3) 

    def rate(self, n0, p0, delta_n, N_n, N_p, e_gap, temp):
        """
        calculate defect-mediated nonradiative recombination rate

        Args:
        n0: equilibrium electron concentration (cm^-3)
        p0: equilibrium hole concentration (cm^-3)
        delta_n: excess carrier concentration (cm^-3)
        N_n: effective electron concentration (cm^-3)
        N_p: effective hole concentration (cm^-3)
        e_gap: band gap (eV)
        temp: temperature (K)
        """
        n = n0 + delta_n
        p = p0 + delta_n

        if self.q3 ==00:
           n1 = N_n*np.exp(-(e_gap-self.E_t1)/kb_in_eV_per_K/temp)
           p1 = N_p*np.exp(-self.E_t1/kb_in_eV_per_K/temp)

           R = (n*p - n0*p0)/((p+p1)/(self.N_t*self.C_n1) + (n+n1)/(self.N_t*self.C_p1))

        else:
           P1=p*self.C_p1+1/self.g*self.C_n1*N_n*np.exp(-(e_gap-self.E_t1)/kb_in_eV_per_K/temp)
           P2=p*self.C_p2+self.g*self.C_n2*N_n*np.exp(-(e_gap-self.E_t2)/kb_in_eV_per_K/temp)
           N1=n*self.C_n1+self.g*self.C_p1*N_p*np.exp(-self.E_t1/kb_in_eV_per_K/temp)
           N2=n*self.C_n2+1/self.g*self.C_p2*N_p*np.exp(-self.E_t2/kb_in_eV_per_K/temp)

           R = (n*p - n0*p0)*((self.C_n1*self.C_p1*P2+self.C_n2*self.C_p2*N1)/(N1*P2+P1*P2+N1*N2))*self.N_t

        return R

    def __repr__(self):
        repr = "{}    ({}/{}/{})  {} {:.2E}  {} {} {:.2E}  {:.2E}  {:.2E}  {:.2E}".format(self.D,
                                                                  self.q1, self.q2, self.q3, self.g, self.N_t, self.E_t1,self.E_t2,self.C_n1, self.C_n2,self.C_p1,self.C_p2)
        return repr

    def __str__(self):
        repr = "{}    ({}/{}/{})  {} {:.2E}  {} {} {:.2E}  {:.2E}  {:.2E}  {:.2E}".format(self.D,
                                                                  self.q1, self.q2, self.q3, self.g, self.N_t, self.E_t, elf.E_t1,self.E_t2,self.C_n1, self.C_n2,self.C_p1,self.C_p2)
        return repr


class tlc(object):
    """
    Class tlc

    ALPHA_FILE: optical absorption coefficient data
    SCFERMI_FILE: sc-fermi file containing defect formation energies, charge states and degeneracy factors
    TRAP_FILE: trap file containing defect levels, charge states and capture coefficients
    """

    ALPHA_FILE = "alpha.csv"
    SCFERMI_FILE = "input-fermi.dat"
    TRAP_FILE = "trap.dat"

    def __init__(self, E_gap, T=300, Tanneal=835, thickness=2000, intensity=1.0, l_sq=False, poscar_path="POSCAR", totdos_path="totdos.dat"):
        """
        initialise tlc class

        Args:
        E_gap: band gap (eV)
        T: operating temperature (K)
        Tanneal: annealing temperature (K)
        thickness: film thickness (nm)
        intensity: light concentration, 1.0 = one Sun, 100 mW/cm^2
        l_sq: Shockley-Queisser limit (True) or Trap limited conversion efficiency (False)
        poscar_path: POSCAR file path
        totdos_path: total DOS file path
        """
        try:
            E_gap, T, thickness, intensity = float(E_gap), float(
                T), float(thickness), float(intensity)
        except:
            raise ValueError(
                "Invalid input for E_gap, T, thickness, or intensity")

        if T <= 0 or E_gap < 0.31:
            raise ValueError("T must be greater than 0 and " +
                             "E_gap cannot be less than 0.31")
        self.Vs = np.arange(-0.1, E_gap, 0.001)
        self.T = T
        self.Tanneal = Tanneal
        self.E_gap = E_gap
        self.thickness = thickness
        self.intensity = intensity  # TODO: Fully implement and remove not in docstring above
        self.Es = Es  # np.arange(0.32, 4.401, 0.002)
        self.l_calc = False
        self.poscar_path = poscar_path
        self.totdos_path = totdos_path
        self.l_sq = l_sq
        if not l_sq:
            self._calc_absorptivity()
        else:
            self.absorptivity = np.heaviside(Es - self.E_gap, 1)  # unit-less
            self.alpha = pd.DataFrame(
                {"E": Es, "alpha": np.heaviside(Es - self.E_gap, 1) * 1E100})
        self.scfermi = None
        self.R_SRH = None
        # self.WLs = np.arange(280, 4001, 1.0)
        # self.AM15nm = np.interp(self.WLs, WL, solar_per_nm)

    def __repr__(self):
        """
        return string representation of tlc class with input params
        """
        if self.l_sq:
            s = "Shockley-Queisser limit (SQ limit)\n"
        else:
           s = "Trap limited conversion efficiency (TLC)\n"
        s += "T: {:.1f} K\n".format(self.T)
        s += "E_gap: {:.2f} eV\n".format(self.E_gap)
        s += "Thickness: {:.1f} nm".format(self.thickness)
        if self.l_calc:
            s += "\n===\n"
            s += "J_sc: {:.3f} mA/cm^2\n".format(self.j_sc)
            s += "J0_rad: {:.3g} mA/cm^2\n".format(self.j0_rad)
            s += "V_oc: {:.3f} V\n".format(self.v_oc)
            s += "V_max, J_max: {:.3f} V, {:.3f} mA/cm^2\n".format(
                self.v_max, self.j_max)
            s += "FF: {:.3f}%\n".format(self.ff*100)
            s += "Efficiency: {:.3f}%".format(self.efficiency*100)
        return s

    def calculate_SRH(self):
        """
        get defect-mediated nonradiative recombination rate
        """
        self._get_scfermi(tlc.SCFERMI_FILE)
        self._run_scfermi(self.Tanneal, self.T, self.poscar_path, self.totdos_path)
        self._read_traps()
        self.R_SRH = self.__get_R_SRH(self.Vs)

    def calculate_rad(self):
        """
        calculate band-to-band radiative recombination rate
        """
        self.j_sc = self.__cal_J_sc()
        self.j0_rad = self.__cal_J0_rad()
        self.jv = self.__cal_jv(self.Vs)
        self.v_oc = self.__cal_v_oc()
        self.v_max, self.j_max, self.efficiency = self.__calc_eff()
        self.ff = self.__calc_ff()
        self.l_calc = True

    def calculate(self):
        self.calculate_SRH()
        self.calculate_rad()

    def __cal_J_sc(self):
        """
        Calculate and return J_sc, the short circuit current
        J_sc = q * (integrate(AM15flux * absorptivity dE) from 0 to E_gap) / EQE_EL
        """
        # unit-less times m^-2 eV^-1 s^-1, integrated over energy -> m^-2 s^-1:
        fluxcumm = cumulative_trapezoid(
            self.absorptivity[::-1] * AM15flux[::-1], self.Es[::-1], initial=0)
        # TODO: no E_gap (should be independent of E_gap; no absorption below E_gap)
        fluxaboveE = (fluxcumm[::-1] * -1  # invert spectrum
                      * self.intensity)  # intensity = 1 for 1 Sun; 100 mW/cm^2
        flux_absorbed = interp1d(self.Es, fluxaboveE)(self.E_gap)  # above-gap photon flux in m^-2 s^-1
        # Below; J_sc: m^-2 s^-1 * C -> (C/s)/m^-2 = A/m^2 = (1000 mA)/(100 cm)^2 = 0.1 mA/cm^-2;
        # so * 0.1 converts final units to -> mA/cm^2:
        J_sc = flux_absorbed * scpc.e * 0.1  # mA/cm^2  (0.1: from A/m2 to mA/cm2)
        return J_sc

    def __cal_J0_rad(self):
        '''
        Calculate and return J0, the dark saturation current
        J0 = q * (integrate(phi dE) from E to infinity)  / EQE_EL
        phi is the black body radiation at T (flux vs energy)
        '''
        phi = 2 * np.pi * (((self.Es*scpc.eV)**2) * scpc.eV / ((scpc.h**3) * (scpc.c**2)) / (
                           np.exp(self.Es*scpc.eV / (scpc.k*self.T)) - 1))
        fluxcumm = cumtrapz(
            self.absorptivity[::-1] * phi[::-1], self.Es[::-1], initial=0)
        # TODO: no E_gap (should be independent of E_gap; no absorption below E_gap)
        fluxaboveE = fluxcumm[::-1] * -1
        flux_absorbed = interp1d(self.Es, fluxaboveE)(self.E_gap)
        j0 = flux_absorbed * scpc.e * 0.1  # (0.1: from A/m2 to mA/cm2)
        return j0

    def __cal_jv(self, Vs):
        """
        Calculate and return J-V curve
        J = -J_sc + J0_rad * (exp(qVs/kT) - 1) + R_SRH

        Args:
        Vs: voltage array
        """
        j_sc, j0_rad = self.j_sc, self.j0_rad

        j = -1.0 * j_sc + j0_rad * (np.exp(scpc.e*Vs / (scpc.k*self.T)) - 1)
        # nonraditive recombination
        if self.R_SRH is not None:
            j += scpc.e * self.R_SRH / 1E-3 * self.thickness * 1E-7

        jv = pd.DataFrame({"V": Vs, "J": j})
        return jv

    def __cal_v_oc(self):
        """
        Calculate and return the open circuit voltage
        """
        def f(v): return interp1d(self.jv.V, self.jv.J)(v)
        sol = root_scalar(f, bracket=[0, self.jv.V.max()])
        return sol.root

    def __find_max_point(self):
        """ 
        Calculate and return the voltage that produces the maximum power
        """
        power = self.jv.J * self.jv.V
        def f(v): return interp1d(self.jv.V, power)(v)
        res = minimize_scalar(f, method='Bounded', bounds=[0, self.jv.V.max()])
        return res.x

    def __calc_eff(self):
        """
        Calculate and return the maximum power point, the current at the maximum power point, and the efficiency
        """
        v_max = self.__find_max_point()
        power = self.jv.J * self.jv.V

        def eff(v): return interp1d(self.jv.V, power)(v) \
            / sun_power * self.intensity

        def j(v): return interp1d(self.jv.V, self.jv.J)(v)
        return v_max, -j(v_max), -eff(v_max)

    def __calc_ff(self):
        """
        Calculate and return the fill factor
        """
        ff = self.v_max * self.j_max / self.v_oc / self.j_sc
        return ff

    # absorptivity functions:
    def __read_alpha(self):
        """
        read optical absorption coefficient data
        """
        alpha = pd.read_csv(tlc.ALPHA_FILE)
        # alpha.plot(x='E', y='alpha')
        self.alpha = alpha

    def _calc_absorptivity(self):
        """
        calculate absorptivity
        """
        self.__read_alpha()
        absorptivity = 1 - np.exp(
            -2 * self.alpha.alpha * self.thickness * 1e-7  # 1e-7 converts thickness in nm -> cm
        )  # then thickness in cm times absorption in cm^-1 -> unit-less
        self.absorptivity = np.interp(Es, self.alpha.E, absorptivity)  # unit-less

    # nonradiative recombination:
    def _get_scfermi(self, file_efrom):
        """
        read formation energies of defects, POSCAR, totdos
        """
        self.scfermi = Scfermi.from_file(file_efrom)

    def _run_scfermi(self, Tanneal, Tfrozen, poscar_path, totdos_path):
        """
        run scfermi 
        1. calculate equilibrium concentrations of defects at Tanneal
        2. calcualte charge states of defects and carrier concentrations at Tfrozen
        """
        run_scfermi_all(self.scfermi, Tanneal=Tanneal, Tfrozen=Tfrozen, poscar_path=poscar_path, totdos_path=totdos_path)

    def _read_traps(self):
        """
        read trap file
        """
        trap_list = []
        df_trap = pd.read_csv(tlc.TRAP_FILE, comment='#', sep=r'\s+', usecols=range(11))

        for index, data in df_trap.iterrows():
           D, E_t1, E_t2, g, C_p1, C_p2, C_n1, C_n2 = data.D, data.level1, data.level2, data.g, data.C_p1, data.C_p2, data.C_n1, data.C_n2
           q1, q2, q3 = data.q1, data.q2, data.q3
           N_t = 0
           trap_list.append(Trap(D, E_t1, E_t2, N_t, q1, q2, q3, g, C_p1, C_p2, C_n1, C_n2))

        self.trap_list = trap_list

    def __get_delta_n(self, V):
        """
        calculate excess carrier concentration

        Args:
        V: voltage (V)
        """
        scfermi = self.scfermi

        def calc_DOS_eff(carrier_concnt, e_f, temp):
            """
            calculate effective DOS
            Args:
            carrier_concnt: carrier concentration (cm^-3)
            e_f: Fermi level (eV)
            temp: temperature (K)
            """
            return carrier_concnt/np.exp(-e_f/(kb_in_eV_per_K*temp))

        n0 = scfermi.n
        p0 = scfermi.p
        e_gap = scfermi.e_gap
        temp = scfermi.T
        Vc = kb_in_eV_per_K*temp

        N_p = calc_DOS_eff(p0, scfermi.fermi_level, scfermi.T)
        N_n = calc_DOS_eff(n0, e_gap - scfermi.fermi_level, scfermi.T)

        print(f"N_p: {N_p}")
        print(f"N_n: {N_n}")

        scfermi.N_p = N_p
        scfermi.N_n = N_n

        delta_n = 1/2. * (-n0 - p0 + np.sqrt((n0 + p0)**2 -
                                             4 * n0 * p0 * (1 - np.exp(V/Vc))))

        return delta_n

    def __cal_R_SRH(self, V):
        """
        calculate defect-mediated nonradiative recombination rate
        """
        assert self.scfermi is not None

        delta_n = self.__get_delta_n(V)

        scfermi = self.scfermi
        n0 = scfermi.n
        p0 = scfermi.p
        N_n = scfermi.N_n
        N_p = scfermi.N_p

        for trap in self.trap_list:
            defect = next(
                defect for defect in scfermi.defects if defect.name == trap.D)
            trap.N_t = sum(
                cs.concnt
                for cs in defect.chg_states
                if cs.q in (trap.q1, trap.q2, trap.q3)
            )
        return np.sum(  # R_SRH
            [
                trap.rate(n0, p0, delta_n, N_n, N_p, scfermi.e_gap, scfermi.T)
                for trap in self.trap_list
            ]
        )

    def __get_R_SRH(self, Vs):
        """
        get defect-mediated nonradiative recombination rate
        """
        Rs = np.array([self.__cal_R_SRH(V) for V in Vs])
        return Rs

    # Plot helper
    def plot_tauc(self):
         """
            plot Tauc figure
        """
        tauc = (self.alpha.alpha*self.alpha.E)**2
        plt.figure(0)
        plt.plot(self.alpha.E, tauc)
        plt.plot([self.E_gap, self.E_gap], [-1E10, 1E10],
                 ls='--', label="Band gap")

        plt.xlabel("Energy (eV)", fontsize=16)
        plt.ylabel(
            "$\mathregular{(ahv)^2}$ ($\mathregular{eV^2cm^{-2}}$)", fontsize=16)
        plt.title("Tauc plot")
        plt.legend()
        plt.xlim((self.E_gap-0.5, self.E_gap+0.5))
        plt.ylim((0, 10E9))
        # plt.yscale("log")
        # plt.show()

    def plot_alpha(self, l_plot_solar=True):
        """
        plot absorption coefficient figure
        """
        self.alpha.plot(x='E', y='alpha', logy=True)
        plt.plot([self.E_gap, self.E_gap], [-1E10, 1E10],
                 ls='--', label="Band gap")
        plt.ylim((10E0, 10E6))
        plt.xlabel("Energy (eV)", fontsize=16)
        plt.ylabel("Absorption coefficient ($\mathregular{cm^{-1}}$)",
                   fontsize=16)
        if not self.l_sq: 
            plt.title("Absorption coefficient (taken from {})".format(tlc.ALPHA_FILE))
        else: 
            plt.title("Absorption coefficient (SQ limit)".format(tlc.ALPHA_FILE))
        plt.legend(loc=1)

        if l_plot_solar:
            # plt.xlim((0, self.E_gap))
            plt.twinx()
            plt.plot(Es, AM15*1E-3, label="AM1.5G", c='gray')
            plt.ylabel("Spectral irradiation  ($\mathregular{kW m^{-2} eV^{-1}}$)",
                    fontsize=16)
            plt.legend(loc=4)

        # plt.show()

    def plot_jv(self):
        """
        plot J-V curve
        """
        self.jv.mask(self.jv.J > 100).plot(x="V", y="J")
        plt.ylim((self.j_sc*-1.2, 0))
        plt.xlim((0, self.E_gap))
        plt.xlabel("Voltage (V)", fontsize=16)
        plt.ylabel("Current density (mA/$\mathregular{cm^2}$)",
                   fontsize=16)
        plt.title("Theoretical J-V for Eg = {:.3f} eV".format(self.E_gap))
        #  plt.show()


if __name__ == "__main__":
    tlc_CZTS = tlc(1.5, T=300)
    tlc_CZTS.calculate()
    print(tlc_CZTS)
    tlc_CZTS.plot_tauc()
