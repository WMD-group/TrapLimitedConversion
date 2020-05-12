#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import deque
# import subprocess
import re
import pandas as pd
import copy
import os
import numpy as np
import pickle
from pymatgen.core.structure import Structure
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
kB = 8.6173303E-5   # eV K-1


class Chg_state:
    def __init__(self, q, energy, g):
        self.q = q              # charge state
        self.energy = energy    # formation energy when E_F = 0
        self.g = g              # degeneracy
        self.concnt = 0         # concentration

    def set_concnt(self, concnt):
        self.concnt = concnt

    def calc_occ(self, E_F, T):
        e_form = self.energy + E_F * self.q    # Formation energy
        # partition function is 0; dilute limit
        return self.g*np.exp(-e_form/kB/T)


class Defect:
    def __init__(self, name, n_charge, n_site, l_frozen=False):
        self.name = name             # name
        self.n_site = n_site         # number of sites in the unit cell
        self.n_charge = n_charge     # number of charge states
        self.chg_states = []         # list of Chg_state objects
        self.concnt = 0              # total concentrations; sum chg_states.concnt
        self.l_frozen = l_frozen     # is frozen? (fixed total concentration)

    def add_chg_state(self, q, energy, g):
        self.chg_states.append(Chg_state(q, energy, g))

    def calc_tot_concnt(self):
        self.concnt = np.sum([cs.concnt for cs in self.chg_states])

    def calc_concnt(self, vol, E_F, T):
        cs_concnts = np.zeros(len(self.chg_states))
        for cs_i, cs in enumerate(self.chg_states):
            occ = cs.calc_occ(E_F, T)
            if occ > 1:
                occ = 1.
            concnt = self.n_site / vol * occ
            cs_concnts[cs_i] = concnt

        for cs_i, cs in enumerate(self.chg_states):
            if self.l_frozen:
                if np.sum(cs_concnts) < 1E-100:
                    cs_concnts += 1E-100
                cs_concnts *= self.concnt / np.sum(cs_concnts)

            cs.set_concnt(cs_concnts[cs_i])


class Scfermi:
    def __init__(self, n_spin, n_elect, e_gap, T, n_defect, defects, n_frozen=0, phase="", verbose=False):
        self.n_spin = n_spin        # spin polarized (2) or not (1)
        self.n_elect = n_elect      # number of electyrons in the unit cell
        self.e_gap = e_gap          # band gap
        self.T = T                  # temperature
        self.n_defect = n_defect    # number of defects
        self.defects = defects      # list of Defect objects
        self.fermi_level = 0        # Fermi level
        self.n_frozen = None        # if frozen?
        self.n = 0
        self.p = 0      # carrier concentrations
        self.excess_charge = 0      # excess charge carriers
        self.phase = phase            # name of scfermi ()
        self.verbose = verbose

    @classmethod
    def from_file(self, path="input-fermi.dat", mode=None, phase=""):
        with open(path, "r", encoding='utf8') as f:
            lines = []
            for line in f:
                if line[0] == '#':
                    continue
                lines.append(line)
            lines = deque(lines)
            # n_spin, n_elect, e_gap, T, n_defect = lines[:5]
            n_spin = int(lines.popleft().split()[0])
            n_elect = int(lines.popleft().split()[0])
            e_gap = float(lines.popleft().split()[0])
            T = float(lines.popleft().split()[0])
            n_defect = int(lines.popleft().split()[0])

            defects = []
            for i in range(n_defect):
                # metadata
                name, n_charge, n_site = lines.popleft().split()
                defect = Defect(name, int(n_charge), int(n_site))
                # chg_states = []
                for i in range(int(n_charge)):
                    q, energy, g = lines.popleft().split()
                    defect.add_chg_state(int(q), float(energy), int(g))
                defects.append(defect)

            # frozen
            if mode == 'frozen':
                n_frozen = int(lines.popleft())
                for _ in range(n_frozen):
                    name, concnt = (T(x) for T, x in zip(
                        [str, float], lines.popleft().split()))
                    defect = next(
                        defect for defect in defects if defect.name == name)
                    defect.l_frozen = True
                    defect.concnt = concnt

            scfermi = Scfermi(n_spin, n_elect, e_gap, T,
                              n_defect, defects, phase=phase)

            return scfermi

    def write_input_file(self, path):
        def write_defect(f, d):
            # metadata
            f.write("{:<10s} {:>2d} {:>2d} \n".format(
                d.name, d.n_charge, d.n_site))
            # formation energy
            for cs in d.chg_states:
                f.write("{:>2d} {:>10f} {:>2d} \n".
                        format(cs.q, cs.energy, cs.g))

        with open(path, "w", encoding='utf8') as f:
            f.write("{:<10d} # n_spin \n".format(self.n_spin))
            f.write("{:<10d} # n_elect \n".format(self.n_elect))
            f.write("{:<10g} # e_gap \n".format(self.e_gap))
            f.write("{:<10g} # T \n".format(self.T))
            f.write("{:<10d} # n_defect \n#\n".format(self.n_defect))

            for d in self.defects:
                write_defect(f, d)

    def write_frozen_input_file(self, T, path):
        self.T = T
        self.write_input_file(path)
        with open(path, "a", encoding='utf8') as f:
            f.write("#\n{} # no. of frozen defects\n".format(self.n_defect))
            f.write("#Frozen defects: (name)     (concentration / cm^-3)\n")
            for defect in self.defects:
                concnt = [cs.concnt for cs in defect.chg_states]
                concnt = sum(concnt)
                f.write("{:<16s} {:<10s} {:<20.20E}\n".format(
                    '', defect.name, concnt))
            f.write("0\n")

    def read_output(self, path):
        with open(path, 'r', encoding='utf8') as f:
            # fermi level
            for line in f:
                match = re.search(r"SC Fermi level :.+", line)
                if match is not None:
                    self.fermi_level = float(match.group().split()[-2])
                    break
            """
            n (electrons)  :   0.1379710524353E+000  cm^-3
            p (holes)      :   0.3486583475164E+019  cm^-3
            """
            for line in f:
                match = re.search(r"n \(electrons\)  :.*", line)
                if match is not None:
                    n = float(match.group().split()[-2])
                    self.n = n
                    break
            for line in f:
                match = re.search(r"p \(holes\)      :.*", line)
                if match is not None:
                    p = float(match.group().split()[-2])
                    self.p = p
                    break

            # read defects
            for line in f:
                if line == "Breakdown of concentrations for each defect charge state:\n":
                    break

            for defect in self.defects:
                f.readline()  # --------...
                name = f.readline().split()[0]

                assert name == defect.name, "name {} {}".format(
                    name, defect.name)
                for i in range(defect.n_charge):
                    concnt = float(f.readline().split()[2])
                    defect.chg_states[i].set_concnt(concnt)
                defect.calc_tot_concnt()

    def _write_output(self, path):
        with open(path, 'w', encoding='utf8') as f:
            # fermi level
            f.write(r"SC Fermi level : ", self.fermi_level, "(eV)\n")

            # carrier concentration
            """
            n (electrons)  :   0.1379710524353E+000  cm^-3
            p (holes)      :   0.3486583475164E+019  cm^-3
            """
            f.write(r"n (electrons)  :   ", self.n, "(cm^-3)\n")
            f.write(r"p (holes)      :   ", self.p, "(cm^-3)\n")

            # defect concentration
            f.write("Breakdown of concentrations for each defect charge state:\n")
            for defect in self.defects:
                f.write("--------------------------------------------------------")
                f.write(
                    "{:s15}:   Charge   Concentration(cm^-3) % total".format(defect.name))
                for cs in defect.chg_states:
                    f.write("               :{:4i}     {0:.16E}    0.00".format(
                        cs.q, cs.concnt))

    # def run(self, Tanneal=None, mode=None):
    #     if mode is None:
    #         result = subprocess.run(['sc-fermi > output.log'],
    #                                 shell=True, stdout=subprocess.PIPE)
    #     if mode == 'frozen':
    #         result = subprocess.run(['frozen-sc-fermi > output-frozen.log'],
    #                                 shell=True, stdout=subprocess.PIPE)

    def _run(self, T=None, mode=None, dopants=[]):
        # built-in sc-fermi run
        def _get_carrier_concnt(energy, dos, fermi_level, T):
            vb_idx = np.where(energy < self.e_gap / 2.)
            vb_idx = np.where(energy <= 0)
            cb_idx = np.where(energy > self.e_gap / 2.)
            cb_idx = np.where(energy >= self.e_gap)

            del_E = (energy[1] - energy[0])
            del_E = (energy[-1] - energy[0])/len(energy)

            # normalize
            tot_elect = np.sum(dos[vb_idx]) * del_E
            dos_ = dos * self.n_elect/tot_elect
            dos_ = dos

            # Fermi-Dirac distribution
            kbT = T * kB
            fd_occ = 1. / (np.exp((energy-fermi_level)/kbT)+1)
            p = np.sum((dos_*del_E)[vb_idx]) - \
                np.sum((dos_*fd_occ*del_E)[vb_idx])
            p = np.sum((dos_*del_E*(1-fd_occ))[vb_idx])
            n = np.sum((dos_*fd_occ*del_E)[cb_idx])
            return n, p

        def _get_excess_charge(E_F, energy, dos, vol, T):
            """ calculate (p + D^+) - (n + A^-) """
            # calc carrier concentration
            n, p = _get_carrier_concnt(energy, dos, E_F, T)

            # calc carged defect concentration
            for defect in self.defects:
                defect.calc_concnt(vol, E_F, self.T)

            # calc total defect and carrier charge
            q = 0  # excess + charge
            for defect in self.defects:
                for cs in defect.chg_states:
                    q += cs.q * cs.concnt
            return q - n/vol + p/vol

        def _freez_all():
            for defect in self.defects:
                defect.l_frozen = True

        if mode == 'frozen':
            _freez_all()

        if T is not None:
            self.T = T

        # add dopants
        self.defects += dopants
        self.n_defect = len(self.defects)

        # read structure for volume
        structure = Structure.from_file("POSCAR")
        vol = structure.volume * 1E-24  # AA-3 to cm-3

        # read dos
        dos = np.loadtxt("totdos.dat", skiprows=1)
        energy = dos[:, 0]
        dos = dos[:, 1]

        sc_efermi = fsolve(_get_excess_charge, x0=self.e_gap /
                           4., args=(energy, dos, vol, self.T))[0]

        self.fermi_level = sc_efermi
        for defect in self.defects:
            defect.calc_concnt(vol, sc_efermi, self.T)
            defect.calc_tot_concnt()

        self.excess_charge = _get_excess_charge(
            sc_efermi, energy, dos, vol, self.T)
        self.n, self.p = _get_carrier_concnt(energy, dos, sc_efermi, self.T)
        self.n /= vol
        self.p /= vol

        assert self.excess_charge < 1E10, print(
            'excess_charge: ', "{:.8E}".format(self.excess_charge))


def run_scfermi_all(scfermi, Tanneal=853, Tfrozen=300, builtin_run=True, dopants_anneal=[]):
    """ 
    run sc-fermi and frozen-sc-fermi subsequently
    """
    # if builtin_run:
    scfermi._run(T=Tanneal, dopants=dopants_anneal)
    scfermi.defects = scfermi.defects[:len(
        scfermi.defects)-len(dopants_anneal)]
    scfermi.n_defect = len(scfermi.defects)
    # else:
    #     scfermi.write_input_file("input-fermi.dat")
    #     scfermi.run()
    #     scfermi.read_output(path="output.log")
    if scfermi.verbose: print("fermi_level (anneal):", scfermi.fermi_level)

    # if builtin_run:
    scfermi._run(T=Tfrozen, mode='frozen')
    # else:
        # scfermi.write_frozen_input_file(
        #     T=Tfrozen, path="input-fermi-frozen.dat")
        # scfermi.run(mode='frozen')
        # scfermi.read_output(path="output-frozen.log")
    if scfermi.verbose: print("fermi_level (frozen):", scfermi.fermi_level)

    if scfermi.verbose: print('run completed at T = {} K'.format(scfermi.T))


def get_all(path_i, path_f, n_points=20):
    """ 
    read two sc-fermi input files and make and return {n_points} sc-fermi list
    """
    def _interpolate_e_form(scfermi_i, scfermi_f, ratio=1):
        # scfermi = copy.deepcopy(scfermi_i)
        # (self, n_spin, n_elect, e_gap, T, n_defect, defects, n_frozen=0):

        defects = []
        assert scfermi_i.n_defect == scfermi_f.n_defect

        for d_i, temp_d in enumerate(scfermi_f.defects):
            assert scfermi_i.defects[d_i].n_charge == scfermi_f.defects[d_i].n_charge
            assert scfermi_i.defects[d_i].name == scfermi_f.defects[d_i].name
            assert scfermi_i.defects[d_i].n_site == scfermi_f.defects[d_i].n_site, print(
                scfermi_i.defects[d_i].name)

            defect = Defect(temp_d.name, temp_d.n_charge,
                            temp_d.n_site, temp_d.l_frozen)

            for cs_i, chg_state in enumerate(temp_d.chg_states):
                assert scfermi_i.defects[d_i].chg_states[cs_i].q == scfermi_f.defects[d_i].chg_states[cs_i].q

                e_i = scfermi_i.defects[d_i].chg_states[cs_i].energy
                e_f = scfermi_f.defects[d_i].chg_states[cs_i].energy
                energy = ((1-ratio)*e_i + ratio*e_f)

                defect.add_chg_state(chg_state.q, energy, chg_state.g)
            defects.append(defect)

        assert scfermi_f.n_defect == len(defects), print(
            scfermi_f.n_defect, len(defects))

        scfermi = Scfermi(scfermi_i.n_spin, scfermi_i.n_elect,
                          scfermi_i.e_gap, 0, scfermi_i.n_defect, defects)
        scfermi.T = ((1 - ratio) * scfermi_i.T + ratio * scfermi_f.T)

        return scfermi

    scfermi_list = []

    scfermi_i = Scfermi.from_file(path_i)
    scfermi_f = Scfermi.from_file(path_f)

    for ratio in np.linspace(0, 1, n_points):
        scfermi_list.append(_interpolate_e_form(scfermi_i, scfermi_f, ratio))
    return scfermi_list


def write_data(scfermi_list, file='scfermi.pkl'):
    # T, n, p, D^q1, D^q2
    n_defect_q = [len(defect.chg_states) for defect in scfermi_list[0].defects]
    n_defect_q = sum(n_defect_q)
    n_extra = 4
    output = np.zeros((len(scfermi_list), n_extra+n_defect_q))
    for i, scfermi in enumerate(scfermi_list):
        T = scfermi.T
        fermi_level = scfermi.fermi_level
        n = scfermi.n
        p = scfermi.p

        concnt = [chg_state.concnt for defect in scfermi.defects
                  for chg_state in defect.chg_states]
        output[i] = np.hstack([[T, fermi_level, n, p], concnt])

    # header = 'T n p '
    header = 'T  Fermi  n  p  ' + '  '.join(
        ['{}^{}'.format(defect.name, chg_state.q)
         for defect in scfermi.defects for chg_state in defect.chg_states])

    np.savetxt('output_scfermi.txt', output, delimiter='  ',
               fmt=['%-5.2f'] + ['%E'] * (n_extra-1+n_defect_q),
               header=header)

    df = pd.DataFrame(data=output, columns=header.split())

    df["phase"] = [scfermi.phase for scfermi in scfermi_list]
    df.set_index("phase")
    print(df)
    df.to_csv('output_scfermi.csv')

    save(scfermi_list, file)


def save(scfermi_list, file='scfermi.pkl'):
    """
    save the list of scfermi objects
    in pickle file
    """
    with open(file, 'wb') as output:
        pickle.dump(scfermi_list, output, pickle.HIGHEST_PROTOCOL)


def main_interpolate(path_i, path_f, file, Tfrozen=300, Tanneal=853, n_points=20):

    scfermi_list = get_all(path_i, path_f, n_points=n_points)

    for i, scfermi in enumerate(scfermi_list):
        print('{}/{}'.format(i, len(scfermi_list)))
        run_scfermi_all(scfermi, Tanneal=853, Tfrozen=Tfrozen)
    write_data(scfermi_list, file)
    print("done")


def main(paths, file, Tfrozen=300, Tanneal=853, n_points=20, dopants_anneal=[]):
    def _read_scfermi_input_all(path="./"):
        scfermi_list = []
        # find input_file
        files = []
        for r, d, f in os.walk(path):
            for file in f:
                if '.dat' in file and 'input-fermi' in file:
                    files.append(os.path.join(r, file))

        for f in files:
            phase = f.split('/')[-1][12:-4]
            scfermi_list.append(Scfermi.from_file(f, phase=phase))
        return scfermi_list

    scfermi_list = _read_scfermi_input_all(paths)

    for i, scfermi in enumerate(scfermi_list):
        print('{}/{}'.format(i, len(scfermi_list)))
        run_scfermi_all(scfermi, Tanneal=Tanneal,
                        Tfrozen=Tfrozen, dopants_anneal=dopants_anneal)

    write_data(scfermi_list, file)
    print("done")


if __name__ == '__main__':
    concnt = 1E19
    Na_i = Defect("Na_i", 1, 1, l_frozen=True)
    cs = Chg_state(1, 1, 1)
    cs.set_concnt(concnt)
    Na_i.chg_states = [cs]
    Na_i.calc_tot_concnt()
    main("./inputs", "scfermi0.pkl", Tfrozen=300,
         Tanneal=853, dopants_anneal=[Na_i])
