#!/usr/bin/env python3 
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import rc
from run_scfermi import Chg_state, Defect, Scfermi, get_all, run_scfermi_all
import pickle
import pandas as pd
from scipy.optimize import fsolve
np.set_printoptions(precision=3)
import SQlimit 
# rc('text', usetex=True)
import seaborn as sns
sns.set_style("ticks")

sq = SQlimit.SQlim

CONCT_LIMS = (1E10, 1E21)
FERMI_LIMS = (0.1, 0.22)
NUM_COLORS = 20
E_BUFF = 0.2
cm = plt.get_cmap('tab20')

e = 1.60217662E-19 # elementary charge
kb = 8.6173303e-5


class Trap():
    def __init__(self, D, E_t, N_t, q1, q2, C_p, C_n):
        self.D = D
        self.E_t = E_t
        self.N_t = N_t
        self.q1 = q1
        self.q2 = q2
        # capture coeff (avoiding div by 0)
        self.C_p = C_p if C_p > 0 else 1E-100
        self.C_n = C_n if C_n > 0 else 1E-100
        self.name = "${{{}}} ({}/{})$".format(D, q1, q2)

    def rate(self, n0, p0, delta_n, N_n, N_p, e_gap, temp):
        n1 = N_n*np.exp(-(e_gap-self.E_t)/kb/temp)
        p1 = N_p*np.exp(-self.E_t/kb/temp)
        n = n0 + delta_n
        p = p0 + delta_n

        R = (n*p - n0*p0)/((p+p1)/(self.N_t*self.C_n) + (n+n1)/(self.N_t*self.C_p))

        return R 

    def __repr__(self):
        repr = "{}    ({}/{})  {:.2E}  {}  {:.2E}  {:.2E}".format(self.D, self.q1, self.q2, self.N_t, self.E_t, self.C_n, self.C_p) 
        return repr 
    def __str__(self):
        repr = "{}    ({}/{})  {:.2E}  {}  {:.2E}  {:.2E}".format(self.D, self.q1, self.q2, self.N_t, self.E_t, self.C_n, self.C_p) 
        return repr 

def read_traps(file='trap.dat'):
    trap_list = []
    df_trap = pd.read_csv(file, comment='#', sep=r'\s+', usecols=range(6))

    for index, data in df_trap.iterrows():
        D, E_t, C_p, C_n = data.D, data.level, data.C_p, data.C_n
        q1, q2 = data.q1, data.q2
        N_t = 0

        trap_list.append(Trap(D, E_t, N_t, q1, q2, C_p, C_n))

    return trap_list


def plot_level(ax, trap_list, x_lim):
    ratio_bar = 0.4
    clr_accpt = "#045a8d"
    clr_donor = "#bd0026"

    df_trap = pd.read_csv('trap.dat', comment='#', sep=r'\s+', usecols=range(6))
    Ds = df_trap.D.unique()
    
    for D_i, D in enumerate(Ds):
        delta_x = x_lim / float(len(Ds))
        x_st = D_i * delta_x + delta_x * (1 - ratio_bar) / 2.
        x_end = x_st + delta_x * (1/2. + ratio_bar/2)

        for i, data in df_trap[df_trap.D==D].iterrows():
            name = "${{{}}} ({}/{})$".format(data.D, data.q1, data.q2)
            clr = clr_donor if data.q1+data.q2 > 0 else clr_accpt
            ax.plot([x_st, x_end], [data.level, data.level], c=clr, lw=3, label=name)


def plot_concentration(ax, scfermi_list):
    n0 = np.array([scfermi.n  for scfermi in scfermi_list])
    p0 = np.array([scfermi.p  for scfermi in scfermi_list])

    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    # carrier concentrations
    if n0.max() > CONCT_LIMS[0]:
        ax.semilogy(n0, '-', lw=2, c='gray', label='n0')
    if p0.max() > CONCT_LIMS[0]:
        ax.semilogy(p0, '-', lw=2, c='k', label='p0')
    ax.set_ylim(CONCT_LIMS)
    ax.set_xlim((0, len(n0)-1))


    # Trap(E_t, N_t, C_p, C_n)
    n_state = np.sum([defect.n_charge for defect in scfermi_list[0].defects])
    concnts = np.zeros((n_state, len(scfermi_list)))
    names = []
    for chmpot_i, scfermi in enumerate(scfermi_list):
        state_i = 0 
        for defect in scfermi.defects:
            for chg_state in defect.chg_states:
                names.append("${{{}}}^{{{}}}$".format(defect.name, chg_state.q))
                concnts[state_i, chmpot_i] = chg_state.concnt
                state_i += 1

    for state_i, concnt in enumerate(concnts):
        if concnt.max() > CONCT_LIMS[0]:
            ax.plot(concnt, '--', label=names[state_i]) 


def plot_concentration_total(ax, scfermi_list):
    phase = ["${{{}}}$".format(scfermi.phase) for scfermi in scfermi_list]
    # phase = ["{}".format(scfermi.phase) for scfermi in scfermi_list]
    n0 = np.array([scfermi.n  for scfermi in scfermi_list])
    p0 = np.array([scfermi.p  for scfermi in scfermi_list])

    ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    # carrier concentrations
    if n0.max() > CONCT_LIMS[0]:
        ax.semilogy(n0, '--', lw=2, c='gray', label='n0')
    if p0.max() > CONCT_LIMS[0]:
        ax.semilogy(p0, '--', lw=2, c='k', label='p0')
    ax.set_ylim(CONCT_LIMS)
    ax.set_xlim((0, len(n0)-1))
    ax.plot((0, len(n0)-1), (1E15, 1E20))


    n_state = np.sum([defect.n_charge for defect in scfermi_list[0].defects])
    concnts = np.zeros((scfermi_list[0].n_defect, len(scfermi_list)))
    names = []
    for chmpot_i, scfermi in enumerate(scfermi_list):
        for defect_i, defect in enumerate(scfermi.defects):
            names.append(defect.name)
            concnts[defect_i, chmpot_i] = defect.concnt

    for defect_i, concnt in enumerate(concnts):
        if concnt.max() > CONCT_LIMS[0]:
            ax.semilogy(concnt, '-', label="${{{}}}$".format(names[defect_i])) 

    # ax.set_xticks(np.arange(len(phase)), phase)
    # ax.set_xticks(np.arange(len(phase)))
    # ax.set_xticklabels(phase, {'fontsize': 5, 'horizontalalignment': 'left'}, rotation=-30)
    ax.get_xaxis().set_visible(False)


def plot_e_fermi(ax, scfermi_list):
    e_gap = scfermi_list[0].e_gap
    temp = np.array([scfermi.T for scfermi in scfermi_list])
    fermi_level = np.array([scfermi.fermi_level  for scfermi in scfermi_list])
    n0 = np.array([scfermi.n  for scfermi in scfermi_list])
    p0 = np.array([scfermi.p  for scfermi in scfermi_list])

    inf = 100
    ax.plot(fermi_level, color='#bdbdbd', lw=4, label="$E_{F}$")
    ax.fill_between([-inf, inf], [0, 0], [-inf, -inf], where=[0, 0] >= [-inf, -inf], facecolor='#2b8cbe', interpolate=True)
    ax.fill_between([-inf, inf], [e_gap, e_gap], [inf, inf], where=[e_gap, e_gap] <= [inf, inf], facecolor='#f03b20', interpolate=True)

    ax.set_ylim((0-E_BUFF, 1.5+E_BUFF))
    phase = ["${{{}}}$".format(scfermi.phase) for scfermi in scfermi_list]
    #ax.set_xlim((0, len(fermi_level)-1))
    ax.set_xlim((0, len(fermi_level)-1))
    ax.set_xticks(np.arange(len(phase)))
    ax.set_xticklabels(phase, {'fontsize': 5, 'horizontalalignment': 'left'}, rotation=-30)


def get_delta_n(scfermi, V):
    def calc_DOS_eff(carrier_concnt, e_f, temp):
        return carrier_concnt/np.exp(-e_f/(kb*temp))

    n0 = scfermi.n
    p0 = scfermi.p
    e_gap = scfermi.e_gap
    temp = scfermi.T
    Vc = kb*temp

    N_p = calc_DOS_eff(p0, scfermi.fermi_level, scfermi.T)
    N_n = calc_DOS_eff(n0, e_gap - scfermi.fermi_level, scfermi.T)
    
    scfermi.N_p = N_p
    scfermi.N_n = N_n
  
    delta_n = 1/2. * (-n0 - p0 + np.sqrt((n0 + p0)**2 - 4 * n0 * p0 * (1 - np.exp(V/Vc))))
    return delta_n


def solve_jv(scfermi, trap_list, V, thickness):
    def _get_R_SRH(scfermi, trap_list, delta_n):
        n0 = scfermi.n
        p0 = scfermi.p
        N_n = scfermi.N_n
        N_p = scfermi.N_p 
        for trap in trap_list:
            defect = next(defect for defect in scfermi.defects if defect.name == trap.D)
            trap.N_t = 0
            for cs in defect.chg_states:
                if cs.q in (trap.q1, trap.q2):
                    trap.N_t += cs.concnt
        R = np.sum([trap.rate(n0, p0, delta_n, N_n, N_p, scfermi.e_gap, scfermi.T) 
                                        for trap in trap_list]) 
        return R

    temp = scfermi.T
    Vc = kb*temp

    param = sq(temp).get_paras(scfermi.e_gap, False)
    voc = param["Voc"]
    jsc = param["Jsc"]
    j0 = param["J0"]
    
    delta_n = get_delta_n(scfermi, V)

    R_SRH = _get_R_SRH(scfermi, trap_list, delta_n)

    j = jsc + j0 * (1 - np.exp(V/Vc)) - e * R_SRH / 1E-3 * thickness * 1E-7  # length -> cm 
    return j


def get_V_OC(scfermi, trap_list, thickness):
    j = lambda V: solve_jv(scfermi, trap_list, V, thickness)
    voc = fsolve(j, 1, xtol=1E-4)[0]
    return voc


def get_e_life_time(scfermi, trap_list):
    tou_inv = 0
    for trap in trap_list:
        defect = next(defect for defect in scfermi.defects if defect.name == trap.D)
        trap.N_t = 0
        for cs in defect.chg_states:
            if cs.q in (trap.q1, trap.q2):
                trap.N_t += cs.concnt
        tou_inv += trap.N_t * trap.C_n
    return 1./tou_inv


def plot_quasi_e_fermi(ax, trap_list, scfermi_list, thickness):
    def _get_quasi_e_fermi(scfermi, V):
        temp = scfermi.T
        e_gap = scfermi.e_gap
        N_p = scfermi.N_p
        N_n = scfermi.N_n
        delta_n = get_delta_n(scfermi, V)
        p = scfermi.p + delta_n
        n = scfermi.n + delta_n
        
        e_f_p = -kb*temp*np.log(p/N_p)
        e_f_n = e_gap+kb*temp*np.log(n/N_n)
       
        return e_f_p, e_f_n    

    n0 = np.array([scfermi.n  for scfermi in scfermi_list])
    p0 = np.array([scfermi.p  for scfermi in scfermi_list])
    temp = np.array([scfermi.T for scfermi in scfermi_list])
    
    e_f_p_list = [] 
    e_f_n_list = [] 
    for scfermi in scfermi_list:
        voc = get_V_OC(scfermi, trap_list, thickness)
        e_f_p, e_f_n = _get_quasi_e_fermi(scfermi, voc)
        e_f_p_list.append(e_f_p)
        e_f_n_list.append(e_f_n)

    ax.plot(e_f_p_list, '-', label='$E_{F,p}$')
    ax.plot(e_f_n_list, '-', label='$E_{F,n}$')

    ax.get_xaxis().set_visible(False)


def plot_iv_simul(ax, trap_list, scfermi, thickness):
    sun_power = 100
    n0 = scfermi.n
    p0 = scfermi.p
    temp = scfermi.T
    e_gap = scfermi.e_gap
    
    vout_list = []
    
    param = sq(temp).get_paras(e_gap, False)
    voc = param["Voc"]
    jsc = param["Jsc"]
    j0 = param["J0"]
    vc = kb*temp

    j_sq = lambda v: jsc + j0 * (1.-np.exp(v/vc))

    print("===================")
    print("SQlimit E_gap: {:.2f} eV at T={:.1f} K with W={:.2f} nm".format(e_gap, temp, thickness))
    print("===================")
    print("VOC: {:.3f} V, JSC: {:.3f} mA/cm-2, J0: {:.3e} mA/cm-2".format(voc, jsc, j0))
    print("PCE: {:.3f} \%".format(param["PCE"]))
    print("FF: {:.3f} \%".format(param["FF"]))
    # print("sun power", voc*jsc*param["FF"]/param["PCE"])
    print("===================")

    voc = get_V_OC(scfermi, trap_list, thickness)
    
    vout_list = np.linspace(0-0.1, e_gap, 100)
    jout_list = []
    for v in vout_list:
        j = solve_jv(scfermi, trap_list, v, thickness)
        jout_list.append(j)

    # jout_list = np.arange(-0.2, jsc+0.1, 0.1)
    # for jout in jout_list:
    #     N_p, N_n, vout, dn = solve_recomb_rate(trap_list, [scfermi], jout=jout, W=thickness)
    #     vout_list.append(vout[0])

    power = jout_list*vout_list
    max_i = np.argmax(power)
    
    print("SRH limited")
    print("-------------------------------"
        "\nJSC: {:.2f} (mA/cm3), ".format(jsc),
          "\nVOC: {:.2f} (V)".format(voc),
          "\n-------------------------------")
    print("-------------------------------"
        "\nJ_max: {:.2f}, ".format(jout_list[max_i]),
          "\nV_max: {:.2f} (V),".format(vout_list[max_i]),
          "\nFF: {:.2f}%".format(100*power[max_i]/jsc/voc),
          "\neff: {:.2f}%".format(100*power[max_i]/sun_power),
          "\n-------------------------------")

    ax.plot(vout_list, jout_list, label="SRH limit")
    vs = np.linspace(0, e_gap, 100)
    ax.plot(vs, j_sq(vs), label="SQ limited")
    

    ax.set_xticks([0, 0.5, 1, 1.5])
    ax.set_yticks(np.arange(0, 50, 10))


    ax.set_ylim((0, 50))
    ax.set_xlim((0, 1.2))


def main(path_i, path_f, thickness, dopants_anneal=[], Taneal=853, Tfrozen=300):
    """ Tool tip missing
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    plt.subplots_adjust(right=0.5)

    scfermi_list = get_all(path_i, path_f, n_points=len(dopants_anneal))
    for i, scfermi in enumerate(scfermi_list):
        print('{}/{}'.format(i, len(scfermi_list)))
        run_scfermi_all(scfermi, Tanneal=Taneal, Tfrozen=Tfrozen, dopants_anneal=dopants_anneal[i])

    # defects
    plot_concentration_total(ax[0], scfermi_list)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=2)

    # # # fermi level
    plot_e_fermi(ax[1], scfermi_list)

    # # # traps
    trap_list = read_traps(file='trap.dat')
    plot_level(ax[1], trap_list, len(scfermi_list))
    plot_quasi_e_fermi(ax[1], trap_list, scfermi_list, thickness)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2)
    
    # # # j-V curve
    scfermi = scfermi_list[-1]
    plot_iv_simul(ax[2], trap_list, scfermi, thickness)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=2)

    tou = get_e_life_time(scfermi, trap_list)
    print("N_p, N_n", scfermi.N_p, scfermi.N_n)  
    print('life_time', tou)
    # for trap in trap_list:
    #     print(str(trap)+"\n")
    print('p0: {:.3E}'.format(scfermi.p))
    print('n0: {:.3E}'.format(scfermi.n))

    
    scfermi = scfermi_list[0]
    plot_iv_simul(ax[2], trap_list, scfermi, thickness)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, ncol=2)
    # for trap in trap_list:
    #     print(str(trap)+"\n")

    tou = get_e_life_time(scfermi, trap_list)
    print("N_p, N_n", scfermi.N_p, scfermi.N_n)  
    print('life_time', tou)
    # for trap in trap_list:
    #     print(str(trap)+"\n")
    print('p0: {:.3E}'.format(scfermi.p))
    print('n0: {:.3E}'.format(scfermi.n))


    # plt.savefig('fig.pdf')



if __name__ == '__main__':
    thickness = 2000 # nanometer
    # concnt = 0
    dopants_list = []
    for concnt in np.arange(0, 20.01, 1):
        print(concnt)
        H_i = Defect("H_i", 1, 1, l_frozen=True)
        cs = Chg_state(1, 1, 1); cs.set_concnt(10**concnt) 
        H_i.chg_states = [cs]; H_i.calc_tot_concnt()
        dopants_list.append([H_i])
    main("inputs/input-fermi_Se8,Zn1Se1,Sn1Se2.dat", "inputs/input-fermi_Se8,Zn1Se1,Sn1Se2.dat", thickness, dopants_list)
	# main("inputs/input-fermi_Cu4,Cu96Se48,Cu8Sn4Se12.dat", "inputs/input-fermi_Se8,Zn1Se1,Sn1Se2.dat", thickness, dopants_list)

    # main("inputs/input-fermi_Cu4,Cu96Se48,Cu8Sn4Se12.dat", "inputs/input-fermi_Se8,Zn1Se1,Sn1Se2.dat", thickness, [H_i])
    # main("inputs/input-fermi_Sn4Se4,Sn1Se2,Cu8Sn4Se12.dat", "inputs/input-fermi_Se8,Zn1Se1,Sn1Se2.dat", thickness, [H_i])
