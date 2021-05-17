## Steps to calculate optical conductivity using VASP and Wannier90

#### 1. Standard VASP run  
- Structural relaxation and SCF calculation to obtain the wavefunction `WAVECAR` file.

#### 2. LWANNIER90 run  
- Add `LWANNIER90=.TRUE.` (which switches on the interface between VASP and WANNIER90) to INCAR.   
- Create a `wannier90.win` file in the same calculation directory. Analyse band composition and set `num_wann`, `num_bands`, `exclude_bands`, disentanglement window and projections block in the `wannier90.win` file.  
- Read `WAVECAR` (`ISTART = 1`) and run VASP to generate `wannier90.mmn`, `wannier90.eig` and `wannier90.amn` files. The `unit_cell_cart`, `atom_cart`, `mp_grid` and `kpoints` blocks in the `wannier90.win` file will be automatically filled by VASP2wannier90 interface.

#### 3. Wannier90 calculation  
- Add the bandstructure plot flags to the `wannier90.win` file. Run wannier90 (wannier90.x wannier90.win) to generate the bandstructure files `wannier90_band.dat`, `wannier90_band.gnu`, `wannier90_band.agr` and other output files `wannier90.wout` and `wannier90.chk`. Compare the bandstructure obtained by WANNIER90 and VASP to make sure the Wannier interpolations are reasonable.
- (Optionally) Adjust the disentanglement window and rerun wannier90. 

#### 4. Optical conductivity calculation  
- Add `berry = true`, `berry_task = kubo`, `fermi_energy`, `berry_kmesh`, `kubo_freq_max` into `wannier90.win` file.  
- Run postw90 (postw90.x wannier90.win) to get `wannier90-kubo_*.dat` files.


### An example of `wannier90.win` file  
```
num_wann = 33   
num_bands = 85  
exclude_bands = 1-11  
dis_froz_max=7  
dis_froz_min=0  
dis_mix_ratio=1  
num_iter=400  
dis_num_iter=2200  

Begin Projections  
Cu:s;d  
Zn:s  
Sn:s;p  
Se:sp3  
End Projections  

wannier_plot = true  
wannier_plot_supercell = 3  
bands_plot      =  true  
begin kpoint_path  
Z 0.5 0.5 -0.5 G 0 0 0   
G 0 0 0 X 0 0 0.5  
X 0 0 0.5 P 0.25 0.25 0.25  
P 0.25 0.25 0.25 G 0 0 0   
G 0 0 0 N 0 0.5 0  
end kpoint_path  
kpath_task = curv+bands  
kpath_num_points = 1000  
bands_plot_format gnuplot  

berry = true  
berry_task = kubo  
kubo_freq_max = 5.0  
berry_kmesh =  180 180 180  
fermi_energy = 3.5  
kubo_freq_step = 0.005  
smr_type = m-p4  
adpt_smr_max = 0.05  
...
```

### Compilation guide  
- VASP: https://friendly-broccoli-22e4d939.pages.github.io/hpc_usage/cx1/compilation/vasp/vasp/  
- QE: https://friendly-broccoli-22e4d939.pages.github.io/hpc_usage/cx1/compilation/wannier90/  

  Note that you need to join the Imperial College Github organisation to access to it ([link](https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/research-support-systems/github/working-with-githubcom/)) 
  
### Tips  
1. You may need to increase `NBANDS` to get reasonable disentanglement window.
2. Check `spread` in the `wannier90.wout` file in step 3 to ensure spread is small enough (usually < 5 Ang^2).
3. Check the convergence with respect to `berry_kmesh` in step 4.
