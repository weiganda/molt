Molt

Ab initio complex dielectric function from AIMD dipole trajectories.

Molt transforms the time-domain dipole autocorrelation into ε(ω) by FFT, reaching the infrared through microwave regimes with no experimental input and no Berry-phase polarization. Includes VASP validation tooling for relaxation, timestep, and thermostat mass.

Author: Alysse Weigand Institution: University of Missouri – Kansas City

Overview

Molt is a Python pipeline for computing the complex dielectric function from VASP ab initio molecular dynamics (AIMD) trajectories. It builds a quasi-dipole moment time series from AIMD output, computes the dipole autocorrelation function (ACF), and applies a numerical derivative followed by a fast Fourier transform to produce the real and imaginary parts of the permittivity. From these it derives refractive index, extinction coefficient, absorption coefficient, reflectivity, energy loss function, and loss tangent.

Because the method rests on the fluctuation-dissipation theorem rather than on a harmonic expansion, it is not restricted to a single spectral window: the accessible frequency range is set by the trajectory length and sampling interval, not by the formalism. It also requires no Berry-phase polarization, so it remains defined for gapless and metallic systems.

The name follows the group convention. Imago is the adult form; molting is the change of representation that produces it. Molt sheds the time-domain form of a trajectory to expose the frequency-domain content that was already in it.

Molt is designed for Linux systems and HPC clusters.

Naming note. This code was previously called aiFTIR, and the dipole engine it calls was previously called OLCAO. The program is now Molt and the dipole engine is now Imago. OLCAO remains the name of the underlying method; Imago is the program that implements it. Older output directories, file headers, and scripts may still carry the former names.

Pipeline
  Stage 0   Structure preparation and relaxation
            skl2poscar.py -> relaxINCAR.py -> VASP -> validate_relaxation.py

  Stage 1   AIMD parameter validation
            nveINCAR.py -> VASP -> validate_POTIM.py
            nvtINCAR.py -> VASP -> validate_SMASS.py
            calculateSamplingRate.py

  Stage 2   Production AIMD
            VASP (NVE or NVT) -> XDATCAR
            unwrap_trajectory.py  (multi-run trajectories only)

  Stage 3   Dipole moment series
            xdatcar2skl.py -> sbatch submit_array.sh -> run_frame.sh (Imago)
            collect_dipoles.py -> dipole_series.csv + vol

  Stage 4   Dielectric function
            molt.py -> CDF.csv, CDF.png, Optical_Properties.png
Dependencies

External software

Software	Purpose
VASP	AIMD trajectories and structural relaxation
Imago	dipole moment calculation (Python implementation of the OLCAO method)
SLURM	required for the Stage 3 array job

Python

Python 3.x with numpy, scipy, matplotlib, pandas. skl2poscar.py additionally needs spglib when expanding an asymmetric unit with a real space group.

Scripts
Structure preparation
Script	Purpose
skl2poscar.py	Convert an OLCAO skeleton (.skl) file to a VASP POSCAR. Handles both already-expanded cells and asymmetric units requiring symmetry expansion.
relaxINCAR.py	Write INCAR and KPOINTS for a structural relaxation, and optionally assemble POTCAR from a local PAW library by reading species from the POSCAR.
validate_relaxation.py	Assess whether a relaxation has converged, from slopes in energy, maximum force, and cell volume over the final ionic steps. Writes a CSV, a summary, a LaTeX table fragment, and a three-panel figure.
AIMD setup and validation
Script	Purpose
nveINCAR.py	Generate an INCAR for NVE molecular dynamics (Andersen thermostat, zero collision probability).
nvtINCAR.py	Generate an INCAR for NVT molecular dynamics (Nosé-Hoover, MDALGO = 2).
validate_POTIM.py	Analyze total energy drift in NVE OUTCAR data to validate the AIMD timestep. Supports sweeps across POTIM values.
validate_SMASS.py	Validate the Nosé-Hoover thermostat mass against an NVE reference, using temperature autocorrelation, VDOS comparison, thermostat energy drift, and the steady-state temperature distribution.
calculateSamplingRate.py	Determine the MD sampling interval needed to resolve a target frequency without violating the Nyquist criterion.
Trajectory to dipole series
Script	Purpose
unwrap_trajectory.py	Remove periodic boundary discontinuities so sequential AIMD runs can be concatenated into one continuous trajectory.
xdatcar2skl.py	Convert XDATCAR frames into per-frame Imago input directories and write a SLURM array submission script.
run_frame.sh	Per-frame worker. Sets a per-frame IMAGO_TEMP scratch directory, runs the dipole calculation, extracts the dipole moment and cell volume, cleans up. Called by each array task; not run by hand.
collect_dipoles.py	Aggregate per-frame results into dipole_series.csv and copy the cell volume to vol. Reports missing or malformed frames.
Analysis
Script	Purpose
molt.py	Compute the dipole ACF, its derivative, the FFT, and the complex dielectric function, plus derived optical properties.
Stage 0 — Structure preparation
bash
# Convert an OLCAO skeleton file to POSCAR
python skl2poscar.py input.skl -o POSCAR
python skl2poscar.py input.skl --supercell 2 2 2

# Write INCAR, KPOINTS, and POTCAR for the relaxation
python relaxINCAR.py -npar 4 -k 2 2 2

# After the relaxation finishes
python validate_relaxation.py --label SiO2

relaxINCAR.py defaults to potpaw_LDA, matching the Ceperley-Alder functional. It locates the PAW library through --pp-root or the VASP_PP_PATH environment variable, and takes --pp-map Si=Si_sv to override the folder for a species. Use --no-potcar to skip POTCAR assembly entirely. VASP 4 style POSCARs require --species in POSCAR order.

validate_relaxation.py reports convergence in energy, force, and volume against criteria set by --energy-req, --force-req, and --volume-req, with the slope taken over the last -w ionic steps (default 5). Output lands in relaxation_output/.

Stage 1 — AIMD parameter validation
bash
# NVE: validate the timestep
python nveINCAR.py -npar 8 --sweep 0.25 0.5 1.0 -T 300 -nsw 2000
python validate_POTIM.py --sweep potim_0.25 potim_0.5 potim_1.0 --label SiO2

# NVT: validate the thermostat mass against an NVE reference
python nvtINCAR.py -npar 8 -potim 0.5 --grid -nsw 2000 -TB 300 -TE 300
python validate_SMASS.py --sweep smass_* --nve ../nve_reference --label SiO2

# Choose a sampling interval for the frequency range of interest
python calculateSamplingRate.py -f 1000 -u cm-1

Both validators take --drop to discard an initial transient (default 0.5), --outdir, --dpi, and --appendix to emit a LaTeX figure fragment alongside the figures. validate_SMASS.py additionally takes --fmax for the upper limit of the VDOS comparison (default 1400 cm⁻¹) and --selftest, which splits the NVE reference in half and compares the halves to establish a noise floor.

Omit --sweep on either validator to analyze the current directory instead of a set of run directories.

The sampling interval reported by calculateSamplingRate.py is what you pass to xdatcar2skl.py -sr and to molt.py -s. These two values must match.

Stage 2 — Production AIMD

Run VASP AIMD in either ensemble. Molt needs the XDATCAR; POTCAR is read only to identify element symbols. Record the POTIM value, which is required in Stage 4.

If the trajectory spans multiple sequential runs, VASP rewraps positions into the cell at each restart, which introduces discontinuities that will corrupt the dipole series. Unwrap each run before concatenating:

bash
cd run1 && python unwrap_trajectory.py -initial
cp last_unwrapped_Configuration ../run2/
cd ../run2 && python unwrap_trajectory.py -continue
# repeat, then concatenate the XDATCAR_unwrapped files

The method assumes no atom moves more than half a box length in one step, which holds for any sane timestep.

Stage 3 — Dipole moment series
bash
# Build per-frame directories and the SLURM array script
python xdatcar2skl.py -sr 10 -d -scf mb -j 32

# Submit, then aggregate once the array finishes
sbatch submit_array.sh
python collect_dipoles.py

Basis selection is one of two mutually exclusive routes: -scf {mb,fb,eb} selects an Imago native contract basis, while -basisdb selects a pre-built basis database, given either as a shorthand name resolved under $IMAGO_DATA/atomicPBDB/ or as an explicit path. Basis choice has a strong effect on the resulting spectrum and should be treated as a parameter of the calculation rather than a detail.

SLURM resources are set with -j (concurrency cap, default 32), --partition, --account, --time (default 06:00:00), and --mem (default 10G). --imagorc gives the path to the Imago environment file sourced by each job.

Stage 3 produces:

dipole_series.csv — three columns (x, y, z) in units of a₀·e, no header
vol — cell volume in Å³, a single number

If frames fail, collect_dipoles.py reports them; check logs/frame_<task>.err and resubmit individually with sbatch --array=<task_id> submit_array.sh.

Stage 4 — Dielectric function
bash
python molt.py -s 10 -a 0.5 -T 300
python molt.py -s 10 -a 0.5 -T 300 -u eV --plot-acf --plot-derivative --plot-fft-xyz
Option	Meaning
-f, --file	Input CSV of dipole vectors (default dipole_series.csv)
-s, --sampling	Sampling interval in MD steps. Must match xdatcar2skl.py -sr. Required.
-a, --aimd	AIMD timestep in fs. Must match POTIM. Required.
-T, --temperature	Average simulation temperature in K. Required.
-u, --units	cm1 (default), eV, or Hz
-o, --outdir	Output directory
--plot-acf	Diagnostic plot of the ACF for x, y, z, and the isotropic average
--plot-derivative	Diagnostic plot of −dΦ/dt on a picosecond axis
--plot-fft-xyz	Per-component real and imaginary FFT output
-v, --verbose	Debug logging

Method. The unnormalized ACF ⟨M(t)·M(t+k)⟩ is computed per component and averaged. A Savitzky-Golay filter (window 5, polyorder 3) gives the numerical first derivative. The negative derivative is Fourier transformed; the complex conjugate is taken to move from NumPy's −iωt convention to +iωt, and only positive frequencies are retained. Gaussian smoothing (σ = 1) is applied, and the permittivity follows as

ε(ω) = 1 + (4π / k_B T) · FFT(−dACF/dt)

in CGS throughout, with dipole moments converted from a₀·e to esu·cm (1 a₀·e = 2.541745e-18 esu·cm).

Output. CDF.csv and CDF.png (ε₁ and ε₂ vs. frequency), Optical_Properties.png (six panels: n, k, α, R and T, ELF, loss tangent), and simulation_info.txt (run parameters and the resolvable frequency window).

Limitations and notes
The lowest resolvable frequency is set by total simulated time; the highest by the Nyquist limit, f_max = 1 / (2 · sampling interval).
Electronic contributions to the high-frequency response are not included.
vol must be present in the working directory before running molt.py. It is produced by collect_dipoles.py.
Stage 3 requires SLURM. submit_array.sh would need adaptation for PBS, LSF, or another scheduler.
The --imagorc default points at the original development environment. Pass the correct path for your Imago installation.
Results depend strongly on basis set. Report the basis alongside any spectrum.
License

Copyright 2026 Alysse Weigand.

Licensed under the Educational Community License, Version 2.0 (the "License"). You may obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0. See LICENSE for the full text.

This matches the license used by Imago.

---

## Citation

If you use Molt in your work, please cite the archived release:

```bibtex
@misc{weigand_molt_2026,
  author    = {Weigand, Alysse},
  title     = {{Molt}: Ab Initio Complex Dielectric Function from MD Dipole Trajectories},
  year      = {2026},
  version   = {v1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22651110},
  url       = {https://github.com/weiganda/molt}
}
```

Molt calls Imago for dipole moment calculations. Imago is a separate program and
is not distributed here; please cite it separately, along with the OLCAO method
reference.
