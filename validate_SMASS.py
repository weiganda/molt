#!/usr/bin/env python3
"""
PROGRAM: validate_SMASS.py
PURPOSE: Validates the Nose-Hoover thermostat mass (SMASS) for NVT molecular
         dynamics by asking whether the thermostat holds the target temperature
         without distorting the vibrational dynamics of the material. Analyses
         a sweep of runs at different SMASS values against an NVE reference,
         and writes the figure, LaTeX table, and CSV needed for the
         dissertation.

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Conceptual design, purpose, and validation by Alysse Weigand.
                 All scientific reasoning, method choices, and interpretation
                 of results by Alysse Weigand.
                 Code implementation and structure assisted by ChatGPT
                 (original version, December 2025) and Claude (Anthropic),
                 September 2026.
LAST MODIFIED:   September 2026

USAGE:
    python validate_SMASS.py --label SiO2 --sweep smass_* --nve NVE_300
    python validate_SMASS.py --label SiO2 --sweep smass_* --nve DIR --appendix

INPUT:
    Each sweep directory needs an OUTCAR and a trajectory from an NVT run.
    The NVE directory needs the same pair from a microcanonical run of the
    same system, and is required: it is the reference against which the
    dynamics are judged.

    The OUTCAR supplies the temperature series and the thermostat energies.
    The trajectory comes from vasprun.xml, or from XDATCAR when vasprun.xml
    is not kept, which is the usual case for a long production run. VASP does not write per-step
    velocities to the OUTCAR under these settings, so velocities are
    obtained by differentiating the positions. With a timestep of a fraction
    of a femtosecond this is well resolved for every vibrational frequency in
    these materials.

    A run stopped by a wall-clock limit leaves its trajectory file without a
    proper ending. Both readers stop at the truncation, keep what was written
    before it, and warn. The number of frames actually recovered is reported
    per run and written to the CSV, since it is that number rather than the
    step count in the OUTCAR that sets what the VDOS was measured over.

OUTPUT:
    SMASS_<label>.png / .pdf        Summary figure across the sweep
    SMASS_<label>_diagnostics.png   Per-run diagnostic panels
    SMASS_<label>.csv               Per-run diagnostic values
    SMASS_<label>.tex               LaTeX table
    SMASS_<label>_fig.tex           Figure fragment (--appendix only)

WHAT THIS SCRIPT IS ACTUALLY TESTING

    The dielectric function computed in this work comes from the time
    correlation of the total dipole moment. That is a statement about
    dynamics, not about statistics. A thermostat that reproduces the correct
    temperature distribution while altering how the atoms move in time will
    produce a spectrum with the wrong linewidths, and nothing in the
    temperature statistics alone would reveal it.

    The criteria are therefore ordered by what the calculation depends on:

    1. Vibrational dynamics.
       The vibrational density of states is computed from the velocity
       autocorrelation function and compared against the same quantity from an
       NVE run of the same system. NVE has no thermostat, so its VDOS is the
       natural dynamics of the material. This is the criterion that matters
       most, because the vibrational motion is what the dipole correlation
       function is built from.

       The difference is judged against the noise floor of the reference
       itself, found by splitting the NVE trajectory in half and comparing
       the halves. In a small cell the height of each VDOS line is set by how
       much energy that mode currently holds, and that wanders even with no
       thermostat present. A run is only called distorted if it departs from
       the reference by clearly more than the reference departs from itself.
       If every run sits at the floor, this criterion passes them all and
       cannot rank them; the report says so.

       The comparison assumes the reference resolves the spectrum as well as
       the runs being judged. A reference much shorter than the NVT runs
       returns a smooth envelope where the runs return resolved lines, and
       the difference between the two is then a matter of resolution rather
       than of thermostat distortion. Its split-half noise floor is small for
       the same reason, so the threshold tightens at exactly the moment it
       should loosen. The report warns when the reference is short.

    2. Mean temperature.
       The time-averaged temperature must be close to the target, because it
       enters the linear response prefactor directly.

    3. Canonical temperature distribution.
       In the canonical ensemble the kinetic temperature of g degrees of
       freedom follows a chi-squared distribution with g degrees of freedom,
       so both its width and its shape are known in advance:
           sigma_exp = T_avg * sqrt( 2 / g )
           skew_exp  = sqrt( 8 / g )
       The observed width is compared against sigma_exp with a band set by
       the sampling uncertainty of the run, 1 / sqrt(2 N_eff), widened to
       three standard errors and floored at 10%. The observed skewness is
       compared against skew_exp with a band of three standard errors,
       sqrt(6 / N_eff). A thermostat that rings produces the right width
       with the wrong shape: the distribution piles up below the mean and
       grows a long tail of hot excursions.

       This is a check that the thermostat is sampling sensibly, and it is
       subordinate to the first two: correct statistics obtained by destroying
       the dynamics are of no use here.

    4. Thermostat energy stationarity.
       The Nose kinetic and potential terms are summed and fitted with a
       straight line. The significance of the fitted slope is corrected for
       the autocorrelation of the residuals, without which an oscillating
       thermostat energy is reported as a highly significant linear drift.

DEGREES OF FREEDOM

    VASP's Nose thermostat removes the motion of the centre of mass, so it
    acts on g = 3N - 3 degrees of freedom, and an NVE run conserves the same
    momentum. Temperatures here are therefore computed as
        T = 2 EKIN / ( (3N - 3) k_B )
    which is the convention VASP itself uses. Dividing by 3N instead reads
    every run low by a factor (3N - 3) / 3N, which for 18 atoms is 5.6% and
    enough on its own to fail the temperature criterion.
"""

##################################################
# Imports
##################################################

import os
import re
import csv
import glob
import logging
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gamma, skew
from scipy.ndimage import gaussian_filter1d

##################################################
# Logging
##################################################

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

##################################################
# Criteria
##################################################

VDOS_FLOOR_FACTOR = 1.5 # a run is distorted above this multiple of the
                        #   NVE split-half noise floor
VDOS_DIFF_MAX = 0.15    # fallback threshold, used only if the self-test fails
T_TOLERANCE = 0.05      # mean temperature within this fraction of the target
ACF_THRESHOLD = 0.1     # temperature correlation considered lost below this
N_EFF_MIN = 30          # gives ~13% precision on sigma_obs
SIGMA_NSIGMA = 3.0      # acceptance band, in sampling standard errors
SIGMA_FLOOR = 0.10      # narrowest fractional band ever applied
SLOPE_TSTAT = 2.0       # corrected |slope| / SE(slope) above this is a drift
FRAME_MISMATCH = 0.02   # warn when the trajectory and the OUTCAR disagree on
                        #   length by more than this fraction
REF_SHORT_FACTOR = 0.5  # warn when the reference trajectory is shorter than
                        #   this fraction of the runs it is judging

CM1_PER_HZ = 3.33564e-11
FS_TO_PS = 1.0e-3
KB_EV = 8.617333262e-5     # Boltzmann constant, eV/K

##################################################
# Parsing
##################################################


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate the Nose-Hoover thermostat mass against an NVE "
                    "reference.")
    p.add_argument("--label", required=True,
                   help="Material label used in filenames and captions.")
    p.add_argument("--sweep", nargs="+", metavar="DIR", default=None,
                   help="Directories, one per SMASS value, each holding an "
                        "OUTCAR. Omit to analyse the current directory.")
    p.add_argument("--nve", metavar="DIR", required=True,
                   help="Directory holding an OUTCAR from an NVE run of the "
                        "same system. This is the reference for the dynamics.")
    p.add_argument("--target-t", type=float, default=None, metavar="K",
                   help="Target temperature. Read from TEBEG if omitted.")
    p.add_argument("--drop", type=float, default=0.5, metavar="FRAC",
                   help="Fraction of the trajectory discarded before analysis. "
                        "(default: 0.5)")
    p.add_argument("--fmax", type=float, default=1400.0, metavar="CM1",
                   help="Upper limit of the VDOS comparison, in cm^-1. "
                        "(default: 1400)")
    p.add_argument("--selftest", action="store_true",
                   help="Accepted for compatibility. The self-test now always "
                        "runs, because criterion 1 is judged against its "
                        "noise floor.")
    p.add_argument("--appendix", action="store_true",
                   help="Also write a LaTeX figure fragment for the appendix.")
    p.add_argument("--outdir", default=".",
                   help="Directory for the output files. (default: .)")
    p.add_argument("--dpi", type=int, default=300,
                   help="Figure resolution. (default: 300)")
    args = p.parse_args()
    if not 0.0 <= args.drop < 1.0:
        p.error("--drop must be in [0, 1)")
    return args


def read_tag(path, tag):
    """Return the float value of an INCAR-style tag, or None."""
    pattern = re.compile(rf"\b{tag}\b\s*=\s*([-+]?\d*\.?\d+([eE][-+]?\d+)?)")
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                line = line.split("#")[0].split("!")[0]
                m = pattern.search(line)
                if m:
                    return float(m.group(1))
    except FileNotFoundError:
        return None
    return None


def read_outcar(directory):
    """Pull the atom count, temperature series, and Nose energy terms."""
    outcar = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(outcar):
        raise FileNotFoundError("no OUTCAR in %s" % directory)

    natoms = None
    ekin, nose_kin, nose_pot = [], [], []

    with open(outcar, errors="replace") as fh:
        for line in fh:
            if natoms is None and "NIONS" in line:
                m = re.search(r"NIONS\s*=\s*(\d+)", line)
                if m:
                    natoms = int(m.group(1))

            # The ionic temperature is derived from the ionic kinetic energy
            #   rather than read directly. VASP prints a line reading
            #   "kin. lattice EKIN_LAT = ... (temperature ... K)" which is the
            #   temperature of the cell degrees of freedom, not of the ions,
            #   and which is zero or meaningless when the cell is held fixed.
            #   Matching on the word "temperature" picks up that line instead
            #   of the quantity wanted here.
            low = line.lower()
            if low.strip().startswith("kinetic energy ekin"):
                m = re.search(r"EKIN\s*=\s*([-+]?[\d.eE+-]+)", line, re.I)
                if m:
                    ekin.append(float(m.group(1)))

            # "nose kinetic EPS" is the thermostat kinetic term and
            # "nose potential ES" is the thermostat potential term. These two
            # were transposed in an earlier version of this script; the sum was
            # unaffected but the individual traces were mislabelled.
            if "nose kinetic   eps" in low:
                nose_kin.append(float(line.split()[-1]))
            elif "nose potential es" in low:
                nose_pot.append(float(line.split()[-1]))

    if natoms is None:
        raise ValueError("could not find NIONS in %s" % outcar)
    if len(ekin) < 40:
        raise ValueError(
            "only %d ionic kinetic energies found in %s; need at least 40. "
            "This script reads the line 'kinetic energy EKIN' from the "
            "molecular dynamics output block." % (len(ekin), outcar))

    # Equipartition over g = 3N - 3 degrees of freedom, the three
    #   centre-of-mass translations being removed by the thermostat and
    #   conserved in NVE. T = 2 EKIN / (g k_B), as VASP reports it.
    ekin = np.asarray(ekin, float)
    temps = 2.0 * ekin / (degrees_of_freedom(natoms) * KB_EV)

    return {
        "natoms": natoms,
        "ekin": ekin,
        "temps": temps,
        "nose_kin": np.asarray(nose_kin, float) if nose_kin else None,
        "nose_pot": np.asarray(nose_pot, float) if nose_pot else None,
    }


def degrees_of_freedom(natoms):
    """Kinetic degrees of freedom with the centre of mass removed."""
    return 3 * natoms - 3


def _iterparse_tolerant(path, events, state):
    """
    Yield parse events from an XML file, stopping quietly at a truncation.

    A run stopped by a wall-clock limit leaves vasprun.xml without its closing
    tags, and a strict parse of the whole file then fails at the last line with
    no element found. Every element that closed before the cut is complete, so
    the frames written up to that point are intact and usable. The caller is
    told through state so it can report how much of the run survived.
    """
    import xml.etree.ElementTree as ET
    try:
        for item in ET.iterparse(path, events=events):
            yield item
    except ET.ParseError as exc:
        state["truncated"] = True
        state["error"] = str(exc)


def read_trajectory(directory):
    """
    Read the trajectory from vasprun.xml, or from XDATCAR if vasprun.xml is
    absent, and return Cartesian positions with shape (steps, atoms, 3), in
    Angstrom.

    Positions are fractional. Atoms that cross a periodic boundary jump by
    close to a full cell length, which would dominate any derivative taken
    from the raw values, so the fractional coordinates are unwrapped before
    being converted through the lattice vectors.

    Only the structures inside each <calculation> block of vasprun.xml are MD
    steps. The file also holds named structures (initialpos, finalpos,
    primitive_cell) with their own positions arrays; reading those as frames
    puts extra, misaligned configurations at the ends of the series, which
    shows up as cell-sized jumps between adjacent frames.

    Either file may end mid-write if the run was stopped before it finished.
    Both readers keep the complete frames and stop there.
    """
    path = os.path.join(directory, "vasprun.xml")
    xdat = os.path.join(directory, "XDATCAR")

    if os.path.isfile(path):
        lattice = None
        frames = []
        in_calc = 0
        struct_name = []
        state = {"truncated": False}
        # iterparse keeps memory flat: a long MD run makes a large file.
        for event, elem in _iterparse_tolerant(path, ("start", "end"), state):
            tag = elem.tag
            if event == "start":
                if tag == "calculation":
                    in_calc += 1
                elif tag == "structure":
                    struct_name.append(elem.get("name"))
                continue
            if tag == "calculation":
                in_calc -= 1
                elem.clear()
                continue
            if tag == "structure":
                struct_name.pop()
                continue
            if tag != "varray":
                continue
            name = elem.get("name")
            if name == "basis" and lattice is None:
                lattice = np.array([[float(x) for x in v.text.split()]
                                    for v in elem.findall("v")], dtype=float)
            elif (name == "positions" and in_calc and struct_name
                  and struct_name[-1] is None):
                frames.append(np.array([[float(x) for x in v.text.split()]
                                        for v in elem.findall("v")], dtype=float))
                elem.clear()
        if state["truncated"]:
            log.warning("vasprun.xml in %s ends mid-write, most likely a run "
                        "stopped before it finished; %d complete frames "
                        "recovered", directory, len(frames))
        if lattice is None:
            raise ValueError("no lattice vectors found in %s" % path)
        frac = np.asarray(frames)
    elif os.path.isfile(xdat):
        path = xdat
        lattice, frac = read_xdatcar(xdat)
    else:
        raise FileNotFoundError(
            "no vasprun.xml or XDATCAR in %s. The trajectory is needed for "
            "the VDOS comparison and VASP does not write per-step velocities "
            "to the OUTCAR under these settings." % directory)

    if len(frac) < 40:
        raise ValueError("only %d trajectory frames in %s; need at least 40"
                         % (len(frac), path))

    # Unwrap: any step of more than half a cell is a boundary crossing.
    deltas = np.diff(frac, axis=0)
    deltas -= np.round(deltas)
    frac = np.concatenate([frac[:1], frac[:1] + np.cumsum(deltas, axis=0)],
                          axis=0)

    return frac @ lattice


def read_xdatcar(path, lattice_only=False):
    """
    Lattice (Angstrom) and fractional positions (steps, atoms, 3) from an
    XDATCAR, used when vasprun.xml is missing. Handles VASP 4 and 5+ headers
    and the variable-cell layout (header repeated per frame); in that case
    the first cell is returned.
    """
    with open(path, errors="replace") as fh:
        lines = fh.read().splitlines()

    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]]
                        for i in (2, 3, 4)], dtype=float)
    if scale < 0:                       # negative scale is the cell volume
        scale = (-scale / abs(np.linalg.det(lattice))) ** (1.0 / 3.0)
    lattice *= scale
    if lattice_only:
        return lattice, None

    # VASP 5+ puts species names on line 6 and counts on line 7.
    try:
        natoms = sum(int(x) for x in lines[5].split())
    except ValueError:
        natoms = sum(int(x) for x in lines[6].split())

    frames = []
    i = 0
    truncated = False
    while i < len(lines):
        if lines[i].lstrip().lower().startswith("direct configuration"):
            block = lines[i + 1:i + 1 + natoms]
            if len(block) < natoms:     # truncated final frame
                truncated = True
                break
            frames.append([[float(x) for x in l.split()[:3]] for l in block])
            i += natoms + 1
        else:
            i += 1

    if truncated:
        log.warning("%s ends mid-frame, most likely a run stopped before it "
                    "finished; %d complete frames recovered", path, len(frames))

    return lattice, np.asarray(frames, dtype=float)


def velocities_from_positions(positions, potim_fs):
    """
    Central-difference velocities in Angstrom per femtosecond.

    The first and last frames are dropped, since a central difference is not
    defined at the ends of the series.
    """
    return (positions[2:] - positions[:-2]) / (2.0 * potim_fs)


##################################################
# Core quantities
##################################################


def autocorrelation(x):
    """Normalised autocorrelation of a 1-D signal, lags 0 .. N-1."""
    x = x - x.mean()
    full = np.correlate(x, x, mode="full")[len(x) - 1:]
    return full / full[0] if full[0] != 0 else full


def velocity_acf(vel):
    """
    Mass-independent velocity autocorrelation function, averaged over atoms
    and Cartesian components.

    vel has shape (steps, atoms, 3).
    """
    nsteps = vel.shape[0]
    acf = np.zeros(nsteps)
    count = 0
    for a in range(vel.shape[1]):
        for k in range(3):
            series = vel[:, a, k]
            series = series - series.mean()
            full = np.correlate(series, series, mode="full")[nsteps - 1:]
            if full[0] != 0:
                acf += full / full[0]
                count += 1
    return acf / count if count else acf


def vdos_from_vacf(vacf, potim_fs, fmax_cm1):
    """
    Vibrational density of states as the cosine transform of the velocity
    autocorrelation function, returned on a wavenumber axis.

    A Hann taper is applied before the transform for the same reason it is
    applied elsewhere in this work: the correlation function does not reach
    zero at the end of the available lags, and transforming a signal with a
    discontinuity at its edge produces ripple across the whole spectrum.

    Only the real part of the transform is kept. The correlation function is
    one-sided and starts at 1, so the imaginary (sine) part of its transform
    carries a tail that falls off as 1/omega. Taking the magnitude, as an
    earlier version did, folded that tail into the spectrum and raised the
    apparent weight at high wavenumber well above the real modes. The real
    part is the cosine transform proper.
    """
    n = vacf.size
    window = np.hanning(2 * n)[n:]
    tapered = vacf * window

    spectrum = np.fft.rfft(tapered, n=4 * n).real
    freqs_hz = np.fft.rfftfreq(4 * n, d=potim_fs * 1e-15)
    wavenumbers = freqs_hz * CM1_PER_HZ

    mask = wavenumbers <= fmax_cm1
    return wavenumbers[mask], spectrum[mask]


def normalised_difference(x_a, y_a, x_b, y_b, x_max):
    """
    Integrated absolute difference between two curves, each first normalised
    to unit area so the comparison is of shape rather than magnitude.

    Zero means identical. Two is the maximum, reached when the two curves do
    not overlap at all.
    """
    grid = np.linspace(0.0, x_max, 3000)
    a = np.interp(grid, x_a, y_a)
    b = np.interp(grid, x_b, y_b)

    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area_a, area_b = trapz(a, grid), trapz(b, grid)
    if area_a <= 0 or area_b <= 0:
        return float("nan")

    return float(trapz(np.abs(a / area_a - b / area_b), grid))


def load_nve_reference(directory, drop_fraction, fmax_cm1):
    """VDOS of an NVE run, the natural-dynamics baseline."""
    data = read_outcar(directory)
    potim = read_tag(os.path.join(directory, "INCAR"), "POTIM")
    if potim is None:
        potim = read_tag(os.path.join(directory, "OUTCAR"), "POTIM")
    if potim is None:
        raise ValueError("could not find POTIM for the NVE reference")

    positions = read_trajectory(directory)
    vel = velocities_from_positions(positions, potim)

    i0 = int(drop_fraction * vel.shape[0])
    vacf = velocity_acf(vel[i0:])
    wn, vdos = vdos_from_vacf(vacf, potim, fmax_cm1)

    i0t = int(drop_fraction * data["temps"].size)
    return {"directory": directory, "natoms": data["natoms"], "potim": potim,
            "nframes": len(positions), "n_analysed": vel.shape[0] - i0,
            "wavenumbers": wn, "vdos": vdos,
            "t_avg": float(data["temps"][i0t:].mean())}


##################################################
# Self-test
##################################################


def selftest(directory, fmax_cm1):
    """
    Split one trajectory in half and compare the two halves.

    Both halves describe the same system under the same conditions, so any
    difference between their vibrational densities of states is measurement
    noise: finite trajectory length, finite frequency resolution, and the
    statistical quality of the velocity autocorrelation function. The value
    returned is therefore the floor below which a difference between two
    different runs cannot be regarded as meaningful.

    If that floor is comparable to the differences seen across a SMASS sweep,
    the sweep is not resolving anything and the trajectories need to be longer.

    The floor is only a floor for the comparison it is used in. Halving an
    already short reference gives two smooth, featureless spectra that agree
    closely with each other, and the small number that results says nothing
    about how well that reference can be compared with a longer run.
    """
    data = read_outcar(directory)
    potim = read_tag(os.path.join(directory, "INCAR"), "POTIM")
    if potim is None:
        potim = read_tag(os.path.join(directory, "OUTCAR"), "POTIM")
    if potim is None:
        raise ValueError("could not find POTIM for the self-test")

    positions = read_trajectory(directory)
    vel = velocities_from_positions(positions, potim)

    n = vel.shape[0]
    half = n // 2
    first, second = vel[:half], vel[half:2 * half]

    wn_a, vdos_a = vdos_from_vacf(velocity_acf(first), potim, fmax_cm1)
    wn_b, vdos_b = vdos_from_vacf(velocity_acf(second), potim, fmax_cm1)
    floor = normalised_difference(wn_a, vdos_a, wn_b, vdos_b, fmax_cm1)

    window_ps = half * potim * FS_TO_PS
    resolution_cm1 = 1.0 / (window_ps * 1e-12) * CM1_PER_HZ

    return {"directory": directory, "floor": floor, "potim": potim,
            "half_steps": half, "window_ps": window_ps,
            "resolution_cm1": resolution_cm1,
            "wavenumbers_a": wn_a, "vdos_a": vdos_a,
            "wavenumbers_b": wn_b, "vdos_b": vdos_b}


def print_selftest(st, runs=None):
    print("\n------------------- VDOS Resolution Self-Test -------------------")
    print("Trajectory             = %s" % st["directory"])
    print("Each half              = %d steps, %.3f ps" %
          (st["half_steps"], st["window_ps"]))
    print("Frequency resolution   = %.0f cm^-1 before tapering"
          % st["resolution_cm1"])
    print("Difference between the two halves = %.3f" % st["floor"])
    print("-" * 64)
    print("Both halves are the same system under the same conditions, so this")
    print("difference is measurement noise. A difference between two separate")
    print("runs is only meaningful if it is well above this value.")

    if runs:
        spread = max(r["vdos_diff"] for r in runs) - min(r["vdos_diff"] for r in runs)
        smallest = min(r["vdos_diff"] for r in runs)
        print()
        print("Across the sweep: smallest difference %.3f, spread %.3f"
              % (smallest, spread))
        if st["floor"] >= 0.5 * spread:
            print()
            print("  -> The noise floor is comparable to the spread across the")
            print("     sweep. The ranking of SMASS values is not resolved by")
            print("     these trajectories. Longer runs are needed before any")
            print("     value can be preferred over another.")
        elif st["floor"] >= 0.5 * smallest:
            print()
            print("  -> The noise floor is a substantial fraction of even the")
            print("     smallest departure from the reference. Differences")
            print("     between runs should be treated with caution.")
        else:
            print()
            print("  -> The noise floor is well below the differences seen")
            print("     across the sweep, so those differences are real.")
    print("-----------------------------------------------------------------\n")


def make_selftest_figure(st, stem, dpi, fmax_cm1):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(st["wavenumbers_a"], st["vdos_a"] / st["vdos_a"].max(),
            color="tab:blue", linewidth=1.2, label="first half")
    ax.plot(st["wavenumbers_b"], st["vdos_b"] / st["vdos_b"].max(),
            color="tab:red", linewidth=1.2, linestyle="--", label="second half")
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized VDOS")
    ax.set_xlim(0, fmax_cm1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Self-test: two halves of one trajectory, difference %.3f"
                 % st["floor"])
    fig.tight_layout()
    fig.savefig("%s_selftest.png" % stem, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


##################################################
# Analysis
##################################################


def analyse_run(directory, drop_fraction, nve, target_t, fmax_cm1,
                vdos_threshold):
    """Apply the criteria to a single NVT directory."""
    data = read_outcar(directory)
    natoms = data["natoms"]
    temps = data["temps"]

    potim = read_tag(os.path.join(directory, "INCAR"), "POTIM")
    if potim is None:
        potim = read_tag(os.path.join(directory, "OUTCAR"), "POTIM")
    if potim is None:
        raise ValueError("could not find POTIM for %s" % directory)

    smass = read_tag(os.path.join(directory, "INCAR"), "SMASS")
    if smass is None:
        smass = read_tag(os.path.join(directory, "OUTCAR"), "SMASS")
    if smass is None:
        raise ValueError("could not find SMASS for %s" % directory)

    if target_t is None:
        target_t = read_tag(os.path.join(directory, "INCAR"), "TEBEG")

    nsteps = temps.size
    i0 = int(drop_fraction * nsteps)
    t_steady = temps[i0:]
    n_steady = t_steady.size

    # --- 1. vibrational dynamics against the NVE reference -----------------
    positions = read_trajectory(directory)
    nframes = len(positions)

    # The temperature series comes from the OUTCAR and the VDOS from the
    #   trajectory file. A run stopped before it finished leaves the two at
    #   different lengths, so the statistics and the spectrum are then
    #   measured over different windows of the same run.
    if nsteps and abs(nframes - nsteps) > FRAME_MISMATCH * nsteps:
        log.warning("%s: the OUTCAR holds %d ionic steps but the trajectory "
                    "holds %d frames; the temperature statistics and the VDOS "
                    "are measured over different windows",
                    directory, nsteps, nframes)

    vel = velocities_from_positions(positions, potim)

    iv = int(drop_fraction * vel.shape[0])
    vacf = velocity_acf(vel[iv:])
    wn, vdos = vdos_from_vacf(vacf, potim, fmax_cm1)
    vdos_diff = normalised_difference(wn, vdos,
                                      nve["wavenumbers"], nve["vdos"], fmax_cm1)
    dynamics_ok = vdos_diff < vdos_threshold

    # --- 2. mean temperature ------------------------------------------------
    t_avg = float(t_steady.mean())
    if target_t:
        t_error = abs(t_avg - target_t) / target_t
        temperature_ok = t_error <= T_TOLERANCE
    else:
        t_error = float("nan")
        temperature_ok = True

    # --- 3. canonical fluctuation magnitude ---------------------------------
    c = autocorrelation(t_steady)
    below = np.where(np.abs(c) < ACF_THRESHOLD)[0]
    if below.size:
        tau_c = int(below[0])
        n_eff = n_steady / tau_c if tau_c > 0 else float(n_steady)
    else:
        tau_c = None
        n_eff = 0.0

    g = degrees_of_freedom(natoms)
    sigma_obs = float(t_steady.std(ddof=1))
    sigma_exp = float(t_avg * np.sqrt(2.0 / g))
    sigma_ratio = sigma_obs / sigma_exp if sigma_exp > 0 else float("nan")

    rel_err = 1.0 / np.sqrt(2.0 * n_eff) if n_eff > 0 else float("inf")
    band = max(SIGMA_NSIGMA * rel_err, SIGMA_FLOOR)
    width_ok = abs(sigma_ratio - 1.0) <= band

    # Shape. A chi-squared distribution with g degrees of freedom has
    #   skewness sqrt(8/g), so a correct thermostat is expected to be mildly
    #   right-skewed. The standard error of a sample skewness is sqrt(6/N),
    #   with N replaced by the number of independent samples.
    skew_obs = float(skew(t_steady))
    skew_exp = float(np.sqrt(8.0 / g))
    skew_band = SIGMA_NSIGMA * np.sqrt(6.0 / n_eff) if n_eff > 0 else float("inf")
    shape_ok = abs(skew_obs - skew_exp) <= skew_band

    fluct_ok = (n_eff >= N_EFF_MIN) and width_ok and shape_ok

    # --- 4. thermostat energy stationarity ----------------------------------
    # The standard error of the fitted slope assumes independent residuals.
    # The thermostat energy is strongly autocorrelated, so the raw standard
    # error is too small by roughly the square root of the correlation time.
    if data["nose_kin"] is not None and data["nose_pot"] is not None:
        e_steady = (data["nose_kin"] + data["nose_pot"])[i0:]
        time_ps = np.arange(i0, nsteps) * potim * FS_TO_PS
        coeffs, cov = np.polyfit(time_ps, e_steady, 1, cov=True)
        slope_e = float(coeffs[0])
        slope_err = float(np.sqrt(cov[0, 0]))
        sigma_e = float(np.std(e_steady, ddof=1))

        e_acf = autocorrelation(e_steady)
        e_below = np.where(np.abs(e_acf) < ACF_THRESHOLD)[0]
        tau_e = int(e_below[0]) if e_below.size else n_steady // 10
        slope_err_corr = slope_err * np.sqrt(max(tau_e, 1))
        slope_t = abs(slope_e) / slope_err_corr if slope_err_corr > 0 else float("inf")
        stationary = slope_t < SLOPE_TSTAT
    else:
        e_steady = None
        slope_e = slope_err_corr = sigma_e = float("nan")
        tau_e = None
        slope_t = float("nan")
        stationary = True

    tau_c_fs = tau_c * potim if tau_c is not None else float("nan")

    # --- verdict ------------------------------------------------------------
    if not dynamics_ok:
        verdict = "dynamics distorted"
    elif not temperature_ok:
        verdict = "temperature off"
    elif not fluct_ok:
        if n_eff < N_EFF_MIN:
            verdict = "undersampled"
        elif not width_ok:
            verdict = ("fluctuations large" if sigma_ratio > 1.0
                       else "fluctuations small")
        else:
            verdict = "distribution skewed"
    elif not stationary:
        verdict = "energy drifting"
    else:
        verdict = "acceptable"

    return {
        "directory": directory, "smass": smass, "potim": potim,
        "natoms": natoms, "nsteps": nsteps, "nframes": nframes,
        "n_steady": n_steady,
        "temps": temps, "temps_steady": t_steady,
        "nose_kin": data["nose_kin"], "nose_pot": data["nose_pot"],
        "energy_steady": e_steady,
        "wavenumbers": wn, "vdos": vdos, "vdos_diff": vdos_diff,
        "dynamics_ok": dynamics_ok,
        "target_t": target_t, "t_avg": t_avg, "t_error": t_error,
        "temperature_ok": temperature_ok,
        "tau_c": tau_c, "tau_c_fs": tau_c_fs, "n_eff": n_eff,
        "sigma_obs": sigma_obs, "sigma_exp": sigma_exp,
        "sigma_ratio": sigma_ratio, "sigma_band": band, "fluct_ok": fluct_ok,
        "ndof": g, "skew_obs": skew_obs, "skew_exp": skew_exp,
        "skew_band": skew_band,
        "slope_e": slope_e, "slope_err": slope_err_corr, "slope_t": slope_t,
        "tau_e": tau_e, "sigma_e": sigma_e, "stationary": stationary,
        "verdict": verdict,
    }


##################################################
# Reporting
##################################################


def print_report(runs, drop_fraction, nve, fmax_cm1, vdos_threshold, st):
    print("\n---------- Simulation Parameters ----------")
    print("Atoms (NIONS)          = %d" % runs[0]["natoms"])
    print("Degrees of freedom     = %d   (3N - 3)" % runs[0]["ndof"])
    print("POTIM                  = %g fs" % runs[0]["potim"])
    print("Runs analysed          = %d" % len(runs))
    print("Analysis uses the last %d%% of each trajectory"
          % round((1.0 - drop_fraction) * 100))

    # Trajectory lengths, which are not always the step counts in the OUTCAR.
    frames = [r["nframes"] for r in runs]
    if min(frames) == max(frames):
        print("Trajectory frames      = %d per run" % frames[0])
    else:
        print("Trajectory frames      = %d to %d across the sweep"
              % (min(frames), max(frames)))
        print("  ! the runs are not all the same length. A shorter run has")
        print("    fewer independent samples, so its tolerances are wider and")
        print("    it is held to a weaker standard than the others.")
        for r in runs:
            if r["nframes"] < max(frames):
                print("      SMASS %-6g %d frames, %.0f%% of the longest"
                      % (r["smass"], r["nframes"],
                         100.0 * r["nframes"] / max(frames)))
    for r in runs:
        if r["nsteps"] and abs(r["nframes"] - r["nsteps"]) > FRAME_MISMATCH * r["nsteps"]:
            print("  ! SMASS %g: %d ionic steps in the OUTCAR against %d "
                  "trajectory frames" % (r["smass"], r["nsteps"], r["nframes"]))

    print("NVE reference          = %s at %.1f K"
          % (nve["directory"], nve["t_avg"]))
    t_gap = abs(nve["t_avg"] - runs[0]["t_avg"]) / max(runs[0]["t_avg"], 1.0)
    if t_gap > 0.10:
        print("  ! the reference is %.0f%% away in temperature from the NVT"
              % (100 * t_gap))
        print("    runs. Vibrational frequencies shift with temperature, so")
        print("    part of the VDOS difference below is a temperature effect")
        print("    rather than thermostat distortion.")

    # A reference much shorter than the runs it judges resolves the spectrum
    #   less finely than they do, and the difference between a smooth envelope
    #   and a resolved line spectrum is large however well the modes agree.
    ref_frames = nve.get("nframes")
    if ref_frames and ref_frames < REF_SHORT_FACTOR * max(frames):
        print("  ! the reference is %d frames against %d in the longest NVT"
              % (ref_frames, max(frames)))
        print("    run. It resolves the spectrum less finely than the runs it")
        print("    is judging, so part of the VDOS difference below is a")
        print("    resolution mismatch rather than thermostat distortion, and")
        print("    the noise floor from splitting it is correspondingly low.")

    print("VDOS compared up to    = %g cm^-1" % fmax_cm1)
    if st is not None:
        print("VDOS noise floor       = %.3f   (NVE split-half)" % st["floor"])
        print("VDOS threshold         = %.3f   (%.1f x floor)"
              % (vdos_threshold, VDOS_FLOOR_FACTOR))
    else:
        print("VDOS threshold         = %.3f   (fixed; self-test unavailable)"
              % vdos_threshold)
    print("-------------------------------------------\n")

    print("--------------------------- SMASS Diagnostics ---------------------------")
    header = ("%-8s %10s %9s %14s %14s %9s %8s  %s"
              % ("SMASS", "VDOS diff", "T_avg/K", "sig_o/sig_e", "skew",
                 "slope t", "tau_C/fs", "verdict"))
    print(header)
    print("-" * len(header))
    for r in runs:
        band = "%.2f+-%.2f" % (r["sigma_ratio"], r["sigma_band"])
        sk = "%.2f+-%.2f" % (r["skew_obs"], r["skew_band"])
        print("%-8g %10.3f %9.1f %14s %14s %9.1f %8.1f  %s"
              % (r["smass"], r["vdos_diff"], r["t_avg"], band, sk,
                 r["slope_t"], r["tau_c_fs"], r["verdict"]))
    print("-" * len(header))
    print("expected skewness for a canonical distribution = %.2f"
          % runs[0]["skew_exp"])
    print("\ncriteria, applied in this order:")
    print("  1. VDOS difference from the NVE reference < %.3f" % vdos_threshold)
    print("  2. mean temperature within %.0f%% of the target" % (100 * T_TOLERANCE))
    print("  3. sigma_obs / sigma_exp within %.0f sampling sigmas of unity,"
          % SIGMA_NSIGMA)
    print("     with N_eff >= %d and a floor of %.0f%%, and skewness within"
          % (N_EFF_MIN, 100 * SIGMA_FLOOR))
    print("     %.0f sampling sigmas of sqrt(8/g)" % SIGMA_NSIGMA)
    print("  4. corrected |slope| / SE(slope) < %.1f for the thermostat energy"
          % SLOPE_TSTAT)
    print("-------------------------------------------------------------------------\n")

    good = [r for r in runs if r["verdict"] == "acceptable"]
    if not good:
        best = min(runs, key=lambda r: r["vdos_diff"])
        print("RESULT: no tested SMASS satisfies every criterion.\n"
              "        The smallest departure from the NVE dynamics is at\n"
              "        SMASS = %g, with a VDOS difference of %.3f.\n"
              % (best["smass"], best["vdos_diff"]))
    elif len(good) == 1:
        print("RESULT: SMASS = %g satisfies every criterion.\n"
              % good[0]["smass"])
    else:
        lo = min(r["smass"] for r in good)
        hi = max(r["smass"] for r in good)
        print("RESULT: SMASS between %g and %g satisfies every criterion."
              % (lo, hi))
        spread = (max(r["vdos_diff"] for r in good)
                  - min(r["vdos_diff"] for r in good))
        if st is not None and st["floor"] >= 0.5 * spread:
            # The VDOS differences are all at the noise floor, so ordering
            #   them would rank measurement noise. Say so instead.
            print("        The VDOS differences within that range are smaller\n"
                  "        than the noise floor, so they do not rank these\n"
                  "        runs. Any value in the range is acceptable.\n")
        else:
            best = min(good, key=lambda r: r["vdos_diff"])
            print("        Within that range the closest match to the NVE\n"
                  "        dynamics is SMASS = %g.\n" % best["smass"])


def write_csv(path, runs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["directory", "smass", "potim_fs", "natoms", "nsteps",
                    "n_traj_frames",
                    "vdos_difference", "t_target_K", "t_avg_K", "t_error",
                    "tau_c_steps", "tau_c_fs", "n_eff", "sigma_obs_K",
                    "sigma_exp_K", "sigma_ratio", "sigma_band",
                    "skew_obs", "skew_exp", "skew_band",
                    "slope_eV_per_ps", "slope_t", "tau_E_steps",
                    "verdict"])
        for r in runs:
            w.writerow([r["directory"], "%g" % r["smass"], "%g" % r["potim"],
                        r["natoms"], r["nsteps"], r["nframes"],
                        "%.4f" % r["vdos_diff"],
                        "" if not r["target_t"] else "%.1f" % r["target_t"],
                        "%.2f" % r["t_avg"], "%.4f" % r["t_error"],
                        "" if r["tau_c"] is None else r["tau_c"],
                        "%.2f" % r["tau_c_fs"], "%.2f" % r["n_eff"],
                        "%.3f" % r["sigma_obs"], "%.3f" % r["sigma_exp"],
                        "%.4f" % r["sigma_ratio"], "%.4f" % r["sigma_band"],
                        "%.4f" % r["skew_obs"], "%.4f" % r["skew_exp"],
                        "%.4f" % r["skew_band"],
                        "%.4e" % r["slope_e"], "%.3f" % r["slope_t"],
                        "" if r["tau_e"] is None else r["tau_e"],
                        r["verdict"]])


def write_tex_table(path, runs, label, nve):
    tag = re.sub(r"[^A-Za-z0-9]", "", label)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{rrrrrrrl}",
        r"\hline",
        (r"SMASS & $\Delta_{\mathrm{VDOS}}$ & $\bar{T}$ (K) & "
         r"$\sigma_{\mathrm{obs}}/\sigma_{\mathrm{exp}}$ & Skewness & "
         r"$|s|/\mathrm{SE}(s)$ & $\tau_C$ (fs) & Verdict \\"),
        r"\hline",
    ]
    for r in runs:
        lines.append(r"%g & %.3f & %.1f & $%.2f \pm %.2f$ & $%.2f \pm %.2f$ "
                     r"& %.1f & %.1f & %s \\"
                     % (r["smass"], r["vdos_diff"], r["t_avg"],
                        r["sigma_ratio"], r["sigma_band"],
                        r["skew_obs"], r["skew_band"], r["slope_t"],
                        r["tau_c_fs"], r["verdict"]))
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"}",
        (r"\caption{Thermostat diagnostics for %s. $\Delta_{\mathrm{VDOS}}$ is "
         r"the integrated difference between the vibrational density of states "
         r"of each run and that of an NVE run of the same system, both "
         r"normalised to unit area, judged against the difference between "
         r"the two halves of the NVE run itself. Temperatures use "
         r"$g = 3N-3$ degrees of freedom. The tolerance on "
         r"$\sigma_{\mathrm{obs}}/\sigma_{\mathrm{exp}}$ is three times the "
         r"sampling uncertainty of that run. The skewness of a canonical "
         r"temperature distribution is $\sqrt{8/g}$, with a tolerance of "
         r"three sampling standard errors. The quantity $|s|/\mathrm{SE}(s)$ "
         r"is the significance of the fitted drift in the thermostat energy, "
         r"corrected for the autocorrelation of the residuals. $\tau_C$ is the "
         r"temperature correlation time.}" % label),
        r"\label{tab:smass_%s}" % tag,
        r"\end{table}",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def write_tex_figure(path, figname, label):
    tag = re.sub(r"[^A-Za-z0-9]", "", label)
    body = [
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\includegraphics[width=\textwidth]{%s}" % figname,
        (r"\caption{Thermostat mass validation for %s. Panel (a) shows the "
         r"vibrational density of states for each SMASS value with the NVE "
         r"reference in black. Panel (b) shows the integrated departure from "
         r"that reference. Panel (c) shows the ratio of observed to expected "
         r"temperature fluctuation, with the acceptance band set by the "
         r"sampling uncertainty of each run.}" % label),
        r"\label{fig:smass_%s}" % tag,
        r"\end{figure}",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(body))


##################################################
# Figures
##################################################


def make_sweep_figure(runs, label, stem, dpi, nve, fmax_cm1,
                      vdos_threshold, st):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(runs)))

    # (a) VDOS with the NVE reference
    ax = axes[0]
    for r, c in zip(runs, colors):
        y = r["vdos"] / r["vdos"].max()
        ax.plot(r["wavenumbers"], y, color=c, linewidth=1.0,
                label="SMASS = %g" % r["smass"])
    y = nve["vdos"] / nve["vdos"].max()
    ax.plot(nve["wavenumbers"], y, color="k", linewidth=1.8, linestyle="--",
            label="NVE reference")
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Normalized VDOS")
    ax.set_title("(a) Vibrational density of states")
    ax.set_xlim(0, fmax_cm1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

    # (b) VDOS departure
    ax = axes[1]
    sm = np.array([r["smass"] for r in runs])
    dv = np.array([r["vdos_diff"] for r in runs])
    ax.axhspan(0, vdos_threshold, color="tab:green", alpha=0.15,
               label="acceptable")
    if st is not None:
        ax.axhline(st["floor"], color="0.4", linestyle=":", linewidth=1.2,
                   label="NVE noise floor")
    ax.plot(sm, dv, marker="D", color="tab:orange", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xticks(sm)
    ax.set_xticklabels(["%g" % s for s in sm])
    ax.minorticks_off()
    ax.set_xlabel("SMASS")
    ax.set_ylabel(r"$\Delta_{\mathrm{VDOS}}$ from NVE")
    ax.set_title("(b) Departure from natural dynamics")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

    # (c) fluctuation ratio
    ax = axes[2]
    ratio = np.array([r["sigma_ratio"] for r in runs])
    band = np.array([r["sigma_band"] for r in runs])
    ax.fill_between(sm, 1.0 - band, 1.0 + band, color="tab:green", alpha=0.15,
                    label="acceptance band")
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1.0)
    ax.plot(sm, ratio, marker="o", color="tab:blue", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xticks(sm)
    ax.set_xticklabels(["%g" % s for s in sm])
    ax.minorticks_off()
    ax.set_xlabel("SMASS")
    ax.set_ylabel(r"$\sigma_{\mathrm{obs}} / \sigma_{\mathrm{exp}}$")
    ax.set_title("(c) Fluctuation magnitude")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("SMASS validation: %s" % label, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (stem, ext), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_diagnostic_figure(runs, label, stem, dpi, nve):
    nrows = len(runs)
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 3.0 * nrows), squeeze=False)

    for i, r in enumerate(runs):
        steps = np.arange(1, r["nsteps"] + 1)
        i0 = r["nsteps"] - r["n_steady"]

        ax = axes[i][0]
        ax.plot(steps, r["temps"], color="tab:red", linewidth=0.6)
        ax.axhline(r["t_avg"], color="k", linestyle="--", linewidth=1.0)
        ax.fill_between(steps[i0:], r["t_avg"] - r["sigma_exp"],
                        r["t_avg"] + r["sigma_exp"],
                        color="tab:green", alpha=0.2)
        ax.set_ylabel("SMASS = %g\nT (K)" % r["smass"])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Temperature")
        if i == nrows - 1:
            ax.set_xlabel("Ionic step")

        ax = axes[i][1]
        ax.hist(r["temps_steady"], bins=60, density=True, alpha=0.6,
                color="tab:red")
        # Canonical reference: chi-squared with g degrees of freedom,
        #   scaled so its mean is T_avg.
        x = np.linspace(r["temps_steady"].min(), r["temps_steady"].max(), 300)
        g = r["ndof"]
        ax.plot(x, gamma.pdf(x, a=g / 2.0, scale=2.0 * r["t_avg"] / g),
                "k--", lw=1.5)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Steady-state distribution")
        if i == nrows - 1:
            ax.set_xlabel("Temperature (K)")

        ax = axes[i][2]
        ax.plot(r["wavenumbers"], r["vdos"] / r["vdos"].max(),
                color="tab:blue", linewidth=1.0, label="NVT")
        ax.plot(nve["wavenumbers"], nve["vdos"] / nve["vdos"].max(),
                color="k", linewidth=1.2, linestyle="--", label="NVE")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("VDOS against NVE")
            ax.legend(fontsize=8, frameon=False)
        if i == nrows - 1:
            ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")

    fig.suptitle("SMASS diagnostics: %s" % label, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig("%s_diagnostics.png" % stem, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


##################################################
# Main
##################################################


def main():
    args = parse_args()

    try:
        nve = load_nve_reference(args.nve, args.drop, args.fmax)
        log.info("NVE reference loaded from %s", args.nve)
    except (FileNotFoundError, ValueError) as exc:
        log.error("could not load the NVE reference: %s", exc)
        raise SystemExit(1)

    if args.sweep:
        dirs = []
        for pattern in args.sweep:
            hits = sorted(glob.glob(pattern))
            dirs.extend(h for h in (hits or [pattern]) if os.path.isdir(h))
        dirs = sorted(set(dirs))
        if not dirs:
            log.error("no directories matched %s", " ".join(args.sweep))
            raise SystemExit(1)
    else:
        dirs = ["."]

    # The self-test comes first because criterion 1 is judged against it.
    st = None
    try:
        st = selftest(args.nve, args.fmax)
        vdos_threshold = VDOS_FLOOR_FACTOR * st["floor"]
    except (FileNotFoundError, ValueError) as exc:
        log.warning("self-test failed (%s); falling back to a fixed VDOS "
                    "threshold of %.2f", exc, VDOS_DIFF_MAX)
        vdos_threshold = VDOS_DIFF_MAX

    runs = []
    for d in dirs:
        try:
            runs.append(analyse_run(d, args.drop, nve, args.target_t, args.fmax,
                                    vdos_threshold))
        except (FileNotFoundError, ValueError) as exc:
            log.warning("skipping %s: %s", d, exc)

    if not runs:
        log.error("no run could be analysed")
        raise SystemExit(1)

    runs.sort(key=lambda r: r["smass"])

    if nve["natoms"] != runs[0]["natoms"]:
        log.warning("the NVE reference has %d atoms but the NVT runs have %d; "
                    "the comparison assumes the same system",
                    nve["natoms"], runs[0]["natoms"])

    print_report(runs, args.drop, nve, args.fmax, vdos_threshold, st)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, "SMASS_%s" % args.label)

    if st is not None:
        print_selftest(st, runs)
        make_selftest_figure(st, stem, args.dpi, args.fmax)

    if len(runs) > 1:
        make_sweep_figure(runs, args.label, stem, args.dpi, nve, args.fmax,
                          vdos_threshold, st)
    make_diagnostic_figure(runs, args.label, stem, args.dpi, nve)
    write_csv(stem + ".csv", runs)
    write_tex_table(stem + ".tex", runs, args.label, nve)

    written = [stem + ".csv", stem + ".tex", stem + "_diagnostics.png"]
    if st is not None:
        written.append(stem + "_selftest.png")
    if len(runs) > 1:
        written = [stem + ".png", stem + ".pdf"] + written
    if args.appendix:
        figtex = stem + "_fig.tex"
        write_tex_figure(figtex, os.path.basename(stem) + ".pdf", args.label)
        written.append(figtex)

    print("Wrote:")
    for w in written:
        print("  %s" % w)
    print()


if __name__ == "__main__":
    main()
