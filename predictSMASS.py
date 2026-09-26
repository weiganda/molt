#!/usr/bin/env python3
"""
PROGRAM: predictSMASS.py
PURPOSE: Turns an NVE reference trajectory into a SMASS value and a sweep
         around it, for a material whose vibrational spectrum is not known in
         advance and with no NVT runs required.

         Nose (J. Chem. Phys. 81, 511, 1984) gives the thermostat mass that
         makes the temperature oscillate with a chosen period t0, and the
         condition on t0 is that it sit among the vibrational periods of the
         material so that thermostat and lattice exchange energy. A thermostat
         much slower than every mode does not thermalise the system: it cycles
         energy in and out on its own timescale, producing large periodic
         temperature swings and a two-humped temperature histogram.

         The one material-dependent input is therefore the vibrational
         timescale, and that is measured here from the velocity spectrum of
         the NVE run rather than assumed or looked up. The rest of the formula
         needs only the degrees of freedom, the temperature, and the cell.

         Probe NVT runs are optional. When given, they are checked against the
         prediction rather than used to build it.

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Conceptual design, purpose, and validation by Alysse Weigand.
                 Code implementation and structure assisted by Claude (Anthropic),
                 September 2026.
LAST MODIFIED:   September 2026

USAGE:
    python predictSMASS.py --nve DIR [--probe DIR ...] [-n 5] [--fmax 3000]

EXAMPLE:
    # Stage 1: characterise a new material, get the probe recipe
    python predictSMASS.py --nve NVE_300

    # Stage 2 (optional): check existing NVT runs against the prediction
    python predictSMASS.py --nve NVE_300 --probe smass_0.1 smass_1.0

    # Any existing sweep directories serve as probes
    python predictSMASS.py --nve NVE_300 --probe smass_*

NOTES:
    Runs standalone. The OUTCAR and VDOS routines are copied from
    validate_SMASS.py rather than imported, so that this script can be used on
    its own before any NVT runs exist. The two copies must stay in step: if one
    is changed, change the other, or the numbers the two scripts report will
    not be comparable.

    The noise baseline is fitted over a band assumed to hold no vibrations,
    which by default is the top quarter of the --fmax window. That assumption
    holds for oxides and other inorganic solids, whose highest modes are well
    below 1500 cm^-1. It fails for anything hydroxylated, hydrated or organic:
    O-H stretches near 3600, N-H near 3300 and C-H near 3000 cm^-1 fall inside
    the default band, and fitting a baseline through a real peak raises the
    detection threshold across the whole spectrum, hiding genuine modes below
    it. For those systems set --noise-band above the highest stretch, for
    example --noise-band 5000 8000. The script checks the band for features
    and says so when it finds one, but the check is a warning rather than a
    guarantee: a broad band of modes filling the whole fitting window can be
    absorbed into the fitted slope without any single bin standing out.

    September 2026 fixes, shared with validate_SMASS.py: the VDOS is now the
    real part of the transform rather than its magnitude, and vasprun.xml is
    read for MD-step structures only. Both routines must be changed in
    validate_SMASS.py as well to keep the two scripts comparable.

    If no mode clears the noise baseline, the script reports what it found
    (frame count, per-step atomic displacement, fitted baseline slope, and the
    bin that came closest to the threshold) before exiting, so that a
    trajectory-reading problem can be told apart from a baseline problem.
"""

##################################################
# Imports
##################################################

import os
import re
import argparse
import numpy as np

##################################################
# Criteria
##################################################

ACF_THRESHOLD = 0.1     # temperature correlation considered lost below this
N_EFF_MIN = 30          # gives ~13% precision on sigma_obs
PEAK_FRACTION = 0.05    # weight above this fraction of the de-noised max is a mode
NOISE_BAND = 0.75       # noise floor estimated above this fraction of fmax
NOISE_MARGIN = 2.0      # a mode must exceed the fitted noise baseline by this
TAU_TARGET_LO = 1.0     # target Nose period, in fastest-mode periods
TAU_TARGET_HI = 2.0
CLIP_MARGIN = 0.95      # fastest mode above this fraction of fmax is clipped
RING_FRACTION = 0.50    # power in one peak above this is a ringing trajectory
PARTIAL_FRACTION = 0.20

CM1_PER_HZ = 3.33564e-11
C_CM_PER_S = 2.99792458e10
KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K
KB_SI = 1.380649e-23    # Boltzmann constant, J/K
AMU_KG = 1.66053906660e-27

##################################################
# Trajectory routines
#
# These are copied verbatim from validate_SMASS.py rather than imported, so
# that this script runs on its own. If either copy is changed, change both:
# the two scripts must describe a trajectory identically or their numbers
# cannot be compared.
##################################################

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

def read_trajectory(directory):
    """
    Read the trajectory from vasprun.xml, or from XDATCAR if vasprun.xml is
    absent, and return Cartesian positions with shape (steps, atoms, 3), in
    Angstrom.

    Positions are fractional. Atoms that cross a periodic boundary jump by
    close to a full cell length, which would dominate any derivative taken
    from the raw values, so the fractional coordinates are unwrapped before
    being converted through the lattice vectors.
    """
    import xml.etree.ElementTree as ET

    path = os.path.join(directory, "vasprun.xml")
    xdat = os.path.join(directory, "XDATCAR")

    if os.path.isfile(path):
        lattice = None
        frames = []
        # Only the structures inside each <calculation> block are MD steps.
        #   vasprun.xml also holds named structures (initialpos, finalpos,
        #   primitive_cell) with their own positions arrays; reading those as
        #   frames puts extra, misaligned configurations at the ends of the
        #   series and shows up as cell-sized jumps between adjacent frames.
        in_calc = 0
        struct_name = []
        # iterparse keeps memory flat: a long MD run makes a large file.
        for event, elem in ET.iterparse(path, events=("start", "end")):
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
    while i < len(lines):
        if lines[i].lstrip().lower().startswith("direct configuration"):
            block = lines[i + 1:i + 1 + natoms]
            if len(block) < natoms:     # truncated final frame
                break
            frames.append([[float(x) for x in l.split()[:3]] for l in block])
            i += natoms + 1
        else:
            i += 1

    return lattice, np.asarray(frames, dtype=float)


def velocities_from_positions(positions, potim_fs):
    """
    Central-difference velocities in Angstrom per femtosecond.

    The first and last frames are dropped, since a central difference is not
    defined at the ends of the series.
    """
    return (positions[2:] - positions[:-2]) / (2.0 * potim_fs)


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
    earlier version did, folded that tail into the spectrum; the noise fit
    then read it as a baseline of slope -1 which, extrapolated down to the
    vibrational range, sat over the real modes and hid them (TiO2 and SrTiO3
    both failed this way). The real part is the cosine transform proper.
    """
    n = vacf.size
    window = np.hanning(2 * n)[n:]
    tapered = vacf * window

    spectrum = np.fft.rfft(tapered, n=4 * n).real
    freqs_hz = np.fft.rfftfreq(4 * n, d=potim_fs * 1e-15)
    wavenumbers = freqs_hz * CM1_PER_HZ

    mask = wavenumbers <= fmax_cm1
    return wavenumbers[mask], spectrum[mask]

##################################################
# Helpers
##################################################

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    p = argparse.ArgumentParser(
        description="Propose a SMASS sweep range from an NVE reference run.")

    p.add_argument("--nve", default=".", metavar="DIR",
                   help="Directory holding the NVE reference. (default: .)")
    p.add_argument("--probe", nargs="+", default=None, metavar="DIR",
                   help="Optional NVT directories, checked against the "
                        "prediction. Not needed to produce one: the SMASS is "
                        "computed analytically from the NVE run alone.")
    p.add_argument("-T", "--temperature", type=float, default=None,
                   metavar="K",
                   help="Temperature the production NVT will run at. The "
                        "thermostat mass depends on it. (default: the mean of "
                        "the NVE reference)")
    p.add_argument("-n", type=int, default=5, metavar="N",
                   help="Number of SMASS values in the proposed sweep. "
                        "(default: 5)")
    p.add_argument("--drop", type=float, default=0.5, metavar="FRAC",
                   help="Fraction of each trajectory discarded as "
                        "equilibration. (default: 0.5)")
    p.add_argument("--fmax", type=float, default=4000.0, metavar="CM1",
                   help="Upper limit of the VDOS window. Keep this well above "
                        "any expected mode: the top quarter of the window is "
                        "used to estimate the noise floor, so it needs to hold "
                        "no real vibrations. (default: 4000)")
    p.add_argument("--noise-band", nargs=2, type=float, default=None,
                   dest="noise_band", metavar=("LO", "HI"),
                   help="Wavenumber range used to fit the noise baseline. It "
                        "must contain no real vibrations. Default is the top "
                        "quarter of the --fmax window, which is safe for "
                        "oxides. Raise it for anything hydroxylated, hydrated "
                        "or organic: O-H, N-H and C-H stretches sit between "
                        "2800 and 3700 cm^-1 and would be fitted as noise.")
    p.add_argument("--target", type=float, default=None, metavar="FS",
                   help="Override the target correlation time in fs instead of "
                        "deriving it from the fastest mode.")

    args = p.parse_args()

    if not 0.0 <= args.drop < 1.0:
        p.error("--drop must be in [0, 1).")
    if args.n < 2:
        p.error("-n must be at least 2.")
    if args.noise_band:
        lo, hi = args.noise_band
        if lo <= 0 or hi <= lo:
            p.error("--noise-band needs 0 < LO < HI.")
    return args


def get_potim(directory: str):
    """POTIM from the INCAR, falling back to the OUTCAR."""
    potim = read_tag(os.path.join(directory, "INCAR"), "POTIM")
    if potim is None:
        potim = read_tag(os.path.join(directory, "OUTCAR"), "POTIM")
    return potim


def read_lattice(directory: str) -> np.ndarray:
    """
    Lattice vectors in Angstrom, from vasprun.xml.

    read_trajectory parses these too but does not return them, and it is kept
    byte-identical to the copy in validate_SMASS.py, so the basis is read
    separately here. Only the first varray is needed, so the parse stops early.
    """
    import xml.etree.ElementTree as ET

    path = os.path.join(directory, "vasprun.xml")
    if not os.path.isfile(path):
        xdat = os.path.join(directory, "XDATCAR")
        if os.path.isfile(xdat):
            return read_xdatcar(xdat, lattice_only=True)[0]
        raise FileNotFoundError("no vasprun.xml or XDATCAR in %s" % directory)

    for _event, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "varray" and elem.get("name") == "basis":
            return np.array([[float(x) for x in v.text.split()]
                             for v in elem.findall("v")], dtype=float)
    raise ValueError("no lattice vectors in %s" % path)


def nose_smass(t0_fs: float, ndof: int, temperature: float,
               l_ang: float) -> float:
    """
    The VASP SMASS that gives a Nose thermostat oscillation period of t0.

    From Nose, J. Chem. Phys. 81, 511 (1984): the thermostat mass that makes
    the temperature oscillate with period t0 is

        Q = (t0 / 2*pi)**2 * 2 * N_dof * k_B * T

    which is an energy times a time squared. VASP integrates the equations of
    motion in fractional coordinates, so its SMASS is that mass expressed in
    atomic mass units times the square of the first lattice vector rather than
    in SI. The conversion is where the cell enters, and it is the reason SMASS
    is not transferable between materials or even between cell sizes of the
    same material.

    The period matters because it has to sit among the vibrational periods of
    the material. A thermostat far slower than every mode does not thermalise
    the system; it exchanges energy with it on its own long cycle, which shows
    up as large periodic temperature swings and a temperature histogram with
    two humps rather than one.
    """
    q_si = (t0_fs * 1e-15 / (2.0 * np.pi)) ** 2 * 2 * ndof * KB_SI * temperature
    return q_si / AMU_KG / (l_ang * 1e-10) ** 2


def nose_period(smass: float, ndof: int, temperature: float,
                l_ang: float) -> float:
    """Inverse of nose_smass: the oscillation period a given SMASS produces."""
    q_si = smass * AMU_KG * (l_ang * 1e-10) ** 2
    return 2.0 * np.pi * np.sqrt(q_si / (2 * ndof * KB_SI * temperature)) * 1e15


def nve_sigma_expected(t_mean: float, natoms: int) -> float:
    """
    Expected temperature fluctuation for a microcanonical trajectory.

    The canonical result, sigma/T = sqrt(2 / g) with g = 3N - 3, is the wrong
    comparison for an NVE run. At fixed total energy a rise in kinetic energy
    must be paid for out of potential energy, which suppresses the fluctuation
    by a factor sqrt(1 - g k_B / 2 C_V). For a harmonic solid C_V = g k_B, so
    the factor is sqrt(1/2) and the expected fluctuation is smaller by about
    0.707.
    Anharmonicity raises C_V and pushes the ratio back toward unity, so a
    measured value slightly above 0.707 is the healthy case.
    """
    canonical = t_mean * np.sqrt(2.0 / degrees_of_freedom(natoms))
    return canonical * np.sqrt(0.5)


def analyse_nve(directory: str, drop: float, fmax: float, band=None) -> dict:
    """Temperature statistics, VDOS landmarks, and the dephasing test."""
    data = read_outcar(directory)
    potim = get_potim(directory)
    if potim is None:
        raise SystemExit("could not find POTIM for %s" % directory)

    temps = data["temps"]
    i0 = int(drop * temps.size)
    steady = temps[i0:]

    lo, hi = band if band else (NOISE_BAND * fmax, fmax)

    lattice = read_lattice(directory)
    l_first = float(np.linalg.norm(lattice[0]))

    positions = read_trajectory(directory)
    vel = velocities_from_positions(positions, potim)
    iv = int(drop * vel.shape[0])
    vacf = velocity_acf(vel[iv:])
    # The transform runs to whichever is higher, so that a noise band placed
    #   above the detection window is still available to fit.
    wn, vdos = vdos_from_vacf(vacf, potim, max(fmax, hi))

    # Noise baseline. Positions in vasprun.xml are stored to finite precision,
    #   so the central-difference velocities carry a quantisation error that
    #   appears as a shelf under the whole spectrum. That shelf is not flat: it
    #   falls off roughly as a power of the wavenumber, so a constant floor
    #   taken from the top of the window badly underestimates the noise in the
    #   middle of it, and a "highest bin above threshold" test then locks onto
    #   noise hundreds of wavenumbers above the last real mode. The shelf is
    #   fitted on a log-log line over a band that holds no vibrations, and
    #   extrapolated downward; a mode has to clear that curve by a margin.
    tail = (wn >= lo) & (wn <= hi)
    feature = None
    slope = float("nan")    # stays NaN if the fallback baseline is used
    if tail.sum() > 20 and np.all(vdos[tail] > 0):
        slope, intercept = np.polyfit(np.log(wn[tail]), np.log(vdos[tail]), 1)
        # The wavenumber axis starts at zero and the fitted slope is negative,
        #   so the baseline is evaluated only where it is defined. The zero bin
        #   is given an infinite baseline: it carries the mean of the velocity
        #   signal, never a vibration.
        baseline = np.full_like(vdos, np.inf)
        positive = wn > 0
        baseline[positive] = np.exp(intercept) * wn[positive] ** slope
        contrast = float(np.median(vdos[tail]) / vdos.max())

        # Flatness check. The fit is only a noise estimate if the band it was
        #   taken from is featureless. A real peak inside the band drags the
        #   fitted curve up, which raises the threshold everywhere below it and
        #   silently hides genuine modes. Rather than trust the band, measure
        #   how far the worst bin in it rises above the fit.
        excess = vdos[tail] / baseline[tail]
        worst = int(np.argmax(excess))
        if excess[worst] > NOISE_MARGIN:
            feature = (float(wn[tail][worst]), float(excess[worst]))
    else:
        baseline = np.full_like(vdos, np.median(vdos[tail]) if tail.sum() else 0.0)
        contrast = float("nan")

    # Fastest mode: the highest wavenumber still carrying real weight above the
    #   fitted noise. This is not the dominant peak. The dominant peak says
    #   where the energy happens to sit right now; the highest mode sets the
    #   shortest period the thermostat has to avoid competing with.
    detect = wn <= fmax
    above = wn[detect & (vdos > NOISE_MARGIN * baseline)]
    w_fast = float(above.max()) if above.size else float("nan")
    w_peak = float(wn[int(np.argmax(vdos[detect]))])

    # Diagnostics, reported only if no mode clears the baseline. They separate
    #   a trajectory-reading problem (wrong frame count, zero or huge atomic
    #   steps) from a baseline problem (a fitted curve that sits above the real
    #   modes once extrapolated down to them).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(detect & (wn > 0), vdos / baseline, 0.0)
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0)
    ibest = int(np.argmax(ratio))
    step = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    diag = {
        "nframes": positions.shape[0],
        "step_mean": float(step.mean()),
        "step_max": float(step.max()),
        "vdos_max": float(vdos.max()),
        "n_tail": int(tail.sum()),
        "slope": float(slope),
        "best_ratio": float(ratio[ibest]),
        "best_wn": float(wn[ibest]),
    }

    # Dephasing, from the kinetic energy fluctuation rather than the VDOS.
    signal = steady - steady.mean()
    power = np.abs(np.fft.rfft(signal * np.hanning(signal.size))) ** 2
    freq = np.fft.rfftfreq(signal.size, d=potim * 1e-15)
    wn_t, power = freq[1:] * CM1_PER_HZ, power[1:]
    total = power.sum()
    ipk = int(np.argmax(power))
    plo, phi = max(0, ipk - 3), min(power.size, ipk + 4)

    return {
        "directory": directory,
        "natoms": data["natoms"],
        "ndof": 3 * data["natoms"] - 3,
        "l_first": l_first,
        "potim": potim,
        "nsteps": temps.size,
        "t_mean": float(steady.mean()),
        "t_std": float(steady.std(ddof=1)),
        "sigma_exp": nve_sigma_expected(float(steady.mean()), data["natoms"]),
        "w_fast": w_fast,
        "w_peak": w_peak,
        "noise": contrast,
        "band": (lo, hi),
        "feature": feature,
        "ke_freq": float(wn_t[ipk]),
        "peak_fraction": float(power[plo:phi].sum() / total),
        "participation": float(total ** 2 / np.sum(power ** 2)),
        "resolution": float(1.0 / (steady.size * potim * 1e-15 * C_CM_PER_S)),
        "diag": diag,
    }


def tau_c(directory: str, drop: float):
    """Temperature correlation time in fs, and the SMASS that produced it."""
    data = read_outcar(directory)
    potim = get_potim(directory)
    smass = read_tag(os.path.join(directory, "INCAR"), "SMASS")
    if smass is None:
        smass = read_tag(os.path.join(directory, "OUTCAR"), "SMASS")
    if potim is None or smass is None:
        return None

    temps = data["temps"]
    steady = temps[int(drop * temps.size):]
    acf = autocorrelation(steady)
    below = np.where(np.abs(acf) < ACF_THRESHOLD)[0]
    if not below.size or below[0] == 0:
        return None

    lag = int(below[0])
    return {"directory": directory, "smass": float(smass),
            "tau": lag * potim, "n_steady": steady.size,
            "n_eff": steady.size / lag, "potim": potim}


def log_grid(vmin: float, vmax: float, n: int):
    """
    Log-spaced values, rounded for readable directory names.

    Rounding to a fixed number of decimals collapses distinct values into each
    other once SMASS falls below about 0.1, which it does for any stiff
    material, so the rounding is to two significant figures instead.
    """
    ratio = (vmax / vmin) ** (1.0 / (n - 1))
    out = []
    for i in range(n):
        v = float("%.2g" % (vmin * ratio ** i))
        if v > 0 and v not in out:
            out.append(v)
    return out

##################################################
# Reporting
##################################################

def report_nve(nve: dict, target_lo: float, target_hi: float) -> None:
    print("\n---------- NVE reference ----------")
    print("  Directory            = %s" % nve["directory"])
    print("  NIONS                = %d" % nve["natoms"])
    print("  POTIM                = %g fs" % nve["potim"])
    print("  Steps                = %d  (%.3f ps)"
          % (nve["nsteps"], nve["nsteps"] * nve["potim"] / 1000.0))
    print("  VDOS resolution      = %.0f cm^-1" % nve["resolution"])

    print("\n---------- Temperature ----------")
    print("  Mean, steady state   = %7.1f K" % nve["t_mean"])
    print("  Std dev              = %7.1f K" % nve["t_std"])
    print("  Expected (NVE)       = %7.1f K   sqrt(2/g) x sqrt(1/2)"
          % nve["sigma_exp"])
    print("  Observed / expected  = %7.2f" % (nve["t_std"] / nve["sigma_exp"]))
    print("      A harmonic solid gives 1.00 here; anharmonicity raises it.")

    print("\n---------- Vibrational spectrum ----------")
    print("  Fastest mode         = %.0f cm^-1   period %.1f fs"
          % (nve["w_fast"], 1.0e15 / (nve["w_fast"] * C_CM_PER_S)))
    print("  Dominant peak        = %.0f cm^-1" % nve["w_peak"])
    print("  Noise band           = %.0f to %.0f cm^-1" % nve["band"])
    print("  Noise floor          = %.1f%% of the VDOS maximum" % (100 * nve["noise"]))
    if nve["feature"]:
        wf, ex = nve["feature"]
        print("      A feature sits at %.0f cm^-1, %.1fx above the fitted"
              % (wf, ex))
        print("      baseline, inside the band used to estimate the noise.")
        print("      The band is supposed to be featureless. A real mode there")
        print("      drags the baseline up and hides genuine modes below it.")
        print("      Move the band with --noise-band, e.g. above any O-H, N-H")
        print("      or C-H stretch for a hydroxylated or organic system.")
    if nve["noise"] > 0.15:
        print("      That floor is high. Velocities come from differencing")
        print("      finite-precision positions, so a weak spectrum sits close")
        print("      to the quantisation noise. Treat the fastest mode as a")
        print("      lower bound and check it against the known chemistry.")
    print("  Target Nose period   = %.1f to %.1f fs" % (target_lo, target_hi))
    print("      Nose's condition is that the thermostat oscillate on the same")
    print("      timescale as the vibrations, so that the two exchange energy.")

    print("\n---------- Dephasing ----------")
    print("  KE fluctuation peak  = %.0f cm^-1  (mode at %.0f cm^-1)"
          % (nve["ke_freq"], nve["ke_freq"] / 2.0))
    print("  Power in that peak   = %.1f%%" % (100 * nve["peak_fraction"]))
    print("  Active frequencies   = %.0f" % nve["participation"])

    if nve["peak_fraction"] > RING_FRACTION:
        print("      RINGING. One mode holds most of the kinetic energy, so")
        print("      the high-frequency modes may be too weakly excited to")
        print("      show. The fastest mode above is probably underestimated.")
    elif nve["peak_fraction"] > PARTIAL_FRACTION:
        print("      PARTIALLY DEPHASED. Usable; note the bias in the write-up.")
    else:
        print("      DEPHASED. Energy is spread across modes.")


def report_no_mode(nve: dict) -> None:
    """
    What the spectrum looked like when no mode cleared the noise baseline.

    Read the lines in order. A frame count far below the OUTCAR step count
    means the trajectory was written every NBLOCK steps and the wavenumber axis
    is wrong. A mean atomic step near zero, or far larger than a few hundredths
    of an Angstrom at sub-femtosecond POTIM, means the positions were misread.
    If both look sensible, the problem is the baseline: a slope near -1 with
    the best bin between 1 and the margin at a few hundred wavenumbers means
    the fitted curve, extrapolated downward, sits over the real modes.
    """
    d = nve["diag"]
    print("\n---------- No mode found ----------")
    print("  No VDOS bin cleared %.1fx the fitted noise baseline."
          % NOISE_MARGIN)
    print("  Trajectory frames    = %d   (OUTCAR steps: %d)"
          % (d["nframes"], nve["nsteps"]))
    print("  Atom step, mean/max  = %.2e / %.2e A"
          % (d["step_mean"], d["step_max"]))
    print("  VDOS maximum         = %.3g" % d["vdos_max"])
    print("  Noise band           = %.0f to %.0f cm^-1, %d bins"
          % (nve["band"][0], nve["band"][1], d["n_tail"]))
    print("  Fitted log-log slope = %.2f" % d["slope"])
    if np.isnan(d["slope"]):
        print("      NaN: too few bins, or a non-positive bin, in the band;")
        print("      a constant median baseline was used instead.")
    print("  Best VDOS/baseline   = %.2f at %.0f cm^-1   (needs > %.1f)"
          % (d["best_ratio"], d["best_wn"], NOISE_MARGIN))
    if d["nframes"] < 0.9 * nve["nsteps"]:
        print("      Fewer frames than steps: check NBLOCK in the INCAR.")
    print()


def report_prediction(nve: dict, temperature: float, n: int,
                      target_lo: float, target_hi: float) -> None:
    """The analytic prediction. Needs no NVT runs at all."""
    ndof, l_first = nve["ndof"], nve["l_first"]

    print("\n---------- Predicted SMASS ----------")
    print("  Degrees of freedom   = %d        3N - 3" % ndof)
    print("  First cell vector    = %.4f A" % l_first)
    print("  Temperature          = %.0f K" % temperature)

    s_lo = nose_smass(target_lo, ndof, temperature, l_first)
    s_hi = nose_smass(target_hi, ndof, temperature, l_first)
    print("  SMASS for %5.1f fs    = %.4f" % (target_lo, s_lo))
    print("  SMASS for %5.1f fs    = %.4f" % (target_hi, s_hi))

    values = log_grid(s_lo / 2.0, s_hi * 2.0, n)
    print("\n---------- Proposed sweep ----------")
    print("  Values  = %s" % ", ".join("%g" % v for v in values))
    print("  Nose period = %s fs"
          % ", ".join("%.0f" % nose_period(v, ndof, temperature, l_first)
                      for v in values))
    print("      The sweep runs from half to twice the bracket, because the")
    print("      prediction fixes a timescale and not a tolerance.")

    # A production run has to be long enough to average over many thermostat
    #   cycles, and the slowest cycle in the sweep sets the requirement.
    slowest = nose_period(max(values), ndof, temperature, l_first)
    nsw = int(np.ceil(2 * N_EFF_MIN * slowest / nve["potim"] / 1000.0) * 1000)
    print("  Minimum NSW = %d   (%.2f ps, %d cycles of the slowest thermostat)"
          % (nsw, nsw * nve["potim"] / 1000.0, N_EFF_MIN))

    print("\n  nvtINCAR.py -npar 6 -potim %g --grid %g %g %d -nsw %d "
          "-TB %.0f -TE %.0f"
          % (nve["potim"], values[0], values[-1], len(values), nsw,
             temperature, temperature))


def report_consistency(probes: list, nve: dict, temperature: float) -> None:
    """
    Check measured behaviour against the analytic prediction.

    The prediction stands on its own, so this is not needed to choose a SMASS.
    What it does is test whether the formula describes these runs: the observed
    temperature correlation time should track the predicted Nose period, and a
    probe whose period is far longer than every vibration in the material is
    the one that will show large periodic swings and a two-humped histogram.
    """
    ndof, l_first = nve["ndof"], nve["l_first"]
    fast = 1.0e15 / (nve["w_fast"] * C_CM_PER_S)

    print("\n---------- Measured runs against the prediction ----------")
    print("  %-8s %10s %12s %10s" % ("SMASS", "tau_C/fs", "Nose t0/fs",
                                     "t0 / period"))
    for p_ in sorted(probes, key=lambda q: q["smass"]):
        t0 = nose_period(p_["smass"], ndof, temperature, l_first)
        ratio = t0 / fast
        note = ""
        if ratio > 3.0:
            note = "  thermostat far slower than any mode"
        elif p_["n_eff"] < N_EFF_MIN:
            note = "  below N_eff min"
        print("  %-8g %10.1f %12.1f %10.1f%s"
              % (p_["smass"], p_["tau"], t0, ratio, note))

    print("\n      tau_C is what the trajectory did; t0 is what the formula")
    print("      says the thermostat should do. The last column is the one to")
    print("      read: a ratio near 1 puts the thermostat among the")
    print("      vibrations, which is where Nose's condition wants it.")


##################################################
# Main
##################################################

def main() -> None:
    args = parse_args()

    nve = analyse_nve(args.nve, args.drop, args.fmax, args.noise_band)

    if np.isnan(nve["w_fast"]):
        report_no_mode(nve)
        raise SystemExit(1)

    if nve["w_fast"] > CLIP_MARGIN * args.fmax:
        print("\nWARNING: the fastest mode (%.0f cm^-1) is at the edge of the "
              "%.0f cm^-1\n         analysis window, so the spectrum is "
              "probably clipped and the\n         real fastest mode is higher. "
              "Re-run with a larger --fmax."
              % (nve["w_fast"], args.fmax))

    period = 1.0e15 / (nve["w_fast"] * C_CM_PER_S)
    if args.target:
        target_lo = target_hi = args.target
    else:
        target_lo, target_hi = TAU_TARGET_LO * period, TAU_TARGET_HI * period

    report_nve(nve, target_lo, target_hi)

    temperature = args.temperature or nve["t_mean"]
    report_prediction(nve, temperature, args.n, target_lo, target_hi)

    probes = []
    for d in (args.probe or []):
        try:
            p = tau_c(d, args.drop)
        except (FileNotFoundError, ValueError) as exc:
            print("  skipping %s: %s" % (d, exc))
            continue
        if p:
            probes.append(p)

    if probes:
        report_consistency(probes, nve, temperature)

    print()


if __name__ == "__main__":
    main()
