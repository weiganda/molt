#!/usr/bin/env python3
"""
PROGRAM: nvtINCAR.py
PURPOSE: Generates VASP INCAR and KPOINTS files pre-configured for NVT
         molecular dynamics using the Nose-Hoover thermostat (MDALGO = 2).
         Intended for SMASS validation runs and for production NVT AIMD.

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Conceptual design, purpose, and validation by Alysse Weigand.
                 Code implementation and structure assisted by Claude (Anthropic),
                 March 2026. Updated September 2026.
LAST MODIFIED:   September 2026

USAGE:
    # Single run
    python nvtINCAR.py -npar 8 -potim 0.25 -smass 1.0 -nsw 5000

    # SMASS grid, log spaced, default 0.1 to 10 in 5 steps
    python nvtINCAR.py -npar 8 -potim 0.25 --grid

    # Custom grid: min, max, number of points
    python nvtINCAR.py -npar 8 -potim 0.25 --grid 0.05 20 7

    # Explicit list instead of a grid
    python nvtINCAR.py -npar 8 -potim 0.25 --sweep 0.5 1.0 2.0

NOTE ON THE GRID:
    SMASS is spaced logarithmically because it behaves like an inertia and
    its useful range spans orders of magnitude. The two failure modes sit at
    opposite ends: too small and the thermostat drives coherent temperature
    oscillations, too large and it responds sluggishly and suppresses natural
    fluctuation. A log grid brackets both and is far more informative than a
    linear one over the same range.
"""

##################################################
# Imports
##################################################

import os
import shutil
import logging
import argparse

##################################################
# Logging
##################################################

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

##################################################
# Grid
##################################################

GRID_MIN_DEFAULT = 0.1
GRID_MAX_DEFAULT = 10.0
GRID_N_DEFAULT = 5


def log_grid(vmin: float, vmax: float, n: int):
    """
    Return n logarithmically spaced values between vmin and vmax, rounded to
    two significant figures so the directory names stay readable.
    """
    if n < 2:
        return [vmin]
    ratio = (vmax / vmin) ** (1.0 / (n - 1))
    values = [vmin * ratio ** i for i in range(n)]

    rounded = []
    for v in values:
        if v >= 10:
            rounded.append(round(v))
        elif v >= 1:
            rounded.append(round(v, 1))
        else:
            rounded.append(round(v, 2))

    # Rounding can collide; keep the values distinct.
    out = []
    for v in rounded:
        if v not in out:
            out.append(float(v))
    return out

##################################################
# Helpers
##################################################

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate VASP INCAR and KPOINTS files for NVT molecular dynamics."
    )

    parser.add_argument("-npar",   type=int,   required=True,
                        help="NPAR value (typically sqrt of the number of cores).")
    parser.add_argument("-potim",  type=float, required=True,
                        help="AIMD timestep in femtoseconds (POTIM). "
                             "Validate with validate_POTIM.py first.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-smass",  type=float, default=None,
                       help="Nose-Hoover thermostat mass for a single run.")
    group.add_argument("--grid",  nargs="*", type=float, default=None,
                       metavar=("MIN MAX N"),
                       help="Generate a logarithmic grid of SMASS values, one "
                            "directory each. With no arguments the default is "
                            f"{GRID_MIN_DEFAULT} to {GRID_MAX_DEFAULT} in "
                            f"{GRID_N_DEFAULT} steps.")
    group.add_argument("--sweep", nargs="+", type=float, default=None,
                       metavar="SMASS",
                       help="Explicit list of SMASS values instead of a grid.")

    parser.add_argument("-nsw",    type=int,   default=5000,
                        help="Number of MD steps NSW. (default: 5000)")
    parser.add_argument("-TB",     type=float, default=300.0,
                        help="Initial temperature TEBEG in Kelvin. (default: 300)")
    parser.add_argument("-TE",     type=float, default=300.0,
                        help="Final temperature TEEND in Kelvin. (default: 300)")
    parser.add_argument("-k", "--kpoints", type=int, nargs=3, default=[1, 1, 1],
                        metavar=("NA", "NB", "NC"),
                        help="Gamma-centered k-point mesh. (default: 1 1 1)")
    parser.add_argument("-e", "--encut", type=float, default=600,
                        help="Plane wave cutoff in eV. (default: 600)")
    parser.add_argument("--ediff",  type=float, default=1e-6,
                        help="SCF convergence criterion in eV. (default: 1e-6)")
    parser.add_argument("--nelmin", type=int, default=4,
                        help="Minimum electronic steps per ionic step. "
                             "(default: 4)")
    parser.add_argument("--sigma",  type=float, default=0.05,
                        help="Smearing width in eV. (default: 0.05)")
    parser.add_argument("-wave",   action="store_true",
                        help="Write WAVECAR (LWAVE = .TRUE.). Default: .FALSE.")
    parser.add_argument("-charge", action="store_true",
                        help="Write CHGCAR (LCHARG = .TRUE.). Default: .FALSE.")

    args = parser.parse_args()

    if args.npar <= 0:
        parser.error("-npar must be a positive integer.")
    if args.potim <= 0:
        parser.error("-potim must be a positive number.")
    if args.nsw <= 0:
        parser.error("-nsw must be a positive integer.")
    if args.TB <= 0 or args.TE <= 0:
        parser.error("temperatures must be positive.")
    if args.smass is not None and args.smass <= 0:
        parser.error("-smass must be a positive number.")
    if args.sweep is not None and any(s <= 0 for s in args.sweep):
        parser.error("--sweep values must all be positive.")

    # Resolve the grid specification into a list of SMASS values.
    if args.grid is not None:
        if len(args.grid) == 0:
            spec = (GRID_MIN_DEFAULT, GRID_MAX_DEFAULT, GRID_N_DEFAULT)
        elif len(args.grid) == 3:
            spec = (args.grid[0], args.grid[1], int(args.grid[2]))
        else:
            parser.error("--grid takes either no arguments or three: MIN MAX N.")
        if spec[0] <= 0 or spec[1] <= spec[0] or spec[2] < 2:
            parser.error("--grid needs 0 < MIN < MAX and N >= 2.")
        args.smass_values = log_grid(*spec)
    elif args.sweep is not None:
        args.smass_values = sorted(args.sweep)
    else:
        args.smass_values = None

    return args


def build_incar(args, smass: float, lwave: str, lcharg: str) -> str:
    """Build and return the INCAR file contents as a string."""
    w = 10

    lines = [
        "# ===============================",
        "# Material-independent settings",
        "# ===============================",
        f"IBRION = {0:<{w}}# Run molecular dynamics",
        f"NSW    = {args.nsw:<{w}}# Number of ionic (MD) steps",
        f"ISIF   = {2:<{w}}# Fixed cell volume and shape (ions move only)",
        f"MDALGO = {2:<{w}}# NVT, Nose-Hoover thermostat",
        f"TEBEG  = {args.TB:<{w}}# Initial temperature (K)",
        f"TEEND  = {args.TE:<{w}}# Final temperature (K)",
        "",
        f"ENCUT  = {args.encut:<{w}.0f}# Plane-wave cutoff energy (eV)",
        f"PREC   = {'Accurate':<{w}}# Plane-wave grid accuracy",
        f"EDIFF  = {args.ediff:<{w}.0E}# SCF convergence criterion (eV)",
        f"NELMIN = {args.nelmin:<{w}}# Minimum electronic steps per ionic step",
        f"ALGO   = {'Normal':<{w}}# Electronic minimization algorithm",
        "",
        f"ISYM   = {0:<{w}}# Disable symmetry",
        f"ISMEAR = {0:<{w}}# Gaussian smearing",
        f"SIGMA  = {args.sigma:<{w}}# Smearing width (eV)",
        "",
        f"NPAR   = {args.npar:<{w}}# Parallelization (sqrt of core count)",
        f"LREAL  = {'Auto':<{w}}# Real-space projection",
        f"LWAVE  = {lwave:<{w}}# Write WAVECAR",
        f"LCHARG = {lcharg:<{w}}# Write CHGCAR",
        "",
        "# ===============================",
        "# Material-dependent settings",
        "# -- Validate POTIM with validate_POTIM.py before production runs.",
        "# -- Validate SMASS with validate_SMASS.py before production runs.",
        "# ===============================",
        f"POTIM  = {args.potim:<{w}}# MD timestep (fs) -- material-dependent",
        f"SMASS  = {smass:<{w}}# Nose-Hoover mass -- material-dependent",
    ]

    return "\n".join(lines) + "\n"


def build_kpoints(na: int, nb: int, nc: int) -> str:
    """Build and return a gamma-centered KPOINTS file."""
    return f"""Automatic mesh
0
Gamma
 {na} {nb} {nc}
 0 0 0
"""


def write_file(content: str, path: str) -> None:
    """Write content to a path, warning if it already exists."""
    if os.path.exists(path):
        log.warning("'%s' already exists and will be overwritten.", path)
    with open(path, "w") as f:
        f.write(content)

##################################################
# Main
##################################################

def main() -> None:
    args = parse_args()

    lwave  = ".TRUE."  if args.wave   else ".FALSE."
    lcharg = ".TRUE."  if args.charge else ".FALSE."

    na, nb, nc = args.kpoints
    mesh = "Gamma point only" if (na, nb, nc) == (1, 1, 1) else f"{na}x{nb}x{nc}"
    kpoints = build_kpoints(na, nb, nc)

    duration_ps = args.nsw * args.potim / 1000.0

    # ---- Single run ----
    if args.smass_values is None:
        write_file(build_incar(args, args.smass, lwave, lcharg), "INCAR")
        write_file(kpoints, "KPOINTS")
        log.info("INCAR and KPOINTS written.")
        log.info("  POTIM   = %g fs", args.potim)
        log.info("  SMASS   = %g", args.smass)
        log.info("  NSW     = %d   (%.2f ps)", args.nsw, duration_ps)
        log.info("  TEBEG   = %g K, TEEND = %g K", args.TB, args.TE)
        log.info("  KPOINTS = %s", mesh)

        print("\nFiles needed for this run:")
        for name in ["POSCAR", "INCAR", "KPOINTS", "POTCAR"]:
            print(f"  {name:<10} {'found' if os.path.isfile(name) else 'MISSING'}")
        return

    # ---- SMASS grid ----
    missing = [n for n in ("POSCAR", "POTCAR") if not os.path.isfile(n)]
    if missing:
        raise SystemExit(
            f"A SMASS grid needs {' and '.join(missing)} in the current "
            "directory so they can be copied into each subdirectory."
        )

    values = args.smass_values
    log.info("Building %d SMASS validation runs, %.2f ps each.",
             len(values), duration_ps)
    log.info("SMASS values: %s", ", ".join("%g" % v for v in values))

    for smass in values:
        d = f"smass_{smass:g}"
        os.makedirs(d, exist_ok=True)
        write_file(build_incar(args, smass, lwave, lcharg),
                   os.path.join(d, "INCAR"))
        write_file(kpoints, os.path.join(d, "KPOINTS"))
        for f in ("POSCAR", "POTCAR"):
            shutil.copy(f, os.path.join(d, f))
        log.info("  %s", d)

    print("\nEach directory has INCAR, KPOINTS, POSCAR, and POTCAR.")
    print("Submit each one, then analyze them together with validate_SMASS.py.")
    print("\nUnlike the POTIM sweep, every run here covers the same amount of")
    print("simulated time, since only the thermostat mass changes.")


if __name__ == "__main__":
    main()
