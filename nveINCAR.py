#!/usr/bin/env python3
"""
PROGRAM: nveINCAR.py
PURPOSE: Generates VASP INCAR and KPOINTS files pre-configured for NVE
         molecular dynamics using the Andersen thermostat with collision
         probability set to zero (true NVE). Intended for POTIM validation
         runs and for production NVE AIMD.

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Conceptual design, purpose, and validation by Alysse Weigand.
                 Code implementation and structure assisted by Claude Opus 5,
                 March 2026. Updated September 2026.
LAST MODIFIED:   September 2026

USAGE:
    python nveINCAR.py -npar <int> -potim <float> [-T <float>] [-nsw <int>]
                       [-k NA NB NC] [-wave] [-charge] [--novel]

    python nveINCAR.py -npar <int> --sweep 0.25 0.5 1.0 2.0 [-nsw <int>]

    Either -potim (one run, written here) or --sweep (one directory per
    value) is required. Convergence settings -e/--encut, --ediff, --nelmin,
    and --sigma have defaults suited to an insulator and should be reviewed
    for the material at hand.

EXAMPLE:
    python nveINCAR.py -npar 8 -potim 0.25 -T 300 -nsw 5000
    python nveINCAR.py -npar 8 --sweep 0.25 0.5 1.0 1.5 2.0 -nsw 2000
    python nveINCAR.py -npar 8 --sweep 0.25 0.5 1.0 --novel

INPUT:
    POSCAR   -- read only when --novel is given, and required by --sweep so
                it can be copied into each subdirectory.
    POTCAR   -- required by --sweep, for the same reason. Not read.

    Neither is needed for a single run; the script reports which of the four
    VASP inputs are present when it finishes.

OUTPUT:
    Single run (-potim):
        INCAR, KPOINTS   written into the current directory.

    Sweep (--sweep):
        potim_<value>/   one directory per timestep, each holding INCAR,
                         KPOINTS, and copies of POSCAR and POTCAR. Submit
                         each, then analyse them together with
                         validate_POTIM.py --sweep potim_*

INTUITION:
    NVE is the right ensemble for validating a timestep because it has no
    thermostat to hide integration error. In NVT a thermostat will quietly
    absorb the energy a too-large timestep injects, and the run looks fine
    while the dynamics are wrong. With no such reservoir, a bad POTIM shows
    up directly as drift in the total energy.

    True NVE is obtained here through MDALGO = 1 (Andersen) with ANDERSEN_PROB
    = 0, so no collisions ever occur. This is preferred over MDALGO = 0
    because it keeps VASP on the same MD code path used for the thermostatted
    runs, so the comparison is like for like.

    On --novel: a POSCAR copied from a finished run carries a velocity block,
    and VASP will start from those velocities rather than initialising fresh
    ones at TEBEG. For a timestep sweep this is actively harmful, since the
    runs would no longer share a starting state and the drift comparison
    between them would be meaningless. --novel strips the block before
    anything is written or copied.

    On sweep lengths: every run uses the same NSW, so a larger timestep covers
    more simulated time. validate_POTIM.py reports drift per picosecond rather
    than per step for exactly this reason, but the runs are not equal in
    duration and their energy traces should not be compared step for step.

    The result of a sweep is a CEILING. Once a timestep passes, every smaller
    one does too, so the test is run once per material and a finer timestep
    can be adopted later without revalidating.
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
# Helpers
##################################################

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate VASP INCAR and KPOINTS files for NVE molecular dynamics."
    )

    parser.add_argument("-npar",   type=int,   required=True,
                        help="NPAR value (typically sqrt of the number of cores).")
    parser.add_argument("-potim",  type=float, default=None,
                        help="AIMD timestep in femtoseconds (POTIM). Validate with "
                             "validate_POTIM.py before production runs.")
    parser.add_argument("--sweep", type=float, nargs="+", default=None,
                        metavar="POTIM",
                        help="Generate one subdirectory per POTIM value, each with "
                             "its own INCAR and KPOINTS, and copies of POSCAR and "
                             "POTCAR. Use this for the timestep validation.")
    parser.add_argument("-T",      type=float, default=300.0,
                        help="Starting temperature TEBEG in Kelvin. (default: 300)")
    parser.add_argument("-nsw",    type=int,   default=2000,
                        help="Number of MD steps NSW. (default: 2000)")
    parser.add_argument("-k", "--kpoints", type=int, nargs=3, default=[1, 1, 1],
                        metavar=("NA", "NB", "NC"),
                        help="Gamma-centered k-point mesh. (default: 1 1 1)")
    parser.add_argument("-e", "--encut", type=float, default=600,
                        help="Plane wave cutoff in eV. (default: 600)")
    parser.add_argument("--ediff",  type=float, default=1e-6,
                        help="SCF convergence criterion in eV. (default: 1e-6) "
                             "Tighter than a static calculation needs, because "
                             "SCF noise in the forces shows up as energy drift.")
    parser.add_argument("--nelmin", type=int, default=4,
                        help="Minimum number of electronic steps per ionic step. "
                             "(default: 4) Prevents wavefunction extrapolation "
                             "from leaving under-converged forces.")
    parser.add_argument("--sigma",  type=float, default=0.05,
                        help="Smearing width in eV. (default: 0.05)")
    parser.add_argument("--novel", action="store_true",
                        help="Strip the velocity block from POSCAR before use. "
                             "A POSCAR copied from a previous run's CONTCAR "
                             "carries velocities, and VASP will start from those "
                             "instead of drawing a fresh Maxwell-Boltzmann "
                             "distribution at TEBEG. The original file is kept "
                             "as POSCAR.withvel.")
    parser.add_argument("-wave",   action="store_true",
                        help="Write WAVECAR (LWAVE = .TRUE.). Default: .FALSE.")
    parser.add_argument("-charge", action="store_true",
                        help="Write CHGCAR (LCHARG = .TRUE.). Default: .FALSE.")

    args = parser.parse_args()

    if args.npar <= 0:
        parser.error("-npar must be a positive integer.")
    if args.T <= 0:
        parser.error("-T temperature must be a positive number.")
    if args.nsw <= 0:
        parser.error("-nsw must be a positive integer.")
    if args.potim is None and args.sweep is None:
        parser.error("give either -potim or --sweep.")
    if args.potim is not None and args.sweep is not None:
        parser.error("give either -potim or --sweep, not both.")
    if args.potim is not None and args.potim <= 0:
        parser.error("-potim must be a positive number.")
    if args.sweep is not None and any(p <= 0 for p in args.sweep):
        parser.error("--sweep values must all be positive.")

    return args


def _starts_with_int(line: str) -> bool:
    """True if the first whitespace-separated token on the line is an integer."""
    tokens = line.split()
    if not tokens:
        return False
    try:
        int(tokens[0])
    except ValueError:
        return False
    return True


def structure_line_count(lines: list) -> int:
    """Return how many lines of a POSCAR hold the structure itself.

    That is the header (comment, scale factor, three lattice vectors, the
    optional VASP 5 species-symbol line, the atom counts, the optional
    selective dynamics line, and the coordinate mode line) plus one line
    per atom. Everything past that is the velocity block and any
    predictor-corrector data VASP appended to the CONTCAR.
    """
    if len(lines) < 8:
        raise SystemExit("POSCAR is too short to be a valid structure file.")

    i = 5                                   # comment, scale, three lattice vectors
    if not _starts_with_int(lines[i]):
        i += 1                              # VASP 5 species-symbol line
    if not _starts_with_int(lines[i]):
        raise SystemExit("Could not find the atom-count line in POSCAR.")

    natoms = sum(int(n) for n in lines[i].split())
    i += 1

    if lines[i].strip()[:1] in ("S", "s"):
        i += 1                              # selective dynamics
    i += 1                                  # Direct / Cartesian line

    return i + natoms


def strip_velocities(path: str = "POSCAR") -> bool:
    """Remove the velocity block from a POSCAR in place.

    The original is copied to POSCAR.withvel first. Returns True if the
    file was changed.
    """
    with open(path) as f:
        lines = f.read().splitlines()

    keep = structure_line_count(lines)
    if len(lines) < keep:
        raise SystemExit(
            f"'{path}' has fewer coordinate lines than its atom counts call for."
        )
    if not any(line.strip() for line in lines[keep:]):
        log.info("'%s' has no velocities; nothing to strip.", path)
        return False

    backup = path + ".withvel"
    if os.path.exists(backup):
        log.warning("'%s' already exists and will not be overwritten.", backup)
    else:
        shutil.copy(path, backup)
        log.info("Original copied to '%s'.", backup)

    with open(path, "w") as f:
        f.write("\n".join(lines[:keep]) + "\n")

    log.info("Stripped %d lines of velocity data from '%s'. VASP will now "
             "initialize velocities at TEBEG.", len(lines) - keep, path)
    return True


def build_incar(args, potim: float, lwave: str, lcharg: str) -> str:
    """Build and return the INCAR file contents as a string."""
    w = 10

    lines = [
        "# ===============================",
        "# Material-independent settings",
        "# ===============================",
        f"IBRION = {0:<{w}}# Run molecular dynamics",
        f"NSW    = {args.nsw:<{w}}# Number of ionic (MD) steps",
        f"ISIF   = {2:<{w}}# Fixed cell volume and shape (ions move only)",
        f"MDALGO = {1:<{w}}# Andersen thermostat",
        f"ANDERSEN_PROB = {0.0:<{w}}# Collision probability = 0 gives true NVE",
        f"TEBEG  = {args.T:<{w}}# Starting temperature (K)",
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
        "# ===============================",
        f"POTIM  = {potim:<{w}}# MD timestep (fs) -- material-dependent",
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


def log_summary(args, potim: float, lwave: str, lcharg: str, mesh: str) -> None:
    """Print a summary of the key values written."""
    log.info("  POTIM   = %g fs  (material-dependent, validate before production)",
             potim)
    log.info("  NSW     = %d   (%.2f ps of trajectory)", args.nsw,
             args.nsw * potim / 1000.0)
    log.info("  TEBEG   = %g K", args.T)
    log.info("  EDIFF   = %.0E eV", args.ediff)
    log.info("  NELMIN  = %d",   args.nelmin)
    log.info("  NPAR    = %d",   args.npar)
    log.info("  KPOINTS = %s",   mesh)
    log.info("  LWAVE   = %s",   lwave)
    log.info("  LCHARG  = %s",   lcharg)

##################################################
# Main
##################################################

def main() -> None:
    args = parse_args()

    # Done before anything else is written so that a --sweep copies the
    # stripped POSCAR into every subdirectory.
    if args.novel:
        if os.path.isfile("POSCAR"):
            strip_velocities("POSCAR")
        else:
            log.warning("--novel was given but there is no POSCAR here.")

    lwave  = ".TRUE."  if args.wave   else ".FALSE."
    lcharg = ".TRUE."  if args.charge else ".FALSE."

    na, nb, nc = args.kpoints
    mesh = "Gamma point only" if (na, nb, nc) == (1, 1, 1) else f"{na}x{nb}x{nc}"
    kpoints = build_kpoints(na, nb, nc)

    # ---- Single run ----
    if args.potim is not None:
        write_file(build_incar(args, args.potim, lwave, lcharg), "INCAR")
        write_file(kpoints, "KPOINTS")
        log.info("INCAR and KPOINTS written.")
        log_summary(args, args.potim, lwave, lcharg, mesh)

        print("\nFiles needed for this run:")
        for name in ["POSCAR", "INCAR", "KPOINTS", "POTCAR"]:
            print(f"  {name:<10} {'found' if os.path.isfile(name) else 'MISSING'}")
        return

    # ---- POTIM sweep ----
    missing = [n for n in ("POSCAR", "POTCAR") if not os.path.isfile(n)]
    if missing:
        raise SystemExit(
            f"--sweep needs {' and '.join(missing)} in the current directory "
            "so they can be copied into each subdirectory."
        )

    log.info("Building %d POTIM validation runs.", len(args.sweep))
    for potim in args.sweep:
        d = f"potim_{potim:g}"
        os.makedirs(d, exist_ok=True)
        write_file(build_incar(args, potim, lwave, lcharg), os.path.join(d, "INCAR"))
        write_file(kpoints, os.path.join(d, "KPOINTS"))
        for f in ("POSCAR", "POTCAR"):
            shutil.copy(f, os.path.join(d, f))
        log.info("  %s  (POTIM = %g fs, %.2f ps)", d, potim,
                 args.nsw * potim / 1000.0)

    print("\nEach directory has INCAR, KPOINTS, POSCAR, and POTCAR.")
    print("Submit each one, then analyze them together with validate_POTIM.py.")
    print("\nNote that a longer timestep covers more simulated time for the same")
    print("NSW. The drift criterion is per picosecond, so this is accounted for,")
    print("but the runs are not all the same length in time.")


if __name__ == "__main__":
    main()
