#!/usr/bin/env python3

"""
PROGRAM: relaxINCAR.py
PURPOSE: Generates the VASP input files needed for a structural relaxation.
         Writes INCAR and KPOINTS, and assembles POTCAR from a local PAW
         library by reading the species from the POSCAR.

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Conceptual design, purpose, and validation by Alysse Weigand.
                 All scientific reasoning, method choices, and interpretation
                 of results by Alysse Weigand.
LAST MODIFIED:   September 2026

USAGE:
    python relaxINCAR.py -npar 4
    python relaxINCAR.py -npar 4 -k 2 2 2
    python relaxINCAR.py -npar 4 --functional potpaw_PBE
    python relaxINCAR.py -npar 4 --pp-map Si=Si_sv
    python relaxINCAR.py -npar 4 --species Si O     # VASP 4 POSCAR
    python relaxINCAR.py -npar 4 --no-potcar        # INCAR + KPOINTS only

INPUT:
    POSCAR   -- required unless --no-potcar is given. Read for the element
                symbols and atom counts, in POSCAR order.

    A PAW potential library, located in this order: --pp-root, then the
    VASP_PP_PATH environment variable, then $CPG_SHARE/vasp_pot. The set
    within it is chosen with --functional.

OUTPUT (written to the current directory):
    INCAR    -- relaxation parameters
    KPOINTS  -- Gamma-centered mesh
    POTCAR   -- species concatenated in POSCAR order, unless --no-potcar

    A VASP relaxation needs four files. POSCAR is yours to supply; the other
    three are written here. The script ends by reporting which of the four are
    present.

INTUITION:
    POTCAR order is not cosmetic. VASP matches the blocks in POTCAR against
    the species lines in POSCAR positionally, so a POTCAR built in the wrong
    order produces a run that completes and is silently wrong. That is why the
    species are read from the POSCAR rather than supplied by hand, and why
    --species exists for VASP 4 files that carry no symbol line: the ordering
    has to come from the same place VASP will read it from.

    On --functional: the default is potpaw_LDA, matching the Ceperley-Alder
    functional used elsewhere in this work. A PAW potential is generated for a
    specific exchange-correlation functional and is not transferable between
    them, so this must agree with the rest of the calculation.

    On --pp-map: some elements need a harder or more-electron potential than
    the bare element folder provides (Si_sv, O_h, and so on). Anything not
    named in --pp-map uses the folder matching the element symbol.

    On EDIFFG: a negative value is a force criterion in eV/A, a positive value
    is an energy criterion in eV. The default here is negative, so the run
    stops when forces fall below the threshold. Pass the same magnitude to
    validate_relaxation.py --force-req when checking the result.

    On ISIF = 3: relaxing the cell volume while the plane wave basis stays
    fixed at the starting volume means the basis is no longer converged once
    the cell changes, an artefact known as Pulay stress. The remedy is to copy
    CONTCAR to POSCAR and rerun until the volume stops moving. The script
    prints this reminder whenever ISIF is 3.

    A note on compressed libraries: PAW sets are often distributed with the
    files left as POTCAR.Z or POTCAR.gz. The .Z form is the older LZW
    'compress' format, which Python's gzip module cannot read, so it is
    decompressed by calling out to the system. Both forms are handled.
"""

##################################################
# Imports
##################################################

import os
import gzip
import argparse
import subprocess


##################################################
# Arguments
##################################################


def parse_args():
    """Parse and return the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate INCAR and KPOINTS for a VASP relaxation."
    )

    parser.add_argument("-npar", type=int, required=True,
                        help="Value for NPAR. Roughly the square root of the number "
                             "of cores used, and at least NCPUs/32.")
    parser.add_argument("-k", "--kpoints", type=int, nargs=3, default=[1, 1, 1],
                        metavar=("NA", "NB", "NC"),
                        help="Gamma-centered k-point mesh (default: 1 1 1, "
                             "which is the Gamma point only).")
    parser.add_argument("-e", "--encut", type=float, default=600,
                        help="Plane wave energy cutoff in eV (default: 600).")
    parser.add_argument("--ediff", type=float, default=1.0e-7,
                        help="Electronic convergence criterion in eV (default: 1e-7).")
    parser.add_argument("--ediffg", type=float, default=-1.0e-5,
                        help="Ionic convergence criterion. Negative values are a "
                             "force criterion in eV/A (default: -1e-5).")
    parser.add_argument("--nsw", type=int, default=100,
                        help="Maximum number of ionic steps (default: 100).")
    parser.add_argument("--isif", type=int, default=3,
                        help="2 or 4 relaxes ions, 7 relaxes volume, 3 relaxes both "
                             "(default: 3).")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Smearing width in eV (default: 0.05, appropriate for "
                             "an insulator).")
    parser.add_argument("-wave", action="store_true",
                        help="Write WAVECAR (LWAVE = .TRUE.)")
    parser.add_argument("-charge", action="store_true",
                        help="Write CHGCAR (LCHARG = .TRUE.)")
    parser.add_argument("--no-potcar", action="store_true",
                        help="Skip POTCAR assembly.")
    parser.add_argument("--functional", default="potpaw_LDA",
                        help="PAW potential set to use (default: potpaw_LDA, which "
                             "matches the Ceperley-Alder functional used elsewhere "
                             "in this work).")
    parser.add_argument("--pp-root", default="",
                        help="Directory holding the PAW potential sets. If omitted, "
                             "the script looks at VASP_PP_PATH, then at "
                             "$CPG_SHARE/vasp_pot.")
    parser.add_argument("--species", nargs="*", default=[], metavar="EL",
                        help="Element symbols in POSCAR order, for example Si O. "
                             "Required for VASP 4 style POSCAR files that do not "
                             "carry a species line.")
    parser.add_argument("--pp-map", nargs="*", default=[],
                        metavar="EL=FOLDER",
                        help="Override the PAW folder for a species, for example "
                             "Si=Si_sv O=O_h. Anything not listed uses the bare "
                             "element name.")

    return parser.parse_args()


##################################################
# INCAR
##################################################


def build_incar(args):
    """Return the INCAR text for a relaxation run."""
    lwave = ".TRUE." if args.wave else ".FALSE."
    lcharg = ".TRUE." if args.charge else ".FALSE."

    return f"""! Relaxation
! ============================================================

ISMEAR = 0        ! Gaussian smearing. -5 creates a tetrahedron error here.
SIGMA  = {args.sigma}     ! Smearing width in eV. Keep small for an insulator.

PREC   = Accurate ! low, medium, normal are other options.
ENCUT  = {args.encut:.0f}       ! Plane wave cutoff in eV.
EDIFF  = {args.ediff:.1E}    ! Electronic convergence criterion, eV.
EDIFFG = {args.ediffg:.1E}   ! Ionic criterion. Negative is a force limit, eV/A.

IBRION = 1        ! 0 for MD, 1 best for relaxation, 2 for difficult cases.
NSW    = {args.nsw}      ! Maximum number of ionic steps.
ISIF   = {args.isif}        ! 2 and 4 ionic, 7 volume, 3 both.

LREAL  = Auto     ! Real space projection. FALSE uses reciprocal space.
NPAR   = {args.npar}        ! Roughly sqrt of the core count.
ALGO   = Normal   ! Electronic minimization algorithm.

LCHARG = {lcharg} ! Write the CHGCAR.
LWAVE  = {lwave} ! Write the WAVECAR.
"""


##################################################
# KPOINTS
##################################################


def build_kpoints(na, nb, nc):
    """Return the KPOINTS text for a Gamma-centered mesh."""
    return f"""Automatic mesh
0
Gamma
 {na} {nb} {nc}
 0 0 0
"""


def mesh_description(na, nb, nc):
    """Human-readable name for the mesh, used in the console message."""
    if (na, nb, nc) == (1, 1, 1):
        return "Gamma point only"
    return f"Gamma-centered {na}x{nb}x{nc} mesh"


##################################################
# POTCAR
##################################################


def read_species(poscar="POSCAR", species_override=None):
    """
    Return the element symbols in POSCAR order.

    VASP 5 style files list the symbols on line 6 and the counts on line 7.
    VASP 4 style files have the counts on line 6 and no symbol line at all,
    in which case the symbols are taken from species_override (the --species
    argument), or failing that from the comment on line 1.
    """
    species_override = species_override or []

    with open(poscar) as f:
        lines = f.readlines()
    if len(lines) < 7:
        raise SystemExit(f"'{poscar}' is too short to be a valid POSCAR.")

    line6 = lines[5].split()
    is_vasp5 = bool(line6) and not line6[0].lstrip("+-").isdigit()

    if is_vasp5:
        species = line6
        counts = lines[6].split()
    else:
        counts = line6
        if species_override:
            species = species_override
            source = "--species"
        else:
            # Last resort: try the comment line, which often names the elements.
            candidates = [w.strip(",;") for w in lines[0].split()]
            species = [w for w in candidates
                       if w.isalpha() and 1 <= len(w) <= 2 and w[0].isupper()]
            source = "the POSCAR comment line"
            if len(species) != len(counts):
                raise SystemExit(
                    f"'{poscar}' is a VASP 4 style file with no species line, "
                    f"and the element symbols could not be read from the "
                    f"comment.\n"
                    f"Line 6 lists {len(counts)} atom counts: {' '.join(counts)}\n"
                    f"Rerun with the symbols in POSCAR order, for example:\n"
                    f"    --species Si O"
                )
        print(f"POSCAR is VASP 4 style; species taken from {source}.")

    if len(species) != len(counts):
        raise SystemExit(
            f"Mismatch in '{poscar}': {len(species)} species "
            f"({' '.join(species)}) but {len(counts)} atom counts "
            f"({' '.join(counts)})."
        )

    total = sum(int(c) for c in counts)
    print(f"Structure: {total} atoms, "
          + ", ".join(f"{n} {el}" for el, n in zip(species, counts)))
    return species


def read_potcar(folder_path):
    """
    Read a POTCAR from a species folder, handling compressed files.

    VASP PAW libraries are often distributed with the files left compressed,
    either as POTCAR.Z (LZW, the older 'compress' format) or POTCAR.gz. The
    .Z format is not readable by Python's gzip module, so it is decompressed
    by calling out to the system.

    Returns the POTCAR text, or None if no readable file was found.
    """
    plain = os.path.join(folder_path, "POTCAR")

    if os.path.isfile(plain):
        with open(plain) as f:
            return f.read()

    gz = plain + ".gz"
    if os.path.isfile(gz):
        with gzip.open(gz, "rt") as f:
            return f.read()

    dotz = plain + ".Z"
    if os.path.isfile(dotz):
        for cmd in (["gzip", "-dc", dotz],
                    ["zcat", dotz],
                    ["uncompress", "-c", dotz]):
            try:
                result = subprocess.run(cmd, capture_output=True,
                                        text=True, check=True)
                return result.stdout
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        raise SystemExit(
            f"Found '{dotz}' but could not decompress it. None of gzip, zcat, "
            "or uncompress worked."
        )

    return None


def resolve_pp_root(pp_root_arg=""):
    """
    Find the directory holding the PAW potential sets.

    Searched in order: the --pp-root argument, VASP_PP_PATH, then
    $CPG_SHARE/vasp_pot. Returns (path, where_it_came_from).
    """
    candidates = [
        (pp_root_arg, "--pp-root"),
        (os.environ.get("VASP_PP_PATH", ""), "VASP_PP_PATH"),
        (os.path.join(os.environ.get("CPG_SHARE", ""), "vasp_pot")
         if os.environ.get("CPG_SHARE") else "", "$CPG_SHARE/vasp_pot"),
    ]
    for path, source in candidates:
        if path and os.path.isdir(path):
            return path, source
    raise SystemExit(
        "Could not locate the PAW potential library. Set CPG_SHARE, set "
        "VASP_PP_PATH, or pass --pp-root."
    )


def parse_pp_map(entries):
    """Turn the --pp-map list of El=Folder strings into a dict."""
    overrides = {}
    for item in entries:
        if "=" not in item:
            raise SystemExit(f"--pp-map entry '{item}' must look like El=Folder.")
        el, folder = item.split("=", 1)
        overrides[el] = folder
    return overrides


def write_potcar(args):
    """
    Assemble POTCAR by concatenating the per-species files in POSCAR order.

    Order matters: VASP pairs POTCAR blocks with POSCAR species positionally,
    so a mis-ordered POTCAR gives a run that completes and is wrong.
    """
    pp_root, source = resolve_pp_root(args.pp_root)
    pp_path = os.path.join(pp_root, args.functional)

    if not os.path.isdir(pp_path):
        available = sorted(d for d in os.listdir(pp_root)
                           if os.path.isdir(os.path.join(pp_root, d)))
        raise SystemExit(
            f"'{args.functional}' not found in {pp_root}.\n"
            f"Available sets: {', '.join(available)}"
        )

    print(f"\nPAW library: {pp_root}  (from {source})")
    print(f"Potential set: {args.functional}")

    species = read_species(species_override=args.species)
    overrides = parse_pp_map(args.pp_map)

    chunks = []
    for el in species:
        folder = overrides.get(el, el)
        folder_path = os.path.join(pp_path, folder)

        if not os.path.isdir(folder_path):
            raise SystemExit(
                f"No folder '{folder}' in {pp_path}. Use --pp-map to point "
                f"'{el}' at the correct one, for example --pp-map {el}={el}_sv."
            )

        text = read_potcar(folder_path)
        if text is None:
            raise SystemExit(
                f"No POTCAR, POTCAR.Z, or POTCAR.gz found in '{folder_path}'."
            )

        chunks.append(text)
        print(f"  {el}: {folder}")

    with open("POTCAR", "w") as f:
        f.write("".join(chunks))
    print(f"POTCAR written for {' '.join(species)}.")


##################################################
# Checklist
##################################################


def print_checklist(args):
    """Report which of the four VASP inputs are present, and warn about ISIF 3."""
    print("\nFiles needed for this run:")
    for name in ["POSCAR", "INCAR", "KPOINTS", "POTCAR"]:
        status = "found" if os.path.isfile(name) else "MISSING"
        print(f"  {name:<10} {status}")

    if args.isif == 3:
        print(
            "\nNote: ISIF = 3 relaxes the cell volume while the plane wave basis is\n"
            "fixed at the starting volume, so the basis is no longer converged once\n"
            "the cell changes. Copy CONTCAR to POSCAR and rerun until the volume\n"
            "stops changing."
        )


##################################################
# Main
##################################################


def main():
    args = parse_args()

    with open("INCAR", "w") as f:
        f.write(build_incar(args))
    print("INCAR written.")

    na, nb, nc = args.kpoints
    with open("KPOINTS", "w") as f:
        f.write(build_kpoints(na, nb, nc))
    print(f"KPOINTS written ({mesh_description(na, nb, nc)}).")

    if not args.no_potcar:
        write_potcar(args)

    print_checklist(args)


if __name__ == "__main__":
    main()
