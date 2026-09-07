#!/usr/bin/env python3
"""
PROGRAM: skl2poscar.py
PURPOSE: Converts an OLCAO skeleton (.skl) file into a VASP POSCAR, so a
         structure prepared for Imago can be relaxed or run through AIMD in
         VASP. Handles symmetry expansion and supercell construction, and
         checks the result for atoms that have landed on top of each other.

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Conceptual design, purpose, and validation by Alysse Weigand.
                 All scientific reasoning, method choices, and interpretation
                 of results by Alysse Weigand.
LAST MODIFIED:   September 2026

USAGE:
    skl2poscar.py input.skl                     # -> POSCAR
    skl2poscar.py input.skl -o POSCAR_MgO
    skl2poscar.py input.skl --supercell 2 2 2   # override the skl directive
    skl2poscar.py input.skl --no-supercell      # ignore the skl directive
    skl2poscar.py input.skl --cartesian         # write Cartesian POSCAR
    skl2poscar.py input.skl --sort              # alphabetize species blocks
    skl2poscar.py input.skl -c "alpha-quartz"   # set the comment line

INPUT:
    An OLCAO .skl file, in either of the two forms cif2skl produces:

      1. Already-expanded cells (space 1 / 1_a), in 'cart' or 'fract'
         coordinates. Needs nothing beyond numpy.

      2. Asymmetric-unit cells with a real space group (e.g. 'space 225').
         Symmetry expansion requires spglib:  pip install --user spglib

    Cartesian input is converted to fractional against the cell, so the 'cell'
    block must appear before a 'cart' block.

OUTPUT:
    POSCAR (or --output), VASP 5 style: comment, scale factor of 1, the three
    lattice vectors, a species line, a counts line, and the positions. Written
    Direct unless --cartesian.

    The console reports the atom count read, the count after expansion, the
    count after any supercell, the species blocks written, and the minimum
    interatomic distance.

INTUITION:
    Two orderings in this conversion matter, and both are silent when wrong.

    Symmetry first, supercell second. OLCAO applies its 'supercell' directive
    after expanding the asymmetric unit, and this script matches that order.
    Reversing them gives a different atom count and a structure that is not
    what the .skl described.

    Species are written in contiguous blocks because VASP requires it: the
    counts line pairs positionally with the species line, and later with the
    blocks in POTCAR. This is the same ordering hazard relaxINCAR.py works
    around by reading species from the POSCAR rather than taking them by hand.
    --sort alphabetizes the blocks; without it they appear in the order first
    encountered in the .skl. Either is valid, but the POTCAR must be built to
    match whichever POSCAR is actually used.

    On the distance check: the most common failure in symmetry expansion is
    generating duplicate atoms, when the input was already expanded but still
    carries a space group, or when positions sit on special sites that the
    symmetry operations map onto themselves. Duplicates are invisible in an
    atom count that looks plausible, but they show up immediately as an
    interatomic distance far below any real bond length.

    The default floor is 0.75 A, just under the 0.74 A H-H bond in H2, so
    anything tripping it cannot be a real bond. That makes the test almost
    free of false alarms but deliberately insensitive: two Si atoms at 1.2 A
    are badly wrong for an oxide and will pass silently. Pass --min-dist with
    the shortest real bond in the system for a stricter check. The warning is
    advisory; the POSCAR is written either way.

    On coordinate cleanup: converting Cartesian to fractional introduces
    float noise that can leave a coordinate at 0.9999999999 instead of 0.
    Positions are rounded and wrapped into [0, 1) before writing, so the
    POSCAR does not carry an atom that appears to sit just outside the cell.
"""

import argparse
import re
import sys

import numpy as np

# Default floor for the interatomic distance sanity check, in Angstroms.
# Set just below the shortest bond that occurs in nature (H-H in H2 is
# 0.74 A), so anything tripping it cannot be a real bond and is almost
# certainly a duplicate atom from symmetry expansion. Deliberately
# insensitive: a pair at 1.2 A is badly wrong for most solids but will not
# be flagged. Raise it with --min-dist when the shortest real bond in the
# system is known.
DEFAULT_MIN_DIST = 0.75

# Keywords that terminate the atom-position block.
BLOCK_KEYWORDS = {
    "space", "supercell", "full", "prim", "primitive", "end",
    "cellfixed", "title", "cell", "cart", "fract", "frac",
}

ELEMENTS = {
    "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al",
    "si", "p", "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn", "fe",
    "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb", "sr",
    "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in", "sn",
    "sb", "te", "i", "xe", "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm",
    "eu", "gd", "tb", "dy", "ho", "er", "tm", "yb", "lu", "hf", "ta", "w",
    "re", "os", "ir", "pt", "au", "hg", "tl", "pb", "bi", "po", "at", "rn",
    "fr", "ra", "ac", "th", "pa", "u", "np", "pu", "am", "cm", "bk", "cf",
    "es", "fm", "md", "no", "lr", "rf", "db", "sg", "bh", "hs", "mt", "ds",
    "rg", "cn", "nh", "fl", "mc", "lv", "ts", "og",
}


def die(msg):
    sys.stderr.write("skl2poscar: error: %s\n" % msg)
    sys.exit(1)


def element_of(tag):
    """'mg1' -> 'Mg', 'o1' -> 'O', 'Fe_2' -> 'Fe'.

    OLCAO atom tags are an element name followed by a species index and
    sometimes a type suffix. Strip everything that is not the element.
    """
    m = re.match(r"[A-Za-z]+", tag.strip())
    if not m:
        die("cannot read an element from atom tag %r" % tag)
    base = m.group(0).lower()
    if base in ELEMENTS:                   # the usual case: 'mg1', 'o1'
        return base.capitalize()
    for n in (2, 1):                       # fall back for suffixed tags
        if base[:n] in ELEMENTS:
            return base[:n].capitalize()
    die("unrecognized element in atom tag %r" % tag)


def cellpar_to_matrix(a, b, c, alpha, beta, gamma):
    """Standard crystallographic convention: a || x, b in the xy plane."""
    al, be, ga = np.radians([alpha, beta, gamma])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0.0:
        die("cell parameters are not physical (c-vector has no real length)")
    vc = np.array([cx, cy, np.sqrt(cz2)])
    lattice = np.vstack([va, vb, vc])
    lattice[np.abs(lattice) < 1e-12] = 0.0
    return lattice


def parse_skl(path):
    """Return dict with lattice, frac coords, element list, sgnum, supercell."""
    with open(path) as fh:
        raw = fh.readlines()

    lines = [ln.rstrip("\n") for ln in raw]
    lattice = None
    coords = None
    elements = None
    sgnum = 1
    sc = [1, 1, 1]

    i = 0
    n = len(lines)
    while i < n:
        tokens = lines[i].split()
        if not tokens:
            i += 1
            continue
        key = tokens[0].lower()

        if key == "title":
            while i < n and lines[i].split() and lines[i].split()[0].lower() != "end":
                i += 1
            i += 1
            continue

        if key == "cell":
            i += 1
            vals = []
            while i < n and len(vals) < 6:
                vals.extend(float(x) for x in lines[i].split())
                i += 1
            if len(vals) < 6:
                die("'cell' block needs 6 values (a b c alpha beta gamma)")
            lattice = cellpar_to_matrix(*vals[:6])
            continue

        if key in ("cart", "fract", "frac"):
            is_frac = key != "cart"
            count = None
            for tok in tokens[1:]:
                try:
                    count = int(tok)
                    break
                except ValueError:
                    continue
            i += 1
            tags, pos = [], []
            while i < n:
                t = lines[i].split()
                if not t:
                    i += 1
                    continue
                if t[0].lower() in BLOCK_KEYWORDS:
                    break
                if len(t) < 4:
                    die("bad atom line %d: %r" % (i + 1, lines[i]))
                tags.append(t[0])
                pos.append([float(v) for v in t[1:4]])
                i += 1
                if count is not None and len(tags) == count:
                    break
            if count is not None and len(tags) != count:
                die("header declared %d atoms but %d were read" % (count, len(tags)))
            if not tags:
                die("no atom positions found")
            pos = np.array(pos, dtype=float)
            if is_frac:
                coords = pos
            else:
                if lattice is None:
                    die("'cart' positions appear before the 'cell' block")
                coords = pos @ np.linalg.inv(lattice)
            elements = [element_of(t) for t in tags]
            continue

        if key == "space":
            if len(tokens) > 1:
                m = re.match(r"(\d+)", tokens[1])
                if m:
                    sgnum = int(m.group(1))
                else:
                    sys.stderr.write(
                        "skl2poscar: warning: non-numeric space group %r, "
                        "assuming the cell is already expanded (P1)\n" % tokens[1])
            i += 1
            continue

        if key == "supercell":
            if len(tokens) >= 4:
                sc = [int(x) for x in tokens[1:4]]
            i += 1
            continue

        i += 1

    if lattice is None:
        die("no 'cell' block found")
    if coords is None:
        die("no 'cart' or 'fract' block found")

    return {"lattice": lattice, "coords": coords, "elements": elements,
            "sgnum": sgnum, "supercell": sc}


def expand_symmetry(lattice, coords, elements, sgnum):
    """Apply space group operations. Requires spglib. Returns new arrays."""
    if sgnum <= 1:
        return coords, elements

    try:
        import spglib
    except ImportError:
        die("space group %d needs symmetry expansion, but spglib is not "
            "installed.\n         Install it with:  pip install --user spglib\n"
            "         Or re-run cif2skl so it writes the full P1 cell."
            % sgnum)

    ops = spglib.get_symmetry_from_database(_hall_from_number(sgnum))
    rots, trans = ops["rotations"], ops["translations"]

    new_coords, new_elems = [], []
    for xyz, el in zip(coords, elements):
        orbit = []
        for R, t in zip(rots, trans):
            p = (R @ xyz + t) % 1.0
            p[np.abs(p - 1.0) < 1e-6] = 0.0
            if not any(np.all(np.abs(_wrap_delta(p - q)) < 1e-5) for q in orbit):
                orbit.append(p)
        new_coords.extend(orbit)
        new_elems.extend([el] * len(orbit))
    return np.array(new_coords), new_elems


def _hall_from_number(sgnum):
    """First (standard) Hall number for a given international number."""
    import spglib
    for hall in range(1, 531):
        t = spglib.get_spacegroup_type(hall)
        if t is None:
            continue
        num = getattr(t, "number", None)          # spglib >= 2.5
        if num is None:
            num = t["number"]                     # older dict interface
        if num == sgnum:
            return hall
    die("space group number %d not found in the spglib database" % sgnum)


def _wrap_delta(d):
    return d - np.round(d)


def make_supercell(lattice, coords, elements, sc):
    nx, ny, nz = sc
    if (nx, ny, nz) == (1, 1, 1):
        return lattice, coords, elements
    if min(sc) < 1:
        die("supercell dimensions must be >= 1")
    new_lat = lattice * np.array(sc)[:, None]
    new_coords, new_elems = [], []
    for xyz, el in zip(coords, elements):
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    new_coords.append((xyz + [ix, iy, iz]) / np.array(sc))
                    new_elems.append(el)
    return new_lat, np.array(new_coords), new_elems


def min_distance(lattice, coords):
    """Shortest interatomic distance under the minimum-image convention."""
    n = len(coords)
    if n < 2:
        return None
    shifts = np.array([[i, j, k] for i in (-1, 0, 1)
                       for j in (-1, 0, 1) for k in (-1, 0, 1)])
    best = np.inf
    for a in range(n - 1):
        d = coords[a + 1:] - coords[a]
        for s in shifts:
            cart = (d + s) @ lattice
            best = min(best, np.sqrt((cart ** 2).sum(axis=1)).min())
    return best


def write_poscar(path, comment, lattice, coords, elements,
                 cartesian=False, sort=False):
    order = []
    for el in elements:
        if el not in order:
            order.append(el)
    if sort:
        order.sort()

    idx = []
    counts = []
    for el in order:
        hits = [k for k, e in enumerate(elements) if e == el]
        idx.extend(hits)
        counts.append(len(hits))

    pos = np.round(coords[idx], 10) % 1.0   # kill cart->frac float noise
    pos[np.abs(pos - 1.0) < 1e-10] = 0.0
    if cartesian:
        pos = pos @ lattice

    with open(path, "w") as fh:
        fh.write(comment.strip() + "\n")
        fh.write("   1.00000000000000\n")
        for v in lattice:
            fh.write("  %20.16f %20.16f %20.16f\n" % tuple(v))
        fh.write("  " + "  ".join("%s" % e for e in order) + "\n")
        fh.write("  " + "  ".join("%d" % c for c in counts) + "\n")
        fh.write("Cartesian\n" if cartesian else "Direct\n")
        for p in pos:
            fh.write("  %19.16f %19.16f %19.16f\n" % tuple(p))
    return order, counts


def main():
    ap = argparse.ArgumentParser(description="Convert an OLCAO .skl file to a VASP POSCAR.")
    ap.add_argument("skl", help="input skeleton file")
    ap.add_argument("-o", "--output", default="POSCAR", help="output file (default: POSCAR)")
    ap.add_argument("--supercell", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                    help="override the supercell directive in the skl file")
    ap.add_argument("--no-supercell", action="store_true",
                    help="ignore the supercell directive entirely")
    ap.add_argument("--cartesian", action="store_true", help="write Cartesian coordinates")
    ap.add_argument("--sort", action="store_true", help="alphabetize the species blocks")
    ap.add_argument("-c", "--comment", help="POSCAR comment line (default: derived from filename)")
    ap.add_argument("--min-dist", type=float, default=DEFAULT_MIN_DIST,
                    metavar="D",
                    help="Warn if any two atoms are closer than D Angstroms "
                         "(default: %(default)s, just under the H-H bond in H2). "
                         "Raise it to the shortest real bond in the system for a "
                         "stricter check. Advisory only; the POSCAR is written "
                         "either way.")
    args = ap.parse_args()

    data = parse_skl(args.skl)
    lattice = data["lattice"]
    coords, elements = data["coords"], data["elements"]
    n_asym = len(elements)

    coords, elements = expand_symmetry(lattice, coords, elements, data["sgnum"])
    n_cell = len(elements)

    sc = [1, 1, 1] if args.no_supercell else (args.supercell or data["supercell"])
    lattice, coords, elements = make_supercell(lattice, coords, elements, sc)

    comment = args.comment or ("%s from %s" %
                               ("".join(sorted(set(elements))), args.skl))
    order, counts = write_poscar(args.output, comment, lattice, coords,
                                 elements, args.cartesian, args.sort)

    print("read %d atom(s) from %s (space group %d)" % (n_asym, args.skl, data["sgnum"]))
    if n_cell != n_asym:
        print("symmetry expansion -> %d atoms in the unit cell" % n_cell)
    if sc != [1, 1, 1]:
        print("supercell %dx%dx%d -> %d atoms" % (sc[0], sc[1], sc[2], len(elements)))
    print("wrote %s: %s = %s (%d atoms total)" %
          (args.output, " ".join(order), " ".join(str(c) for c in counts), len(elements)))

    dmin = min_distance(lattice, coords % 1.0)
    if dmin is not None:
        print("minimum interatomic distance: %.4f A" % dmin)
        if dmin < args.min_dist:
            sys.stderr.write("skl2poscar: WARNING: atoms are closer than "
                             "%.3f A. Check for duplicates from symmetry "
                             "expansion.\n" % args.min_dist)


if __name__ == "__main__":
    main()
