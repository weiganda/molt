#!/usr/bin/env python3
"""
PROGRAM: validate_POTIM.py
PURPOSE: Validates the AIMD timestep (POTIM) for a material by measuring the
         total-energy drift of a short NVE run. Analyses either a single run
         or a sweep of runs at different POTIM values, and writes the figure,
         LaTeX table, and CSV needed for the dissertation.

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Conceptual design, purpose, and validation by Alysse Weigand.
                 All scientific reasoning, method choices, and interpretation
                 of results by Alysse Weigand.
                 Code implementation and structure assisted by Claude Opus 5.
                 September 2026.
LAST MODIFIED:   September 2026

USAGE:
    # Single NVE run, analysed in the current directory
    python validate_POTIM.py --label SiO2

    # Sweep produced by:  nveINCAR.py -npar 8 --sweep 0.25 0.5 1.0 2.0
    python validate_POTIM.py --label SiO2 --sweep potim_*

    # Add LaTeX fragments for the appendix
    python validate_POTIM.py --label SiO2 --sweep potim_* --appendix

INPUT (per directory):
    OUTCAR   -- required. VASP output from an NVE run.
    INCAR    -- optional. Used for POTIM if present; falls back to OUTCAR.

OUTPUT:
    POTIM_<label>.png / .pdf   Figure (energy traces; drift vs POTIM in sweep)
    POTIM_<label>.csv          Per-run drift data
    POTIM_<label>.tex          LaTeX table, caption below the tabular
    POTIM_<label>_fig.tex      Figure fragment (--appendix only)

INTUITION:
    POTIM must be small enough to resolve the fastest vibration in the
    material. Too large and the integrator fails to conserve energy, so the
    total energy drifts. Too small and the run is merely expensive.

    The validation therefore produces a CEILING, not a target: once a POTIM
    passes, every smaller value also passes, so the test is done once per
    material. If a later spectral analysis needs a finer sampling rate, a
    POTIM below the validated ceiling is free to use (check the sampling rate
    with calculateSamplingRate.py).

    Drift criteria, on the slope-based number, in eV/atom/ps:
        < 1e-5        excellent
        1e-5 to 1e-4  acceptable
        > 1e-4        POTIM is too large

    Note on the figure: panel (a) plots energy against ionic step, so every
    run occupies the same horizontal range regardless of its timestep. The
    slopes visible there are per step, not per picosecond, and are therefore
    NOT directly comparable between curves. Panel (b) carries the quantity the
    criterion is applied to.
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
matplotlib.use("Agg")          # headless: Hellbender compute nodes have no display
import matplotlib.pyplot as plt

##################################################
# Logging
##################################################

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

##################################################
# Constants
##################################################

DRIFT_EXCELLENT = 1.0e-5       # eV/atom/ps
DRIFT_ACCEPTABLE = 1.0e-4      # eV/atom/ps

FS_TO_PS = 1.0e-3

##################################################
# Parsing
##################################################


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    p = argparse.ArgumentParser(
        description="Validate the AIMD timestep (POTIM) from NVE energy drift.")

    p.add_argument("--label", required=True,
                   help="Material label used in filenames and captions "
                        "(e.g. SiO2).")
    p.add_argument("--sweep", nargs="+", metavar="DIR", default=None,
                   help="Directories, one per POTIM value, each holding an "
                        "OUTCAR. Omit to analyse the current directory.")
    p.add_argument("--drop", type=float, default=0.5, metavar="FRAC",
                   help="Fraction of the trajectory discarded before the "
                        "slope fit, to skip the initial transient. "
                        "(default: 0.5)")
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


def read_natoms(outcar_lines):
    """Number of ions, needed to normalise drift per atom."""
    for line in outcar_lines:
        if "NIONS" in line:
            m = re.search(r"NIONS\s*=\s*(\d+)", line)
            if m:
                return int(m.group(1))
    return None


def read_potim(directory, outcar_lines):
    """POTIM in fs. Prefer INCAR; fall back to the OUTCAR parameter echo."""
    incar = os.path.join(directory, "INCAR")
    if os.path.isfile(incar):
        with open(incar) as fh:
            for line in fh:
                line = line.split("#")[0].split("!")[0].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip().upper() == "POTIM":
                    try:
                        return float(value.split()[0])
                    except (ValueError, IndexError):
                        pass

    for line in outcar_lines:
        if "POTIM" in line:
            m = re.search(r"POTIM\s*=\s*([-\d.eE+]+)", line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
    return None


def read_energies(outcar_lines):
    """Conserved quantity ETOTAL = TOTEN + EKIN, one value per MD step."""
    energies = []
    for line in outcar_lines:
        if "ETOTAL" in line.upper():
            m = re.search(r"ETOTAL\s*=\s*([-\d.eE+]+)", line, re.I)
            if m:
                energies.append(float(m.group(1)))
    return energies


def read_ensemble(directory, outcar_lines):
    """Recover MDALGO / ANDERSEN_PROB / SMASS so a thermostatted run can be
    flagged. Drift analysis is only meaningful without a thermostat."""
    info = {}
    text = "".join(outcar_lines)

    m = re.search(r"MDALGO\s*=\s*(\d+)", text)
    if m:
        info["MDALGO"] = int(m.group(1))
    m = re.search(r"ANDERSEN_PROB\s*=\s*([-\d.eE+]+)", text)
    if m:
        info["ANDERSEN_PROB"] = float(m.group(1))
    m = re.search(r"SMASS\s*=\s*([-\d.eE+]+)", text)
    if m:
        info["SMASS"] = float(m.group(1))

    incar = os.path.join(directory, "INCAR")
    if os.path.isfile(incar):
        with open(incar) as fh:
            for line in fh:
                line = line.split("#")[0].split("!")[0].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().upper()
                if k in ("MDALGO", "ANDERSEN_PROB", "SMASS"):
                    try:
                        info[k] = float(v.split()[0])
                    except (ValueError, IndexError):
                        pass
    return info


def is_nve(info):
    """True if the run looks like NVE (no active thermostat)."""
    algo = info.get("MDALGO")
    if algo is None:
        return True                       # cannot tell; assume the user knows
    if int(algo) == 0:
        return True                       # plain Verlet
    if int(algo) == 1:                    # Andersen; NVE only if prob == 0
        return abs(info.get("ANDERSEN_PROB", 1.0)) < 1e-12
    return False                          # Nose-Hoover, Langevin, ...


##################################################
# Analysis
##################################################


def analyse_run(directory, drop_fraction):
    """Read one NVE run and compute its energy drift.

    Two independent drift measures are reported:

    1. Endpoint drift, from the first and last energies. Simple, but a single
       unlucky fluctuation at either end moves it.

    2. Slope drift, from a least-squares line fitted to the steady-state part
       of the trajectory. This is the number the criterion is applied to,
       because it uses every point and is insensitive to the endpoints.

       drift = |slope| / (POTIM * 1e-3) / N_atoms      eV/atom/ps

       with slope in eV/step. Fitting only the steady-state portion avoids
       the initial equilibration transient.

    The reported uncertainty is the least-squares standard error of the slope.
    It does not account for correlation between successive MD steps, so it is
    a lower bound on the true uncertainty and is quoted for scale only.
    """
    outcar = os.path.join(directory, "OUTCAR")
    if not os.path.isfile(outcar):
        raise FileNotFoundError("no OUTCAR in %s" % directory)

    with open(outcar, errors="replace") as fh:
        lines = fh.readlines()

    natoms = read_natoms(lines)
    if natoms is None:
        raise ValueError("could not find NIONS in %s" % outcar)

    potim = read_potim(directory, lines)
    if potim is None:
        raise ValueError("could not find POTIM in %s or its INCAR" % directory)

    energies = np.asarray(read_energies(lines), dtype=float)
    if energies.size < 20:
        raise ValueError("only %d MD steps found in %s; need at least 20"
                         % (energies.size, outcar))

    info = read_ensemble(directory, lines)
    nve = is_nve(info)

    nsteps = energies.size
    steps = np.arange(1, nsteps + 1)
    t_ps = steps * potim * FS_TO_PS
    t_total_ps = nsteps * potim * FS_TO_PS

    # --- endpoint drift ---------------------------------------------------
    dE_endpoint = energies[-1] - energies[0]
    drift_endpoint = abs(dE_endpoint) / (natoms * t_total_ps)

    # --- slope drift over the steady-state portion ------------------------
    i0 = int(drop_fraction * nsteps)
    ss_steps = steps[i0:]
    ss_energies = energies[i0:]

    coeffs, cov = np.polyfit(ss_steps, ss_energies, 1, cov=True)
    slope = coeffs[0]                                   # eV per step
    slope_err = float(np.sqrt(cov[0, 0]))

    per_ps = 1.0 / (potim * FS_TO_PS)                   # steps per ps
    drift_slope = abs(slope) * per_ps / natoms
    drift_slope_err = slope_err * per_ps / natoms

    if drift_slope < DRIFT_EXCELLENT:
        verdict = "excellent"
    elif drift_slope <= DRIFT_ACCEPTABLE:
        verdict = "acceptable"
    else:
        verdict = "too large"

    return {
        "directory": directory,
        "potim": potim,
        "natoms": natoms,
        "nsteps": nsteps,
        "t_total_ps": t_total_ps,
        "energies": energies,
        "t_ps": t_ps,
        "steps": steps,
        "fit_start": i0,
        "fit_line": np.polyval(coeffs, ss_steps),
        "fit_t_ps": t_ps[i0:],
        "fit_steps": ss_steps,
        "dE_endpoint": dE_endpoint,
        "drift_endpoint": drift_endpoint,
        "drift_slope": drift_slope,
        "drift_slope_err": drift_slope_err,
        "verdict": verdict,
        "nve": nve,
        "ensemble": info,
    }


##################################################
# Reporting
##################################################


def print_report(runs, drop_fraction):
    """Terminal summary."""
    print("\n---------- Simulation Parameters ----------")
    print("Material atoms (NIONS) = %d" % runs[0]["natoms"])
    print("Runs analysed          = %d" % len(runs))
    print("Steady-state fit uses the last %d%% of each trajectory"
          % round((1.0 - drop_fraction) * 100))
    print("-------------------------------------------\n")

    print("--------------- Energy Drift --------------")
    header = ("%-9s %8s %9s %13s %13s  %s"
              % ("POTIM/fs", "steps", "time/ps", "endpoint", "slope", "verdict"))
    print(header)
    print("-" * len(header))
    for r in runs:
        print("%-9.3f %8d %9.3f %13.2e %13.2e  %s"
              % (r["potim"], r["nsteps"], r["t_total_ps"],
                 r["drift_endpoint"], r["drift_slope"], r["verdict"]))
    print("-" * len(header))
    print("drift in eV/atom/ps; criterion applied to the slope value")
    print("  < %.0e excellent | <= %.0e acceptable | above that, too large"
          % (DRIFT_EXCELLENT, DRIFT_ACCEPTABLE))
    print("------------------------------------------\n")

    # The ceiling is the largest timestep such that every smaller value also
    # passes. Taking the largest passing value on its own would be wrong when
    # the drift is not monotonic: a point that scatters just under the
    # criterion above a failing one is not a stable timestep.
    ceiling = None
    for r in runs:                      # runs are already sorted by potim
        if r["verdict"] == "too large":
            break
        ceiling = r["potim"]

    if ceiling is None:
        print("RESULT: no tested POTIM meets the criterion. Reduce the "
              "timestep and rerun.\n")
    else:
        print("RESULT: largest acceptable timestep is POTIM = %.3f fs.\n"
              "        Any smaller value is also valid.\n" % ceiling)
        larger = [r["potim"] for r in runs
                  if r["potim"] > ceiling and r["verdict"] != "too large"]
        if larger:
            print("        Note that %s also fell below the criterion, but a\n"
                  "        larger timestep failed in between, so these are\n"
                  "        scatter rather than evidence of stability.\n"
                  % ", ".join("%g fs" % p for p in larger))


def write_csv(path, runs):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["directory", "potim_fs", "natoms", "nsteps", "time_ps",
                    "dE_endpoint_eV", "drift_endpoint_eV_atom_ps",
                    "drift_slope_eV_atom_ps", "drift_slope_err", "verdict"])
        for r in runs:
            w.writerow([r["directory"], "%.4f" % r["potim"], r["natoms"],
                        r["nsteps"], "%.4f" % r["t_total_ps"],
                        "%.6f" % r["dE_endpoint"],
                        "%.6e" % r["drift_endpoint"],
                        "%.6e" % r["drift_slope"],
                        "%.6e" % r["drift_slope_err"], r["verdict"]])


def write_tex_table(path, runs, label):
    """LaTeX table. Caption below the tabular, no publisher-specific macros."""
    tag = re.sub(r"[^A-Za-z0-9]", "", label)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{rrrrrl}",
        r"\hline",
        (r"POTIM (fs) & Steps & Time (ps) & Endpoint drift & Slope drift "
         r"& Verdict \\"),
        (r" &  &  & (eV/atom/ps) & (eV/atom/ps) &  \\"),
        r"\hline",
    ]
    for r in runs:
        lines.append(r"%.2f & %d & %.2f & %s & %s & %s \\"
                     % (r["potim"], r["nsteps"], r["t_total_ps"],
                        _sci(r["drift_endpoint"]), _sci(r["drift_slope"]),
                        r["verdict"]))
    lines += [
        r"\hline",
        r"\end{tabular}",
        (r"\caption{Total-energy drift of NVE test runs for %s. All runs used "
         r"the same number of ionic steps, so they cover different amounts of "
         r"simulated time; the drift is reported per picosecond to account for "
         r"this. The criterion is applied to the slope-based drift.}" % label),
        r"\label{tab:potim_%s}" % tag,
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
        r"\includegraphics[width=0.85\textwidth]{%s}" % figname,
        (r"\caption{POTIM validation for %s. Panel (a) shows the total energy "
         r"of each NVE test run against ionic step, with the steady-state fit "
         r"overlaid. Because the runs cover different amounts of simulated "
         r"time, the slopes shown there are per step and are not directly "
         r"comparable between curves. Panel (b) gives the drift per "
         r"picosecond, to which the criterion is applied.}" % label),
        r"\label{fig:potim_%s}" % tag,
        r"\end{figure}",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(body))


def _sci(x):
    """Format a number as LaTeX scientific notation."""
    if x == 0:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10.0 ** exp
    return r"$%.2f \times 10^{%d}$" % (mant, exp)


##################################################
# Figure
##################################################


def make_figure(runs, label, stem, dpi):
    """Panel (a): energy per atom relative to its starting value against ionic
    step, so every run occupies the same horizontal range and the smallest
    timestep is visible. Note that the slopes there are per step rather than
    per picosecond.
    Panel (b): slope drift against POTIM, with the criterion bands. Drawn
    only for a sweep, where it is the result of the validation."""
    sweep = len(runs) > 1
    ncols = 2 if sweep else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 4.2))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(runs)))
    for r, c in zip(runs, colors):
        dE = (r["energies"] - r["energies"][0]) / r["natoms"] * 1000.0  # meV
        ax.plot(r["steps"], dE, color=c, linewidth=0.9,
                label="POTIM = %.2f fs" % r["potim"])
        fit = (r["fit_line"] - r["energies"][0]) / r["natoms"] * 1000.0
        ax.plot(r["fit_steps"], fit, color="k", linewidth=1.1,
                linestyle="--", alpha=0.8)

    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_xlabel("Ionic step")
    ax.set_ylabel(r"$E_{\mathrm{tot}} - E_{0}$ per atom (meV)")
    ax.set_title("(a) Energy conservation" if sweep else
                 "Energy conservation, %s" % label)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)

    if sweep:
        ax = axes[1]
        potims = np.array([r["potim"] for r in runs])
        drifts = np.array([r["drift_slope"] for r in runs])
        errs = np.array([r["drift_slope_err"] for r in runs])

        ax.axhspan(1e-12, DRIFT_EXCELLENT, color="tab:green", alpha=0.12)
        ax.axhspan(DRIFT_EXCELLENT, DRIFT_ACCEPTABLE, color="tab:olive",
                   alpha=0.12)
        ax.axhspan(DRIFT_ACCEPTABLE, 1e3, color="tab:red", alpha=0.10)
        ax.axhline(DRIFT_ACCEPTABLE, color="tab:red", linestyle="--",
                   linewidth=1.0, label=r"criterion, $10^{-4}$")

        ax.errorbar(potims, drifts, yerr=errs, marker="o", color="tab:blue",
                    linewidth=1.2, markersize=5, capsize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(potims)
        ax.set_xticklabels(["%g" % p for p in potims])
        ax.set_xlim(potims.min() * 0.8, potims.max() * 1.25)
        ax.minorticks_off()
        ax.set_xlabel("POTIM (fs)")
        ax.set_ylabel("Energy drift (eV/atom/ps)")
        ax.set_title("(b) Drift against timestep")
        ax.set_ylim(min(drifts.min() * 0.3, DRIFT_EXCELLENT * 0.3),
                    max(drifts.max() * 3.0, DRIFT_ACCEPTABLE * 3.0))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle("POTIM validation: %s" % label, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig("%s.%s" % (stem, ext), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


##################################################
# Main
##################################################


def main():
    args = parse_args()

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

    runs = []
    for d in dirs:
        try:
            runs.append(analyse_run(d, args.drop))
        except (FileNotFoundError, ValueError) as exc:
            log.warning("skipping %s: %s", d, exc)

    if not runs:
        log.error("no run could be analysed")
        raise SystemExit(1)

    runs.sort(key=lambda r: r["potim"])

    for r in runs:
        if not r["nve"]:
            log.warning("%s does not look like NVE (%s). Energy drift is not "
                        "a valid timestep test with an active thermostat.",
                        r["directory"],
                        ", ".join("%s=%s" % kv for kv in
                                  sorted(r["ensemble"].items())))

    natoms = {r["natoms"] for r in runs}
    if len(natoms) > 1:
        log.warning("runs have different atom counts %s; check that they are "
                    "the same system", sorted(natoms))

    potims = [r["potim"] for r in runs]
    if len(set(potims)) != len(potims):
        log.warning("repeated POTIM values in the sweep: %s", potims)

    print_report(runs, args.drop)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, "POTIM_%s" % args.label)

    make_figure(runs, args.label, stem, args.dpi)
    write_csv(stem + ".csv", runs)
    write_tex_table(stem + ".tex", runs, args.label)

    written = [stem + ".png", stem + ".pdf", stem + ".csv", stem + ".tex"]
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
