#!/usr/bin/env python3

"""
PROGRAM: validate_relaxation.py
PURPOSE: Decides whether a VASP structural relaxation has actually converged,
         by measuring how the total energy, the maximum force on any atom, and
         the cell volume are still changing at the end of the run. Writes the
         figure, LaTeX table, CSV, and summary needed for the dissertation.

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Conceptual design, purpose, and validation by Alysse Weigand.
                 All scientific reasoning, method choices, and interpretation
                 of results by Alysse Weigand.
                 Code implementation and structure assisted by Claude Opus 5.
                 Parsing and calculations verified 12/20/25.
                 Publication output added 09/05/26.
LAST MODIFIED:   September 2026

USAGE:
    # Analyse OUTCAR in the current directory
    python validate_relaxation.py --label SiO2

    # Point at a specific file and widen the slope window
    python validate_relaxation.py -f run3/OUTCAR -o relax_SiO2 -w 10

    # Match the criteria to the INCAR that produced the run
    python validate_relaxation.py --label SiO2 --energy-req 1e-7 --force-req 1e-5

INPUT:
    OUTCAR   -- required. VASP output from a relaxation run (IBRION 1 or 2).

    The criteria are supplied on the command line rather than read from the
    INCAR, so they must be set to match the run being analysed. Note that
    EDIFFG is negative in the INCAR when it specifies a force criterion;
    --force-req takes the magnitude (an INCAR EDIFFG of -1e-5 corresponds to
    --force-req 1e-5).

OUTPUT (written to --outdir, default relaxation_output/):
    relaxation_data.csv         Per-step energy, max force, g(Force), volume
    relaxation_summary.txt      Human-readable convergence summary
    relaxation_table.tex        LaTeX table fragment for the dissertation
    relaxation_convergence.pdf  Three-panel figure, vector
    relaxation_convergence.png  Same figure, raster at 300 dpi

INTUITION:
    A relaxation is finished when the structure has stopped moving, not when
    the ionic loop happens to exit. VASP will stop on NSW, on EDIFFG, or on a
    line-minimisation failure, and only one of those means convergence. This
    script asks the question directly: over the final few ionic steps, is
    anything still trending?

    Three quantities are tested, each against a criterion the user supplies:

        Energy   slope of the total energy over the final --window steps.
                 A flat energy is necessary but not sufficient, since energy
                 is quadratic near a minimum and goes flat before the forces
                 do.

        Force    the maximum force on any single atom at the LAST ionic step.
                 This is the quantity EDIFFG refers to. It is deliberately not
                 the maximum over the whole run, which would fail a converged
                 relaxation because of a large force in an early step. The
                 slope over the window is checked as well, so a run that is
                 still descending toward the criterion is not passed on the
                 strength of one lucky final step.

        Volume   slope of the cell volume over the window, divided by the mean
                 volume over that window. The criterion is a RELATIVE change
                 per step, so an absolute slope in A^3/step cannot be compared
                 against it directly; a large cell would fail a threshold a
                 small cell passes for identical physics. Only meaningful when
                 the cell was allowed to relax (ISIF 3 or 7).

    A FAIL means the run needs to continue, not that the structure is wrong.
    Restart from CONTCAR and let it proceed far enough to establish a slope.

    Two parsing details worth knowing, both handled here:
      - g(Force) is the optimiser's internal gradient norm and is typically
        orders of magnitude smaller than the max atomic force. It is recorded
        in the CSV for reference but is NOT the convergence criterion.
      - An ionic step is recorded only once both an energy and a force block
        have been seen, so the electronic SCF cycles within a step do not
        each produce a data point.
"""

##################################################
# Imports
##################################################

import re
import os
import csv
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")          # write files rather than opening a window
import matplotlib.pyplot as plt


##################################################
# Arguments
##################################################


def parse_args():
    """Parse and return the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Check convergence of a VASP relaxation and write publication output."
    )
    parser.add_argument("-f", "--file", default="OUTCAR",
                        help="OUTCAR file to analyze (default: OUTCAR).")
    parser.add_argument("-o", "--outdir", default="relaxation_output",
                        help="Directory for output files (default: relaxation_output).")
    parser.add_argument("-w", "--window", type=int, default=5,
                        help="Number of final ionic steps used for the slope (default: 5).")
    parser.add_argument("--label", default="",
                        help="Material name, used in the table caption and titles.")
    parser.add_argument("--energy-req", type=float, default=1e-7,
                        help="Energy convergence criterion, EDIFF from the INCAR (eV).")
    parser.add_argument("--force-req", type=float, default=1e-5,
                        help="Force convergence criterion, EDIFFG from the INCAR (eV/A).")
    parser.add_argument("--volume-req", type=float, default=1e-4,
                        help="Relative volume slope criterion (fraction per step).")
    return parser.parse_args()


##################################################
# Parsing
##################################################


def parse_outcar(path):
    """
    Read one relaxation OUTCAR and return per-ionic-step quantities.

    Returns (total_energies, max_forces, gforces, volumes). The force and
    energy lists are the same length by construction: a step is recorded only
    once both have been seen, so the electronic SCF cycles inside an ionic
    step do not each produce an entry.
    """
    total_energies = []
    max_forces = []      # largest force on any single atom, eV/A (what EDIFFG uses)
    gforces = []         # the optimizer gradient norm reported as g(Force)
    volumes = []

    current_gforce = None
    current_energy = None
    current_maxforce = None

    in_force_block = False
    force_block_rows = []

    with open(path, "r") as f:
        for line in f:

            # --- Energy ---
            if "free energy" in line.lower() and "toten" in line.lower():
                match = re.search(r"=\s*([-\d\.Ee]+)", line)
                if match:
                    current_energy = float(match.group(1))

            # --- Maximum force on any atom, from the TOTAL-FORCE block ---
            # This is the quantity EDIFFG refers to. It is not the same as
            #   g(Force), which is the gradient norm used internally by the
            #   optimizer and is typically orders of magnitude smaller.
            if "TOTAL-FORCE" in line:
                in_force_block = True
                force_block_rows = []
                continue

            if in_force_block:
                if line.strip().startswith("---"):
                    # A dashed line either opens or closes the block.
                    if force_block_rows:
                        forces = np.array(force_block_rows)
                        current_maxforce = float(
                            np.max(np.linalg.norm(forces, axis=1))
                        )
                        in_force_block = False
                    continue
                fields = line.split()
                if len(fields) >= 6:
                    try:
                        force_block_rows.append([float(v) for v in fields[3:6]])
                    except ValueError:
                        pass
                continue

            # --- Optimizer gradient norm ---
            if "g(force)" in line.lower():
                match = re.search(
                    r"g\(Force\)\s*=\s*([-+]?\d*\.\d+(?:[Ee][-+]?\d+)?)",
                    line,
                    re.I
                )
                if match:
                    current_gforce = float(match.group(1))

            # --- Volume ---
            if "volume of cell" in line.lower():
                match = re.search(r"volume of cell\s*[:=]\s*([-\d\.Ee]+)", line, re.I)
                if match:
                    volumes.append(float(match.group(1)))

            # --- Save step when both energy and force are available ---
            # If this is not done, then you get an energy point for each step of
            #   the scf convergence cycle. Not what we want.
            if current_energy is not None and current_maxforce is not None:
                total_energies.append(current_energy)
                max_forces.append(current_maxforce)
                gforces.append(current_gforce)
                current_energy = None
                current_maxforce = None
                current_gforce = None

    # Due to the formatting of the OUTCAR file the initial volume is read in twice
    if len(volumes) > len(total_energies):
        volumes = volumes[1:]

    return total_energies, max_forces, gforces, volumes


##################################################
# Convergence analysis
##################################################


def final_slope(values, n):
    """Absolute value of the linear slope over the final n points."""
    if len(values) < n:
        return None
    y = values[-n:]
    x = np.arange(n)
    slope, _ = np.polyfit(x, y, 1)
    return abs(slope)


def assess_convergence(total_energies, max_forces, volumes, args):
    """
    Apply the three convergence tests and return the results as a dict.

    See the INTUITION section of the module docstring for why each quantity is
    tested the way it is.
    """
    window = args.window

    # --- Energy ---
    energy_slope = final_slope(total_energies, window)
    final_energy = total_energies[-1]

    # --- Force ---
    # NOTE: this is the force at the LAST ionic step, which is what the convergence
    #   criterion refers to. Using the maximum over the whole run would fail a
    #   perfectly converged relaxation because of a large force early on.
    final_force = max_forces[-1] if max_forces else None
    force_slope = final_slope(max_forces, window)

    # --- Volume ---
    # NOTE: the criterion is a RELATIVE change, so the slope is divided by the mean
    #   volume over the window. Comparing an absolute slope in A^3/step against a
    #   dimensionless threshold would be a unit mismatch.
    volume_slope_abs = final_slope(volumes, window)
    if volume_slope_abs is not None and len(volumes) >= window:
        volume_slope_rel = volume_slope_abs / np.mean(volumes[-window:])
    else:
        volume_slope_rel = None

    rel_delta_volume = ((volumes[-1] - volumes[0]) / volumes[0]) if len(volumes) > 1 else None
    final_volume = volumes[-1] if volumes else None

    # --- Convergence checks ---
    energy_converged = energy_slope is not None and energy_slope < args.energy_req
    force_converged = (final_force is not None and final_force < args.force_req) \
                      and (force_slope is None or force_slope < args.force_req)
    volume_converged = volume_slope_rel is not None and volume_slope_rel < args.volume_req

    return {
        "energy_slope": energy_slope,
        "final_energy": final_energy,
        "final_force": final_force,
        "force_slope": force_slope,
        "volume_slope_abs": volume_slope_abs,
        "volume_slope_rel": volume_slope_rel,
        "rel_delta_volume": rel_delta_volume,
        "final_volume": final_volume,
        "energy_converged": energy_converged,
        "force_converged": force_converged,
        "volume_converged": volume_converged,
        "all_converged": energy_converged and force_converged and volume_converged,
    }


##################################################
# Output: data table
##################################################


def write_csv(path, total_energies, max_forces, gforces, volumes):
    """Write the per-ionic-step data table."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ionic_step", "total_energy_eV", "max_force_eV_per_A",
                         "g_force", "volume_A3"])
        for i in range(len(total_energies)):
            vol = volumes[i] if i < len(volumes) else ""
            gf = gforces[i] if i < len(gforces) and gforces[i] is not None else ""
            writer.writerow([i + 1, total_energies[i], max_forces[i], gf, vol])


##################################################
# Output: summary
##################################################


def fmt(x, spec=".6e"):
    """Format a number for the text summary, or N/A if it could not be computed."""
    return "N/A" if x is None else format(x, spec)


def build_summary(results, nsteps, args):
    """Return the human-readable convergence summary as a single string."""
    r = results
    lines = []
    lines.append("=" * 60)
    lines.append("RELAXATION CONVERGENCE SUMMARY")
    if args.label:
        lines.append(f"Material: {args.label}")
    lines.append(f"Source file: {args.file}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Ionic steps completed:        {nsteps}")
    lines.append(f"Slope window:                 final {args.window} steps")
    lines.append("")
    lines.append("Convergence criteria")
    lines.append(f"  Energy (EDIFF):             {args.energy_req:.1e} eV")
    lines.append(f"  Force (EDIFFG):             {args.force_req:.1e} eV/A")
    lines.append(f"  Volume (relative):          {args.volume_req:.1e} per step")
    lines.append("")
    lines.append("Final values")
    lines.append(f"  Total energy:               {r['final_energy']:.6f} eV")
    lines.append(f"  Max force on any atom:      {fmt(r['final_force'])} eV/A")
    lines.append(f"  Cell volume:                {fmt(r['final_volume'], '.4f')} A^3")
    lines.append(f"  Total relative dV:          {fmt(r['rel_delta_volume'], '.6e')}")
    lines.append("")
    lines.append(f"Slopes over the final {args.window} steps")
    lines.append(f"  Energy:                     {fmt(r['energy_slope'])} eV/step     {'PASS' if r['energy_converged'] else 'FAIL'}")
    lines.append(f"  Max force:                  {fmt(r['force_slope'])} eV/A/step")
    lines.append(f"  Volume (absolute):          {fmt(r['volume_slope_abs'])} A^3/step")
    lines.append(f"  Volume (relative):          {fmt(r['volume_slope_rel'])} per step  {'PASS' if r['volume_converged'] else 'FAIL'}")
    lines.append("")
    lines.append("Convergence status")
    lines.append(f"  Energy converged:           {'Yes' if r['energy_converged'] else 'No'}")
    lines.append(f"  Force converged:            {'Yes' if r['force_converged'] else 'No'}")
    lines.append(f"  Volume converged:           {'Yes' if r['volume_converged'] else 'No'}")
    lines.append("")
    if r["all_converged"]:
        lines.append("RESULT: Relaxation complete. Energy, forces, and volume are converged.")
    else:
        lines.append("RESULT: Relaxation not fully converged. Consider continuing the")
        lines.append("        relaxation or tightening the INCAR parameters so that the run")
        lines.append("        proceeds for enough additional steps to establish a slope.")
    lines.append("=" * 60)
    return "\n".join(lines)


##################################################
# Output: LaTeX table fragment
##################################################


def tex(x, spec=".2e"):
    """Format a number for LaTeX, as a x 10^b where appropriate."""
    if x is None:
        return "N/A"
    s = format(x, spec)
    if "e" in s:
        mant, exp = s.split("e")
        return f"${mant} \\times 10^{{{int(exp)}}}$"
    return f"${s}$"


def write_tex_table(path, results, args):
    """Write the LaTeX table fragment for the dissertation."""
    r = results
    yes_no = lambda b: "Yes" if b else "No"
    caption_label = f" for {args.label}" if args.label else ""

    lines = [
        r"\begin{table}[!h]",
        r"\centering",
        r"\resizebox{0.75\textwidth}{!}{",
        r"\begin{tabular}{|l|c|c|c|}",
        r"\hline",
        r"\textbf{Quantity} & \textbf{Final value} & \textbf{Slope} & \textbf{Converged} \\",
        r"\hline",
        f"Total energy (eV) & ${r['final_energy']:.4f}$ & {tex(r['energy_slope'])} & {yes_no(r['energy_converged'])} \\\\",
        r"\hline",
        f"Max force (eV/\\AA) & {tex(r['final_force'])} & {tex(r['force_slope'])} & {yes_no(r['force_converged'])} \\\\",
        r"\hline",
        f"Cell volume (\\AA$^3$) & {tex(r['final_volume'], '.3f')} & {tex(r['volume_slope_rel'])} & {yes_no(r['volume_converged'])} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"}",
        f"\\caption{{Relaxation convergence{caption_label}. "
        f"Slopes are taken over the final {args.window} ionic steps; the volume slope is relative.}}",
        r"\label{tbl:relaxation_convergence}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


##################################################
# Output: figure
##################################################


def plain_axis(ax, values):
    """
    Force full numeric tick labels instead of matplotlib's offset notation,
    choosing enough decimal places to resolve the range of the data.
    """
    from matplotlib.ticker import FormatStrFormatter
    span = max(values) - min(values) if len(values) > 1 else 0.0
    if span > 0:
        decimals = max(0, int(math.ceil(-math.log10(span))) + 2)
    else:
        decimals = 4
    decimals = min(decimals, 10)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.yaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))


def make_figure(total_energies, max_forces, volumes, args, pdf_path, png_path):
    """
    Three stacked panels sharing an ionic-step axis: total energy, maximum
    force on a log scale with the criterion marked, and cell volume. The slope
    window is shaded on every panel so the reader can see which steps the
    convergence numbers were taken from.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    })

    steps_energy = list(range(1, len(total_energies) + 1))
    steps_forces = list(range(1, len(max_forces) + 1))
    steps_volume = list(range(1, len(volumes) + 1))

    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)

    # --- Energy ---
    axs[0].plot(steps_energy, total_energies, marker="s", markersize=3,
                color="tab:blue", linewidth=1)
    axs[0].set_ylabel("Total energy (eV)")
    plain_axis(axs[0], total_energies)

    # --- Max force ---
    axs[1].plot(steps_forces, max_forces, marker="o", markersize=3,
                color="tab:red", linewidth=1, label="Max force")
    axs[1].axhline(args.force_req, color="k", linestyle="--", linewidth=1,
                   label=f"Criterion ({args.force_req:.0e})")
    axs[1].set_ylabel(r"Max force (eV/$\mathrm{\AA}$)")
    axs[1].set_yscale("log")
    axs[1].legend(loc="best", frameon=False)

    # --- Volume ---
    if volumes:
        axs[2].plot(steps_volume, volumes, marker="^", markersize=3,
                    color="tab:green", linewidth=1)
        axs[2].set_ylabel(r"Cell volume ($\mathrm{\AA}^3$)")
        plain_axis(axs[2], volumes)

    axs[2].set_xlabel("Ionic step")

    # Shade the slope window on every panel
    for ax in axs:
        if len(total_energies) >= args.window:
            ax.axvspan(len(total_energies) - args.window + 1, len(total_energies),
                       color="grey", alpha=0.15, zorder=0)

    fig.align_ylabels(axs)
    plt.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


##################################################
# Main
##################################################


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    out = lambda name: os.path.join(args.outdir, name)

    total_energies, max_forces, gforces, volumes = parse_outcar(args.file)

    if not total_energies:
        raise SystemExit(f"No ionic steps found in '{args.file}'. Check the file path.")

    results = assess_convergence(total_energies, max_forces, volumes, args)

    write_csv(out("relaxation_data.csv"),
              total_energies, max_forces, gforces, volumes)

    summary_text = build_summary(results, len(total_energies), args)
    print("\n" + summary_text)
    with open(out("relaxation_summary.txt"), "w") as f:
        f.write(summary_text + "\n")

    write_tex_table(out("relaxation_table.tex"), results, args)

    make_figure(total_energies, max_forces, volumes, args,
                out("relaxation_convergence.pdf"),
                out("relaxation_convergence.png"))

    print(f"\nOutput written to '{args.outdir}':")
    for name in ["relaxation_data.csv", "relaxation_summary.txt",
                 "relaxation_table.tex", "relaxation_convergence.pdf",
                 "relaxation_convergence.png"]:
        print(f"  {name}")


if __name__ == "__main__":
    main()
