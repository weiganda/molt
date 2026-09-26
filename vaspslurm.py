#!/usr/bin/env python3
"""
PROGRAM: mkslurm.py
PURPOSE: Generates a SLURM job submission script for running VASP on the
         rulisp-lab partition. The job name and output file prefixes are
         derived automatically from the current working directory name.

         The generated script is self-contained: it sources the VASP
         virtual environment and loads the required modules itself rather
         than inheriting them from the submitting shell. Relying on the
         inherited environment makes job success depend on whether the
         `svasp` alias happened to be run in that terminal, which produces
         silent MPI startup hangs (job shows as RUNNING, no OUTCAR, empty
         .o and .e files).

AUTHOR:          Alysse Weigand
CONTRIBUTORS:    Code structure and production refactoring assisted by
                 Claude (Anthropic), March 2026.
LAST MODIFIED:   September 8, 2026

USAGE:
    python mkslurm.py [-n <cores>] [-t <time>] [-m <memory>]
                      [-p <partition>] [-v <vasp_binary>] [-o <filename>]
                      [-l <launcher>]

EXAMPLE:
    python mkslurm.py
    python mkslurm.py -n 32 -t 24:00:00 -m 16G -v vasp_std
    python mkslurm.py -l srun
"""

##################################################
# Imports
##################################################

import os
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
# Defaults
##################################################

DEFAULT_CORES     = 64
DEFAULT_TIME      = "48:00:00"
DEFAULT_MEMORY    = "30G"
DEFAULT_PARTITION = "rulisp-lab,general"
DEFAULT_ACCOUNT   = "rulisp-lab"
DEFAULT_VASP      = "vasp_gam"
DEFAULT_NODES     = 1
DEFAULT_LAUNCHER  = "mpirun"
OUTPUT_FILENAME   = "slurm"

# Environment setup, mirroring the interactive `svasp` alias. Aliases do not
# exist in a non-interactive batch shell, so these must be written into the
# job script explicitly.
#
# NOTE: if $CPG_VENV_VASP is set by your login profile rather than by the
# system module tree, it will be empty inside the batch job. Run
#   echo $CPG_VENV_VASP
# on the login node and paste the literal path here instead of the variable.
VASP_VENV = "$CPG_VENV_VASP"

VASP_MODULES = [
    "mvapich/3.0b_gcc_9.5.0",
    "vasp/5.4.4_mvapich",
]

##################################################
# Helpers
##################################################

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a SLURM job submission script for VASP."
    )

    parser.add_argument("-n", "--cores", type=int, default=DEFAULT_CORES,
                        help=f"Number of MPI tasks (default: {DEFAULT_CORES}).")
    parser.add_argument("-t", "--time",  default=DEFAULT_TIME,
                        help=f"Wall time limit in HH:MM:SS (default: {DEFAULT_TIME}).")
    parser.add_argument("-m", "--mem",   default=DEFAULT_MEMORY,
                        help=f"Memory per node (default: {DEFAULT_MEMORY}).")
    parser.add_argument("-p", "--partition", default=DEFAULT_PARTITION,
                        help=f"SLURM partition (default: {DEFAULT_PARTITION}).")
    parser.add_argument("-v", "--vasp",  default=DEFAULT_VASP,
                        choices=["vasp_gam", "vasp_std", "vasp_ncl"],
                        help=f"VASP binary to run (default: {DEFAULT_VASP}).")
    parser.add_argument("-l", "--launcher", default=DEFAULT_LAUNCHER,
                        choices=["mpirun", "srun"],
                        help=f"MPI launcher (default: {DEFAULT_LAUNCHER}). "
                             f"Use 'srun' if mpirun hangs at startup.")
    parser.add_argument("-o", "--output", default=OUTPUT_FILENAME,
                        help=f"Output filename for the SLURM script "
                             f"(default: {OUTPUT_FILENAME}).")

    args = parser.parse_args()

    if args.cores <= 0:
        parser.error("--cores must be a positive integer.")

    return args


def build_slurm(job_name: str, workdir: str, cores: int, time: str,
                mem: str, partition: str, vasp: str, launcher: str) -> str:
    """
    Build and return the SLURM script contents as a string.

    Parameters
    ----------
    job_name  : str -- Job name (derived from current directory).
    workdir   : str -- Absolute path to the working directory.
    cores     : int -- Number of MPI tasks.
    time      : str -- Wall time limit (HH:MM:SS).
    mem       : str -- Memory per node (e.g. '30G').
    partition : str -- SLURM partition name.
    vasp      : str -- VASP binary to run.
    launcher  : str -- MPI launcher ('mpirun' or 'srun').

    Returns
    -------
    str -- Formatted SLURM script content.
    """
    if launcher == "srun":
        # --mpi=pmi2 matches MVAPICH's PMI bootstrap. Check `srun --mpi=list`
        # if this is rejected on a future Hellbender update.
        launch_cmd = f"srun --mpi=pmi2 -n {cores} {vasp}"
    else:
        launch_cmd = f"mpirun -n {cores} {vasp}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH -p {partition}",
        f"#SBATCH -A {DEFAULT_ACCOUNT}",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -o {job_name}.o%J",
        f"#SBATCH -e {job_name}.e%J",
        f"#SBATCH -N {DEFAULT_NODES}",
        f"#SBATCH -n {cores}",
        f"#SBATCH -t {time}",
        f"#SBATCH --mem={mem}",
        "#",
        "# --- environment (do not rely on the submitting shell) ---",
        f'source {VASP_VENV} || {{ echo "ERROR: could not source VASP venv" >&2; exit 1; }}',
    ]

    for mod in VASP_MODULES:
        lines.append(
            f'module load {mod} || {{ echo "ERROR: module load {mod} failed" >&2; exit 1; }}'
        )

    lines += [
        "export OMP_NUM_THREADS=1",
        "#export MV2_ENABLE_AFFINITY=0",
        "#",
        f"cd {workdir} || exit 1",
        "#",
        "# --- provenance / failure diagnostics ---",
        'echo "host:    $(hostname)"',
        'echo "start:   $(date)"',
        'echo "binary:  $(which ' + vasp + ' || echo NOT_FOUND)"',
        'echo "workdir: $(pwd)"',
        "#",
        launch_cmd,
        "#",
        'echo "exit:    $?"',
        'echo "end:     $(date)"',
    ]
    return "\n".join(lines) + "\n"


def write_slurm(content: str, path: str) -> None:
    """Write SLURM script to file, warning if a file already exists."""
    if os.path.exists(path):
        log.warning("'%s' already exists and will be overwritten.", path)
    with open(path, "w") as f:
        f.write(content)


##################################################
# Main
##################################################

def main() -> None:
    """Generate and write the SLURM submission script."""
    args = parse_args()

    workdir  = os.getcwd()
    job_name = os.path.basename(workdir)

    content = build_slurm(
        job_name=job_name,
        workdir=workdir,
        cores=args.cores,
        time=args.time,
        mem=args.mem,
        partition=args.partition,
        vasp=args.vasp,
        launcher=args.launcher,
    )

    write_slurm(content, args.output)

    log.info("SLURM script written to '%s'.", args.output)
    log.info("  Job name:  %s", job_name)
    log.info("  Partition: %s", args.partition)
    log.info("  Cores:     %d", args.cores)
    log.info("  Time:      %s", args.time)
    log.info("  Memory:    %s", args.mem)
    log.info("  VASP:      %s", args.vasp)
    log.info("  Launcher:  %s", args.launcher)
    log.info("Submit with: sbatch %s", args.output)


if __name__ == "__main__":
    main()
