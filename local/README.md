# Local Workspace

This directory is for machine-local work that should not be committed:

- `sbatch/`: temporary or exploratory Slurm scripts;
- `logs/`: Slurm standard output and error logs.

Both subdirectories retain only their `.gitignore` files. Submit the tracked
Slurm scripts from the repository root so their relative log paths resolve to
`local/logs/`.
