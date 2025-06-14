from pathlib import Path

# ---
# The working directory is $ProjectFileDir$.
DATA_DIR = Path("./data")

# ---
# Magnetic field line trace.
RLIM = 10.0  # Upper limit of the geocentric distance, where the tracing is terminated.
MAXLOOP = 1000  # Maximum number of tracing steps.
