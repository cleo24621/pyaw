from pathlib import Path

# ---
# The working directory is $ProjectFileDir$.
DATA_DIR = Path(
    r"D:\_Projects\PycharmProjects\PyAW\data"
)  # Modify the base data directory path as needed.

# ---
# Magnetic field line trace.
RLIM = 10.0  # Upper limit of the geocentric distance, where the tracing is terminated.
MAXLOOP = 1000  # Maximum number of tracing steps.
