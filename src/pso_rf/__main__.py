"""``python -m pso_rf``: see :mod:`pso_rf.experiments.cli`."""

import sys

from pso_rf.experiments.cli import main

if __name__ == "__main__":  # guard required on Windows (loky process spawning)
    sys.exit(main())
