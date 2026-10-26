"""Run the whole project end to end (about 6 minutes on a laptop).

    python run_all.py

Each stage writes its outputs to data/, figures/, reports/ or results/.
The optional multi-seed simulation study (about 25 minutes) is not included;
run it separately with  python src/simulation_study.py
"""

import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pymc  # noqa: E402,F401  (imported first so its log level can be set in main)
import data_generation  # noqa: E402
import decision  # noqa: E402
import fit_models  # noqa: E402
import prior_checks  # noqa: E402
import sequential  # noqa: E402
import validate  # noqa: E402

STAGES = [
    ("1. Simulate the portfolio", data_generation.main),
    ("2. Prior predictive checks", prior_checks.main),
    ("3. Fit the three models", fit_models.main),
    ("4. Validate", validate.main),
    ("5. Month-by-month updating", sequential.main),
    ("6. Credit limit decision", decision.main),
]


def main():
    warnings.filterwarnings("ignore")
    logging.getLogger("pymc").setLevel(logging.ERROR)
    start = time.time()
    for name, stage in STAGES:
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        stage()
        print(f"--- {name}: {time.time() - t:.0f}s", flush=True)
    print(f"\nAll stages finished in {(time.time() - start) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
