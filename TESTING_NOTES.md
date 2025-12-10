# Testing Notes

Attempted to run `python scripts/4_run_final_model_comparison.py` to reproduce model comparison results, but the environment lacks required Python dependencies.

* `pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn` failed because outbound HTTP(S) requests are blocked by a proxy (`403 Forbidden`), so no wheels could be downloaded.
* `apt-get update` failed with the same proxy restrictions (`403 Forbidden` for ubuntu and llvm repositories), preventing installation of system packages like `python3-pandas`.
* Running `python scripts/4_run_final_model_comparison.py` currently raises `ModuleNotFoundError: No module named 'pandas'` because the dependencies above remain unavailable in this environment.

If you have an environment with network access, install the listed packages (or create the conda env from `environment.yml`) before rerunning the script; the project data files are already present.

For a step-by-step walkthrough on setting up a clean environment and executing the scripts, see `docs/LOCAL_TESTING.md`.
