# Tourette–Neupulse Model

An independent research project inspired by Neupulse. This repository is not an official Neupulse product or software release.

Research code exploring rhythmic median nerve stimulation through a forced Kuramoto coupled oscillator model, including an algorithm for individualized stimulation frequency selection.

## Contents

- `fkm/individualize.py`: aperiodic PSD fitting, mu-peak estimation, uncertainty-aware shrinkage, frequency quantization, and fallback behavior.
- `fkm/core.py`: analytic forced oscillator model.
- `fkm/numerics.py`: oscillator simulations, circular cumulants, and autocorrelation estimators.
- `fkm/observation.py` and `fkm/waveform.py`: observation and stimulation waveform models.
- `scripts/`: demonstrations and numerical validation.
- `tests/`: recovered test suite.
- `paper/`: Chinese manuscript sources and supplied PDF.
- `docs/`: original research notes and extraction details.

## Setup and quick start

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/ -m "not slow"
PYTHONPATH=. python scripts/individualize_demo.py
```

Run the full test suite with `python -m pytest -q tests/`. Some simulation tests can take longer.

## Recovery and validation status

This repository was reconstructed from file creation and text-edit records in three exported research conversations. All recorded text edits matched, and every recovered Python file passed syntax parsing. Before publication, the recovered fast test suite was rerun: **122 passed, 11 deselected** (`python -m pytest -q tests/ -m "not slow"`). The 11 slow simulation tests were not rerun. Numerical conclusions in the original research notes remain historical reports rather than a fresh reproduction of all simulations.

The numerical result caches `results/m2_oa.json`, `results/m2_spont.json`, and `results/m2_driven.json` were not recoverable from file paths alone. Regenerate the simulation results before using the cached report mode of `scripts/m2_validation.py`; consult that script's command-line help.

The PDF was supplied separately and was not rebuilt from the recovered LaTeX sources. Code, manuscript, and historical notes may describe different stages of the research.

This is exploratory modeling code. Model-derived gains and synthetic cohort results are not clinical validation or treatment recommendations.
