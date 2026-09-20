# Reproducibility

## Integrity

`python scripts/verify_release.py` checks the repository manifest. `python scripts/verify_release.py --root ../research-assets` checks the companion assets. Verify the SHA-256 sidecars before extracting downloaded ZIPs. Git attributes preserve source line endings.

## Final saved-output analysis

Install `requirements-audit.txt` and run:

```bash
python scripts/reproduce_final.py --assets ../research-assets --output .reproduced/final
```

The script checks the assets manifest, receipt outputs, model and event pins, sample alignment, fixed fusion, the frozen draw hash, all endpoints and every stored derived array. Checkpoint serialization is inspected by a restricted inert decoder, without importing PyTorch or executing a model. The output JSON records numerical comparisons and environment versions. Allow approximately 2 GB available RAM, plus storage for the extracted assets; this is an estimate, not a measured peak.

The independent math tests compare sufficient-statistic bootstrap calculations against literal repeated examples on synthetic data. Run `python -m unittest discover -s tests -v`; only NumPy is needed.

## Descriptive error analysis

```bash
python scripts/describe_paired_errors.py --assets ../research-assets --output .reproduced/decision_changes
```

This reports introduced and corrected errors, all 120 classes and complete label-pair matrices from the saved decisions. See [output interpretation](../analysis/decision_changes/README.md). Choose a fresh output directory; the script does not overwrite existing output.

## Latency reanalysis

```bash
python scripts/reproduce_latency.py --trace-root evidence/latency/raw --protocol-file evidence/latency/PROTOCOL.json --output .reproduced/latency
```

The script processes the original 18 paired traces: three folds, two regimes and three replicates. It recomputes saved measurements from the earlier seed-271828 models. It does not measure the final checkpoints or current hardware.

## Figures and tables

```bash
python -m pip install -r requirements-figures.txt
python scripts/build_paper_assets.py --output .reproduced/figures
```

Figures are generated from the verified numerical tables into the selected output directory. LaTeX table fragments are written under its `tables/` subdirectory. The source tables and distributed figures remain unchanged.

## Training and data

`runtime/` contains the original model modules, preprocessing helpers, final scientific scripts and PLAN configurations. These preserve the experiment's semantics and execution checks. The historical runtime manifest describes the original runtime kit, rather than the curated `runtime/` directory.

Raw NTU data, the 1.6 GB sparse frame store and its complete preprocessing environment are not bundled. Saved-output analysis is reproducible with this release; end-to-end training from raw data has not been independently checked as a portable workflow. The retained training implementation supports inspection of the experiment. Benchmark data must be obtained from the provider under its access terms.

The final training environment used PyTorch 2.11.0+cu128 and an RTX 4090; exact configurations are in `evidence/PLAN.json`. The analysis reference environment used Python 3.12.13 and NumPy 2.3.5. The floating-point portability tolerance for bootstrap values is 2e-15; exact equality is reported separately.

CSV fractions use the 0–1 scale unless a field name explicitly indicates pct or pp. Multiply fractions by 100 to express percentages or percentage points.
