# Reproduction levels

## 1. Integrity
`python scripts/verify_release.py` checks the exact release manifest. `python scripts/verify_release.py --root ../research-assets` verifies the larger companion files. SHA-256 sidecars verify distributed ZIPs before extraction. Git attributes preserve source line endings.

## 2. Final saved-output analysis (CPU)
Install `requirements-audit.txt` and run the README command. The script checks the assets manifest; receipt outputs and model/event pins; sample alignment; fixed fusion; frozen draw hash; all endpoints; and every stored derived array. Checkpoint serialization is inspected by a restricted inert decoder, without importing PyTorch or executing a model. A successful run writes JSON with comparisons and environment versions. Allocate approximately 2 GB available RAM for the analysis, plus room for the extracted assets; this is an estimate, not a measured peak.

The independent math tests use synthetic examples and compare sufficient-statistic bootstrap calculations with literal repeated examples. Run `python -m unittest discover -s tests -v`; only NumPy is needed.

## 3. Latency reanalysis (CPU)
The README command processes the original 18 paired traces (three folds, two regimes, three replicates). It recomputes saved measurements, not fresh hardware latency. Historical generated reports refer to the next stage as of that experiment; the current project status is in README.

## 4. Figures
Install requirements-figures.txt, then `python scripts/build_paper_assets.py`. This regenerates figures and LaTeX table fragments from verified tables. It changes derived file hashes, so do it in a working copy. Full article submission sources remain a separate manuscript deliverable.

## 5. Training and data reconstruction
The exact model modules, preprocessing helpers, final scientific scripts and PLAN configs are included under runtime. They preserve experiment semantics, including fail-closed execution checks. Provider controllers, private authorization records and obsolete launchers were deliberately omitted. The old runtime manifest describes the **original** kit and is provenance, not a claim that this curated folder is the complete launch kit. Do not run it as a fresh cloud launcher.

End-to-end training from raw NTU data is not certified by this public release. Raw datasets, the 1.6 GB sparse frame store and its full preprocessing environment are not bundled. This release certifies saved-output reproduction; it preserves the original training implementation for inspection. A portable training entry point and a complete, independently checked preprocessing environment remain follow-up work before claiming turnkey training reproducibility. Release licensing and the retained CTR-GCN attribution are documented in LICENSE and THIRD_PARTY_NOTICES.md. Reproducing independent training should use new directories and a separately identified replication protocol, not mutate the sealed original test event.

Original final training environment: PyTorch 2.11.0+cu128, RTX 4090; see evidence/PLAN.json for exact configurations. Analysis reference environment: Python 3.12.13, NumPy 2.3.5. Floating point portability tolerance for bootstrap is 2e-15; exact equality is reported separately.

CSV fractions use the 0–1 scale unless field names explicitly say pct/pp. Convert to percent/percentage points by multiplying by 100.
