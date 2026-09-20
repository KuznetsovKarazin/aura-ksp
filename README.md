# AURA-KSP

**Stream reduction for skeleton action recognition**

AURA-KSP studies whether a multi-stream action recognizer can use fewer input streams while retaining prespecified aggregate quality. This repository contains the fixed Full-4 versus Stream-3 comparison on NTU RGB+D 120 XSet: model code, protocols, numerical results and scripts for reproducing the analysis. Companion research assets contain the four final checkpoints, aligned predictions and bootstrap draws.

## Method and results

Both systems use the same retained checkpoints and 64 temporal positions. Full-4 combines joint, bone, joint-motion and bone-motion logits with weights `[0.3, 0.3, 0.2, 0.2]`. Stream-3 removes bone-motion and renormalizes the weights to `[0.375, 0.375, 0.25]`. Fusion uses the fixed float32 weighted sum of logits.

Training used 54,468 retained official-training examples, with the final checkpoint at epoch 65. Evaluation used 59,477 XSet test examples. Differences below are Stream-3 minus Full-4; pp denotes percentage points.

| Endpoint | Full-4 (%) | Stream-3 (%) | Difference (pp) | Lower bound (pp) | Margin (pp) |
|---|---:|---:|---:|---:|---:|
| Overall Top-1 | 89.105032 | 89.027691 | −0.077341 | −0.204138 | −1.0 |
| A61–A120 Top-1 | 87.067234 | 86.866383 | −0.200851 | −0.369294 | −1.5 |
| Macro-F1, all 120 | 89.105911 | 89.023340 | −0.082571 | −0.264688 | −1.5 |

All three prespecified non-inferiority criteria were met. Bounds use 10,000 fixed paired setup-bootstrap draws, centered-basic intervals and Bonferroni alpha 0.05/3. An independent implementation reproduced the complete stored bootstrap array exactly in its recorded environment.

![Prespecified quality criteria](figures/FINAL_NONINFERIORITY.png)

Full-4 correctly classified **52,997** examples and Stream-3 **52,951**: a net loss of **46**. This comprises 605 introduced errors and 559 corrections. Labels changed for 1,778 examples (2.9894%), including 614 changes between two incorrect labels. Class recall decreased for 61 classes, was unchanged for 22 and increased for 37; the largest decrease was A073, −2.459 pp. See the [descriptive error analysis](analysis/decision_changes/README.md).

### Earlier latency measurements

| RTX 4090, batch 1 | Full-4 (ms) | Stream-3 (ms) | Latency reduction |
|---|---:|---:|---:|
| Warm reuse | 65.2755 | 48.9959 | 24.9399% |
| Fresh mapping | 68.5097 | 51.8634 | 24.2977% |

These measurements used earlier seed-271828 models. **The final checkpoints were not timed again.** The reported percentages are latency reductions; throughput increases use a different denominator. Original paired traces are included under `evidence/latency/`.

### Scope

The result concerns this fixed system pair and the prespecified aggregate margins. It does not establish preservation of every class, a new architecture, state-of-the-art accuracy, or generalization across training seeds and devices. A61–A120 is the strongest held-out action subset within NTU120; A1–A60 is connected to earlier NTU60 work. Neither is an external dataset.

The earlier temporal K32 branch failed its required criterion and is retained as negative evidence. Stream selection was exploratory and preceded the frozen final evaluation. See the [research history](docs/RESEARCH_HISTORY.md).

## Reproduce the analysis

Requires Python 3.11 or 3.12. Extract `repository/` and `research-assets/` as sibling directories. From `repository/`:

```bash
python -m pip install -r requirements-audit.txt
python scripts/verify_release.py
python scripts/reproduce_final.py --assets ../research-assets --output .reproduced/final
python scripts/reproduce_latency.py --trace-root evidence/latency/raw --protocol-file evidence/latency/PROTOCOL.json --output .reproduced/latency
python scripts/describe_paired_errors.py --assets ../research-assets --output .reproduced/decision_changes
```

On Windows PowerShell, replace `python` with `py -3.11`. These CPU commands analyze saved predictions and traces. [Reproducibility details](docs/REPRODUCIBILITY.md) describe the checks, figure generation and the scope of the retained training code.

## Repository contents

| Location | Contents |
|---|---|
| `scripts/`, `tests/` | Integrity checks, numerical analysis and figure generation |
| `tables/`, `figures/` | Numerical results and research figures |
| `evidence/` | Frozen protocol, plan, bindings, earlier quality audit and raw latency traces |
| `runtime/` | Original scientific training, preprocessing and model code with source pins |
| `reports/` | Independent numerical reproduction records |
| `analysis/decision_changes/` | Descriptive error decomposition, all 120 classes and label-pair matrices |
| `docs/` | Reproducibility, methods, limitations and research history |
| `../research-assets/` | Four final models, four prediction files, fixed draws, sample metadata, receipts and derived comparison |

Raw NTU recordings, skeleton sequences and preprocessing caches are not redistributed. Benchmark access is governed by the dataset provider's terms.

## Authors and funding

Aida Issembayeva · Oleksandr Kuznetsov · Anargul Shaushenova · Ardak Nurpeisova · Lyazzat Zhumaliyeva · Maral Ongarbayeva.

Affiliations and ORCID identifiers are recorded in [CITATION.cff](CITATION.cff).

This research has been funded by the Science Committee of the Ministry of Science and Higher Education of the Republic of Kazakhstan (Grant No. AP23486538 Research and development of a system for recognizing images in video streams based on artificial intelligence).

## Citation and license

Cite the version used in your research: [AURA-KSP 1.0.1](https://doi.org/10.5281/zenodo.22858090). Machine-readable citation metadata are provided in [CITATION.cff](CITATION.cff).

Research code and companion assets are licensed under **CC BY-NC 4.0**. Original article figures and tables retain the **CC BY 4.0** exception described in [LICENSE_SCOPE.md](LICENSE_SCOPE.md). Third-party components and dependencies retain their own terms; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

[Contributing](CONTRIBUTING.md) · [Reproducibility](docs/REPRODUCIBILITY.md)
