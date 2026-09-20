# Descriptive decomposition of the fixed final predictions

This is a post-hoc descriptive analysis of saved AURA-KSP final decisions. It changes no model, fusion rule, tolerance, endpoint, or bootstrap procedure, and executes no training or test forward. It does not conduct new significance tests. It does not infer explanations about joints, attention, or graph topology.

## Reproduction

From the repository root (place the companion research assets next to the repository):

```bash
python analysis/decision_changes/describe_paired_errors.py --assets ../research-assets --output .reproduced/decision_changes
```

Only Python and NumPy are required. Choose a fresh output directory: the script refuses to overwrite previous output. The input is the already audited `final/run/analysis/FINAL_TEST_PAIRED_RESULTS.npz` under the supplied assets directory. The script verifies that single file against the assets manifest, confirms the saved argmax decisions and confusion counts, and records the source and script SHA-256. It does not repeat the earlier record/provenance audit or inspect bootstrap values.

## Output interpretation

- `correctness_summary.csv`: exhaustive event counts for all samples and the two true-label groups A001–A060 and A061–A120.
- `correctness_2x2.csv`: four correctness combinations, separately for all three populations.
- `per_class_error_changes_all120.csv`: all 120 true action classes, in action-ID order.
- `true_error_pairs_all120x120.csv`: each true-class/error-class pair, separately counting errors introduced and errors corrected by Stream-3; all 14,400 pairs remain, including zeros.
- `directed_prediction_transitions_all120x120.csv`: Full-4 predicted class → Stream-3 predicted class, partitioned by correctness event; all 14,400 transitions remain.
- `top10_directed_label_changes.csv`: the ten largest off-diagonal predicted-label transitions, ordered by total count descending, then Full-4 and Stream-3 action IDs ascending. This selection was made after the final test result was known and is descriptive only.
- `DESCRIPTIVE_ERROR_ANALYSIS.json`: input provenance, summaries, top observed introduced/corrected error pairs, and explicit analysis restrictions.
- `MANIFEST_SHA256.txt`: output hashes.

The two pair tables use different axes. For example, the directed predicted-label change A073 → A076 consists of 14 introduced errors on true A073 samples, 18 corrected errors on true A076 samples, and one sample that remains wrong. In the true/error table those two correctness changes occupy `(true A073, wrong A076)` and `(true A076, wrong A073)`, respectively.

Action IDs follow the source labels plus one. The numerical tables use IDs only. For optional prose labels, `OFFICIAL_ACTION_NAME_CHECK.json` records a check of two names against the official ROSE Lab action-class table on 20 September 2026: A073 is "staple book" and A076 is "cutting paper". This semantic lookup is separate from the numerical calculation.

## Exact descriptive results

| Population | Both correct | Full-4 correct only | Stream-3 correct only | Both wrong | Both wrong, same label | Both wrong, different labels |
|---|---:|---:|---:|---:|---:|---:|
| All 120 actions | 52,392 | 605 | 559 | 5,921 | 5,307 | 614 |
| A001–A060 | 27,178 | 243 | 256 | 2,425 | 2,215 | 210 |
| A061–A120 | 25,214 | 362 | 303 | 3,496 | 3,092 | 404 |

The global net difference of 46 errors is the difference between 605 introduced errors and 559 corrected errors. It should not be described as only 46 changed decisions: 1,778 labels changed (2.9894%), including 614 changes between two incorrect labels. Correctness changed for 1,164 samples (1.9571%).

For A061–A120, 362 introduced errors and 303 corrected errors yield 59 additional errors. For A001–A060, 243 introduced and 256 corrected errors yield 13 fewer errors. These subgroup decompositions are descriptive and do not constitute new confirmed subgroup hypotheses.

The largest observed directed label change is A073 → A076 (33 samples), illustrating that the same shift can help one true class while harming another. This is an output-level observation; it does not establish a biomechanical or graph-topology mechanism.

## Suggested main-text paragraph

> The small aggregate accuracy difference concealed a larger redistribution of individual decisions. The two systems agreed on the correctness of 58,313 examples, comprising 52,392 jointly correct and 5,921 jointly incorrect predictions. Removing bone motion introduced 605 errors and corrected 559 reference errors. Predicted labels changed for 1,778 examples (2.9894%); 614 of these changes were between two incorrect labels. Thus, the net cost of 46 additional errors should not be interpreted as near-identical predictions.

> This redistribution was not uniform across the two action groups. A61–A120 accounted for 362 introduced and 303 corrected errors, whereas A1–A60 had 243 introduced and 256 corrected errors. The largest observed directed change, A073 to A076, harmed 14 examples of the former class and corrected 18 examples of the latter, with one further example remaining incorrect. These post-hoc counts describe the balance of errors without assigning a causal explanation to skeleton topology or motion features.

No confidence bounds or inferential conclusions are attached to the new descriptive slices. The original frozen quality criteria remain the sole confirmatory decision rule.
