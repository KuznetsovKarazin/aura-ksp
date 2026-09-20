#!/usr/bin/env python3
"""Describe fixed AURA-KSP predictions; no inference, refitting, or new tests.

Input is the already audited FINAL_TEST_PAIRED_RESULTS.npz. All 120 classes
and all 120 x 120 directed prediction transitions are retained, including zeros.
Labels in the source arrays are zero based; human-readable action IDs are A001
through A120. No semantic action names are inferred or assigned here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action(label: int) -> str:
    return f"A{int(label) + 1:03d}"


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix(rows: np.ndarray, cols: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros((120, 120), dtype=np.int64)
    np.add.at(out, (rows[mask], cols[mask]), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True,
                        help="Curated research-assets root")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory (must not already exist)")
    args = parser.parse_args()
    check(not args.output.exists(), "Output directory already exists")
    relative = Path("final/run/analysis/FINAL_TEST_PAIRED_RESULTS.npz")
    source = args.assets / relative
    input_hash = sha256(source)

    # This verifies the single analysis input, not the entire historical package.
    source_manifest = args.assets / "MANIFEST_SHA256.txt"
    entries = {}
    for line in source_manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip().removeprefix("*")] = digest
    check(entries.get(relative.as_posix()) == input_hash,
          "Derived input does not match its curated manifest entry")

    with np.load(source, allow_pickle=False) as z:
        y = z["labels"].copy()
        ref = z["reference_predictions"].copy()
        cand = z["candidate_predictions"].copy()
        ids = z["sample_ids"].copy()
        check(np.array_equal(z["reference_logits"].argmax(axis=1), ref),
              "Saved Full-4 decisions disagree with saved logits")
        check(np.array_equal(z["candidate_logits"].argmax(axis=1), cand),
              "Saved Stream-3 decisions disagree with saved logits")
        old_ref_conf = z["reference_confusion"].copy()
        old_cand_conf = z["candidate_confusion"].copy()

    n = len(y)
    check(n == 59477 and y.shape == ref.shape == cand.shape == ids.shape,
          "Unexpected sample count or decision shapes")
    check(len(np.unique(ids)) == n, "Nonunique sample identifiers")
    for values in (y, ref, cand):
        check(np.issubdtype(values.dtype, np.integer) and
              bool(np.all((values >= 0) & (values < 120))), "Invalid class index")
    all_samples = np.ones(n, dtype=bool)
    check(np.array_equal(matrix(y, ref, all_samples), old_ref_conf),
          "Full-4 confusion counts disagree with saved matrix")
    check(np.array_equal(matrix(y, cand, all_samples), old_cand_conf),
          "Stream-3 confusion counts disagree with saved matrix")

    ref_ok, cand_ok = ref == y, cand == y
    events = {
        "both_correct": ref_ok & cand_ok,
        "lost_correct": ref_ok & ~cand_ok,
        "gained_correct": ~ref_ok & cand_ok,
        "both_wrong_same_label": ~ref_ok & ~cand_ok & (ref == cand),
        "both_wrong_different_labels": ~ref_ok & ~cand_ok & (ref != cand),
    }
    check(bool(np.all(np.sum(np.stack(list(events.values())), axis=0) == 1)),
          "Correctness events are not an exhaustive disjoint partition")
    groups = {"all120": all_samples, "A001-A060": y < 60, "A061-A120": y >= 60}
    group_rows, two_by_two = [], []
    for name, mask in groups.items():
        size = int(mask.sum())
        counts = {k: int((v & mask).sum()) for k, v in events.items()}
        both_wrong = counts["both_wrong_same_label"] + counts["both_wrong_different_labels"]
        row = {"group": name, "n": size, **counts,
               "both_wrong": both_wrong,
               "reference_correct": int((ref_ok & mask).sum()),
               "candidate_correct": int((cand_ok & mask).sum()),
               "changed_predicted_label": int(((ref != cand) & mask).sum()),
               "unchanged_predicted_label": int(((ref == cand) & mask).sum()),
               "correctness_changed": counts["lost_correct"] + counts["gained_correct"],
               "net_additional_errors": counts["lost_correct"] - counts["gained_correct"]}
        row.update({
            "label_change_pct": 100.0 * row["changed_predicted_label"] / size,
            "correctness_change_pct": 100.0 * row["correctness_changed"] / size,
            "net_additional_errors_per_1000": 1000.0 * row["net_additional_errors"] / size,
            "delta_accuracy_pp": -100.0 * row["net_additional_errors"] / size,
        })
        group_rows.append(row)
        for rk, ck, count in ((True, True, counts["both_correct"]),
                              (True, False, counts["lost_correct"]),
                              (False, True, counts["gained_correct"]),
                              (False, False, both_wrong)):
            two_by_two.append({"group": name, "reference_correct": rk,
                               "candidate_correct": ck, "n": count})

    total = group_rows[0]
    check((total["lost_correct"], total["gained_correct"],
           total["both_correct"], total["both_wrong"]) == (605, 559, 52392, 5921),
          "Input does not match the established final comparison")

    class_rows = []
    for c in range(120):
        mask = y == c
        support = int(mask.sum())
        counts = {k: int((v & mask).sum()) for k, v in events.items()}
        rc, cc = int((ref_ok & mask).sum()), int((cand_ok & mask).sum())
        class_rows.append({"action_id": action(c), "group": "A001-A060" if c < 60 else "A061-A120",
                           "n": support, **counts, "reference_correct": rc,
                           "candidate_correct": cc,
                           "net_additional_errors": counts["lost_correct"] - counts["gained_correct"],
                           "delta_recall_pp": 100.0 * (cc - rc) / support,
                           "changed_predicted_label": int(((ref != cand) & mask).sum())})

    # True-class/error-class representation permits matched interpretation of the
    # introduction and correction of the same type of classification error.
    lost_by_true_wrong = matrix(y, cand, events["lost_correct"])
    gained_by_true_wrong = matrix(y, ref, events["gained_correct"])
    error_pairs = []
    for truth in range(120):
        for wrong in range(120):
            loss, gain = int(lost_by_true_wrong[truth, wrong]), int(gained_by_true_wrong[truth, wrong])
            error_pairs.append({"true_action_id": action(truth), "wrong_action_id": action(wrong),
                                "lost_correct": loss, "gained_correct": gain,
                                "net_additional_errors": loss - gain})

    # Directed prediction transitions use Full-4 as the source, Stream-3 as the
    # destination; the source is true for losses and the destination true for gains.
    trans = {key: matrix(ref, cand, mask) for key, mask in events.items()}
    transition_rows = []
    for before in range(120):
        for after in range(120):
            counts = {key: int(value[before, after]) for key, value in trans.items()}
            transition_rows.append({"full4_action_id": action(before),
                                    "stream3_action_id": action(after), **counts,
                                    "all_events": sum(counts.values())})
    check(sum(r["all_events"] for r in transition_rows) == n, "Transition count mismatch")
    for key, mask in events.items():
        check(sum(r[key] for r in class_rows) == int(mask.sum()), "Class event count mismatch")

    def top_pairs(key: str) -> list[dict]:
        rows = [r for r in error_pairs if r[key] > 0]
        return sorted(rows, key=lambda r: (-r[key], r["true_action_id"], r["wrong_action_id"]))[:10]

    changed_transitions = [r for r in transition_rows
                           if r["full4_action_id"] != r["stream3_action_id"] and r["all_events"] > 0]
    top_directed = sorted(changed_transitions, key=lambda r: (
        -r["all_events"], r["full4_action_id"], r["stream3_action_id"]))[:10]
    outputs = {
        "correctness_summary.csv": group_rows,
        "correctness_2x2.csv": two_by_two,
        "per_class_error_changes_all120.csv": class_rows,
        "true_error_pairs_all120x120.csv": error_pairs,
        "directed_prediction_transitions_all120x120.csv": transition_rows,
        "top10_directed_label_changes.csv": top_directed,
    }
    args.output.mkdir(parents=True)
    for name, rows in outputs.items():
        write_csv(args.output / name, rows)
    summary = {
        "analysis": "Post-hoc descriptive decomposition of fixed final predictions",
        "analysis_timing": "Descriptive analysis designed after the final test result was known; not a prespecified or confirmatory endpoint",
        "input_relative_path": relative.as_posix(), "input_sha256": input_hash,
        "script_sha256": sha256(Path(__file__)),
        "python": platform.python_version(), "numpy": np.__version__,
        "groups": group_rows,
        "top10_introduced_error_pairs": top_pairs("lost_correct"),
        "top10_corrected_error_pairs": top_pairs("gained_correct"),
        "top10_directed_label_changes": top_directed,
        "class_effect_counts": {
            "recall_decreased": sum(r["net_additional_errors"] > 0 for r in class_rows),
            "recall_unchanged": sum(r["net_additional_errors"] == 0 for r in class_rows),
            "recall_increased": sum(r["net_additional_errors"] < 0 for r in class_rows),
        },
        "restrictions": {
            "new_training": False, "new_test_forward": False, "changed_fusion": False,
            "changed_margins": False, "changed_checkpoints": False, "new_statistical_gates": False,
            "bootstrap_read_or_modified": False, "semantic_names_assigned": False,
            "causal_anatomical_or_topology_claims": False,
        },
        "interpretation": "Descriptive counts only; not independently confirmed subgroup hypotheses. No action names are assigned without a verified official mapping.",
        "direction_note": "Directed matrix: Full-4 predicted label to Stream-3 predicted label. Error-pair matrix: true label and erroneous alternative, separately for introduced and corrected errors.",
        "top_pair_tie_rule": "Count descending, then true action ID and wrong action ID ascending; all tied/untied pairs remain in complete CSV.",
        "top_directed_transition_selection_rule": "Off-diagonal Full-4-to-Stream-3 predicted-label transitions ranked by total count descending, then Full-4 action ID and Stream-3 action ID ascending. First 10 reported descriptively; all 14,400 transitions, including zeros, retained in complete CSV.",
    }
    (args.output / "DESCRIPTIVE_ERROR_ANALYSIS.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    paths = sorted(p for p in args.output.iterdir() if p.is_file())
    (args.output / "MANIFEST_SHA256.txt").write_text(
        "".join(f"{sha256(p)}  {p.name}\n" for p in paths), encoding="utf-8")
    print(json.dumps({"groups": group_rows,
                      "top_introduced": summary["top10_introduced_error_pairs"][:5],
                      "top_corrected": summary["top10_corrected_error_pairs"][:5]}, indent=2))


if __name__ == "__main__":
    main()
