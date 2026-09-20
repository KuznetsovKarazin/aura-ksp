"""Independent numerical core for an audit of SAVED predictions only.

No dataset/model imports, file writes, training, forward passes, or selection.
Caller must verify all manifests/receipts/data/model bindings BEFORE using it
on the final evidence. This module does not certify those bindings.
Fusion deliberately retains the frozen float32 einsum operation order.
Bootstrap uses per-setup sufficient statistics instead of weighted confusion
matrices; all 120 class terms are retained in every macro-F1 replicate.
"""
import numpy as np

N_CLASSES = 120
ENDPOINTS = ("all_top1", "a61_a120_top1", "macro_f1_all120")
MARGINS = (-0.010, -0.015, -0.015)
ALPHA = 0.05 / 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vectors(labels, reference, candidate):
    arrays = tuple(np.asarray(a) for a in (labels, reference, candidate))
    require(all(a.ndim == 1 and a.dtype.kind in "iu" for a in arrays), "Expected integer vectors")
    require(len(arrays[0]) > 0 and all(a.shape == arrays[0].shape for a in arrays), "Length mismatch")
    require(all(np.all((a >= 0) & (a < N_CLASSES)) for a in arrays), "Label/prediction outside 0..119")
    return arrays


def fixed_fusion(stream_logits):
    require(len(stream_logits) == 4, "Exactly four streams required in frozen order")
    values = [np.asarray(x) for x in stream_logits]
    require(all(x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 120
                and x.shape == values[0].shape and np.isfinite(x).all() for x in values),
            "Expected aligned finite float32 [N,120] logits")
    stacked = np.stack(values, axis=1)
    reference = np.einsum("nsc,s->nc", stacked, np.array([.3, .3, .2, .2], dtype=np.float32))
    candidate = np.einsum("nsc,s->nc", stacked[:, :3], np.array([.375, .375, .25], dtype=np.float32))
    return reference, candidate


def class_stats(y, p):
    return (np.bincount(y[y == p], minlength=120),
            np.bincount(y, minlength=120), np.bincount(p, minlength=120))


def f1(tp, true_count, pred_count):
    den = true_count + pred_count
    return np.divide(2.0 * tp, den, out=np.zeros(den.shape, dtype=np.float64), where=den > 0).mean(axis=-1)


def endpoints(y, p):
    y, p, _ = vectors(y, p, p)
    new = (y >= 60) & (y < 120)
    require(new.any(), "A61-A120 support is zero")
    return np.array([(y == p).mean(), (y[new] == p[new]).mean(), f1(*class_stats(y, p))])


def paired_bootstrap(y, reference, candidate, setups, draws, setup_order):
    y, reference, candidate = vectors(y, reference, candidate)
    setups, draws, order = map(np.asarray, (setups, draws, setup_order))
    require(setups.shape == y.shape and setups.dtype.kind in "iu", "Invalid setup vector")
    require(order.ndim == 1 and len(order) > 0 and len(np.unique(order)) == len(order), "Invalid setup order")
    require(np.array_equal(np.unique(setups), np.sort(order)), "Setup coverage mismatch")
    require(draws.ndim == 2 and draws.shape[1] == len(order) and len(draws) > 0 and
            draws.dtype.kind in "iu" and np.all((draws >= 0) & (draws < len(order))), "Invalid draws")
    count = np.zeros((len(draws), len(order)), dtype=np.int64)
    np.add.at(count, (np.arange(len(draws))[:, None], draws), 1)
    all_n, new_n, all_d, new_d, r_stats, c_stats = [], [], [], [], [], []
    difference = (candidate == y).astype(np.int64) - (reference == y).astype(np.int64)
    new = y >= 60
    for s in order:
        m = setups == s
        all_n.append(m.sum()); new_n.append((m & new).sum())
        all_d.append(difference[m].sum()); new_d.append(difference[m & new].sum())
        r_stats.append(class_stats(y[m], reference[m]))
        c_stats.append(class_stats(y[m], candidate[m]))
    new_den = count @ np.array(new_n)
    require(np.all(new_den > 0), "Undefined frozen draw: zero A61-A120 support")
    r_stats, c_stats = np.asarray(r_stats), np.asarray(c_stats)
    result = np.empty((len(draws), 3), dtype=np.float64)
    result[:, 0] = (count @ np.array(all_d)) / (count @ np.array(all_n))
    result[:, 1] = (count @ np.array(new_d)) / new_den
    # Integer weighted sums preserve exact confusion-matrix sufficient statistics.
    for start in range(0, len(draws), 100):
        batch = count[start:start+100]
        rs = [batch @ r_stats[:, j, :] for j in range(3)]
        cs = [batch @ c_stats[:, j, :] for j in range(3)]
        result[start:start+len(batch), 2] = f1(*cs) - f1(*rs)
    return result


def assess(delta, distribution):
    delta, distribution = np.asarray(delta), np.asarray(distribution)
    require(delta.shape == (3,) and distribution.ndim == 2 and distribution.shape[1] == 3
            and len(distribution) > 0 and np.isfinite(delta).all() and np.isfinite(distribution).all(),
            "Invalid endpoint/bootstrap values")
    output = []
    for j, (name, margin) in enumerate(zip(ENDPOINTS, MARGINS)):
        centered = distribution[:, j] - delta[j]
        # Same algebra and rounding order as the frozen centered-basic code.
        lower = float(delta[j] - np.quantile(centered, 1-ALPHA, method="higher"))
        p = float((1 + np.count_nonzero(centered >= delta[j] - margin))/(len(distribution)+1))
        output.append({"endpoint": name, "delta": float(delta[j]), "one_sided_lcb": lower,
                       "margin": margin, "margin_null_p": p,
                       "bonferroni_adjusted_p": min(1., 3*p), "passed": bool(lower > margin)})
    return output
