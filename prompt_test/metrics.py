import numpy as np


def calculate_overlap_ratio(pred_keys, true_keys):
    pred_set = set(pred_keys)
    true_set = set(true_keys)
    if not true_set:
        return 0.0, 0, len(pred_set)
    overlap = pred_set.intersection(true_set)
    return len(overlap) / len(true_set), len(true_set - pred_set), len(pred_set - true_set)


def evaluate_dataset(predictions, ground_truths):
    overlaps_relevant, misses_relevant, extras_relevant = [], [], []
    overlaps_utilized, misses_utilized, extras_utilized = [], [], []

    for pred, true in zip(predictions, ground_truths):
        # relevant_sentence_keys
        overlap_r, miss_r, extra_r = calculate_overlap_ratio(
            pred.get('all_relevant_sentence_keys', []), 
            true.get('all_relevant_sentence_keys', [])
        )
        overlaps_relevant.append(overlap_r)
        misses_relevant.append(miss_r)
        extras_relevant.append(extra_r)

        # utilized_sentence_keys
        overlap_u, miss_u, extra_u = calculate_overlap_ratio(
            pred.get('all_utilized_sentence_keys', []), 
            true.get('all_utilized_sentence_keys', [])
        )
        overlaps_utilized.append(overlap_u)
        misses_utilized.append(miss_u)
        extras_utilized.append(extra_u)

    metrics = {
        'relevant_overlap_mean': np.mean(overlaps_relevant),
        'relevant_miss_mean': np.mean(misses_relevant),
        'relevant_extra_mean': np.mean(extras_relevant),
        'utilized_overlap_mean': np.mean(overlaps_utilized),
        'utilized_miss_mean': np.mean(misses_utilized),
        'utilized_extra_mean': np.mean(extras_utilized)
    }
    return metrics