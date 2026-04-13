# metrics.py
from typing import Dict, List, Optional
import torch

THRESHOLD_CANDIDATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
def compute_trace_metrics_inference(logits, masks, threshold=0.3):

    rel_pred  = (torch.sigmoid(logits['logits_relevance'].detach().cpu())  > threshold)
    util_pred = (torch.sigmoid(logits['logits_utilization'].detach().cpu())> threshold)
    adh_pred  = (torch.sigmoid(logits['logits_adherence'].detach().cpu())   > threshold)

    ctx_m  = masks['context_mask'].detach().cpu()
    resp_m = masks['response_mask'].detach().cpu()

    def rate(pred, mask):
        # sum(pred & mask) / sum(mask)
        num = (pred & mask).sum(dim=1).float()
        den = mask.sum(dim=1).float().clamp(min=1)
        return num.div(den)

    relevance_rate   = rate(rel_pred,  ctx_m)
    utilization_rate = rate(util_pred, ctx_m)
    adherence_rate   = rate(adh_pred,  resp_m)

    num_ru = (rel_pred & util_pred & ctx_m).sum(dim=1).float()
    den_r  = rel_pred.sum(dim=1).float().clamp(min=1)
    completeness = num_ru.div(den_r)

    return {
        'relevance_rate':   relevance_rate,    
        'utilization_rate': utilization_rate,  
        'adherence_rate':   adherence_rate,
        'completeness':     completeness
    }
def compute_classification_metrics(logits: Dict[str, torch.Tensor],
                                   labels: Dict[str, torch.Tensor],
                                   masks: Dict[str, torch.Tensor],
                                   threshold: float = 0.3) -> Dict[str, float]:
    rel_pred  = (torch.sigmoid(logits['logits_relevance'])  > threshold) & masks['context_mask']
    util_pred = (torch.sigmoid(logits['logits_utilization'])> threshold) & masks['context_mask']
    adh_pred  = (torch.sigmoid(logits['logits_adherence'])   > threshold) & masks['response_mask']

    rel_true  = (labels['labels_relevance']   == 1) & masks['context_mask']
    util_true = (labels['labels_utilization']== 1) & masks['context_mask']
    adh_true  = (labels['labels_adherence']   == 1) & masks['response_mask']

    def prf(pred, true):
        tp = (pred & true).sum().float()
        fp = (pred & ~true).sum().float()
        fn = (~pred & true).sum().float()
        prec = tp.div((tp + fp).clamp(min=1))
        rec  = tp.div((tp + fn).clamp(min=1))
        f1   = 2 * prec * rec / (prec + rec).clamp(min=1)
        return float(prec), float(rec), float(f1)

    rel_p,  rel_r,  rel_f1  = prf(rel_pred,  rel_true)
    util_p, util_r, util_f1 = prf(util_pred, util_true)
    adh_p,  adh_r,  adh_f1  = prf(adh_pred,  adh_true)

    return {
        'relevance_precision':    rel_p,
        'relevance_recall':       rel_r,
        'relevance_f1':           rel_f1,
        'utilization_precision':  util_p,
        'utilization_recall':     util_r,
        'utilization_f1':         util_f1,
        'adherence_precision':    adh_p,
        'adherence_recall':       adh_r,
        'adherence_f1':           adh_f1,
    }


def compute_classification_metrics_stream(
    all_logits: List[Dict[str, torch.Tensor]],
    all_labels: List[Dict[str, torch.Tensor]],
    all_masks:  List[Dict[str, torch.Tensor]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Вариант, который работает по всему датасету.
    Здесь мы сначала считаем предсказания/истину для каждого батча,
    затем разворачиваем их в вектор и конкатим по всем батчам.
    """
    rel_pred_list, rel_true_list = [], []
    util_pred_list, util_true_list = [], []
    adh_pred_list, adh_true_list = [], []

    for logits, labels, masks in zip(all_logits, all_labels, all_masks):
        rel_pred  = (torch.sigmoid(logits['logits_relevance'])   > threshold) & masks['context_mask']
        util_pred = (torch.sigmoid(logits['logits_utilization']) > threshold) & masks['context_mask']
        adh_pred  = (torch.sigmoid(logits['logits_adherence'])   > threshold) & masks['response_mask']

        rel_true  = (labels['labels_relevance']    == 1) & masks['context_mask']
        util_true = (labels['labels_utilization']  == 1) & masks['context_mask']
        adh_true  = (labels['labels_adherence']    == 1) & masks['response_mask']

        rel_pred_list.append(rel_pred.reshape(-1))
        util_pred_list.append(util_pred.reshape(-1))
        adh_pred_list.append(adh_pred.reshape(-1))

        rel_true_list.append(rel_true.reshape(-1))
        util_true_list.append(util_true.reshape(-1))
        adh_true_list.append(adh_true.reshape(-1))

    rel_pred  = torch.cat(rel_pred_list,  dim=0)
    util_pred = torch.cat(util_pred_list, dim=0)
    adh_pred  = torch.cat(adh_pred_list,  dim=0)

    rel_true  = torch.cat(rel_true_list,  dim=0)
    util_true = torch.cat(util_true_list, dim=0)
    adh_true  = torch.cat(adh_true_list,  dim=0)

    def prf(pred, true):
        tp = (pred & true).sum().float()
        fp = (pred & ~true).sum().float()
        fn = (~pred & true).sum().float()

        prec = tp.div((tp + fp).clamp(min=1))
        rec  = tp.div((tp + fn).clamp(min=1))
        f1   = 2 * prec * rec / (prec + rec).clamp(min=1)
        return float(prec), float(rec), float(f1)

    rel_p,  rel_r,  rel_f1  = prf(rel_pred,  rel_true)
    util_p, util_r, util_f1 = prf(util_pred, util_true)
    adh_p,  adh_r,  adh_f1  = prf(adh_pred,  adh_true)

    return {
        'relevance_precision':    rel_p,
        'relevance_recall':       rel_r,
        'relevance_f1':           rel_f1,
        'utilization_precision':  util_p,
        'utilization_recall':     util_r,
        'utilization_f1':         util_f1,
        'adherence_precision':    adh_p,
        'adherence_recall':       adh_r,
        'adherence_f1':           adh_f1,
    }


def compute_metrics_with_thresholds(
    all_logits,
    all_labels,
    all_masks,
    thresholds,
):
    """Like compute_classification_metrics_stream but with per-target thresholds.

    Args:
        thresholds: {"relevance": t_rel, "utilization": t_util, "adherence": t_adh}
                    Missing keys fall back to 0.5.
    """
    t_rel  = thresholds.get("relevance",   0.5)
    t_util = thresholds.get("utilization", 0.5)
    t_adh  = thresholds.get("adherence",   0.5)

    rel_pred_list,  rel_true_list  = [], []
    util_pred_list, util_true_list = [], []
    adh_pred_list,  adh_true_list  = [], []

    for logits, labels, masks in zip(all_logits, all_labels, all_masks):
        rel_pred  = (torch.sigmoid(logits['logits_relevance'])   > t_rel)  & masks['context_mask']
        util_pred = (torch.sigmoid(logits['logits_utilization']) > t_util) & masks['context_mask']
        adh_pred  = (torch.sigmoid(logits['logits_adherence'])   > t_adh)  & masks['response_mask']

        rel_true  = (labels['labels_relevance']   == 1) & masks['context_mask']
        util_true = (labels['labels_utilization'] == 1) & masks['context_mask']
        adh_true  = (labels['labels_adherence']   == 1) & masks['response_mask']

        rel_pred_list.append(rel_pred.reshape(-1))
        util_pred_list.append(util_pred.reshape(-1))
        adh_pred_list.append(adh_pred.reshape(-1))
        rel_true_list.append(rel_true.reshape(-1))
        util_true_list.append(util_true.reshape(-1))
        adh_true_list.append(adh_true.reshape(-1))

    rel_pred  = torch.cat(rel_pred_list)
    util_pred = torch.cat(util_pred_list)
    adh_pred  = torch.cat(adh_pred_list)
    rel_true  = torch.cat(rel_true_list)
    util_true = torch.cat(util_true_list)
    adh_true  = torch.cat(adh_true_list)

    def prf(pred, true):
        tp = (pred & true).sum().float()
        fp = (pred & ~true).sum().float()
        fn = (~pred & true).sum().float()
        prec = tp.div((tp + fp).clamp(min=1))
        rec  = tp.div((tp + fn).clamp(min=1))
        f1   = 2 * prec * rec / (prec + rec).clamp(min=1)
        return float(prec), float(rec), float(f1)

    rel_p,  rel_r,  rel_f1  = prf(rel_pred,  rel_true)
    util_p, util_r, util_f1 = prf(util_pred, util_true)
    adh_p,  adh_r,  adh_f1  = prf(adh_pred,  adh_true)

    return {
        'relevance_precision':   rel_p,
        'relevance_recall':      rel_r,
        'relevance_f1':          rel_f1,
        'utilization_precision': util_p,
        'utilization_recall':    util_r,
        'utilization_f1':        util_f1,
        'adherence_precision':   adh_p,
        'adherence_recall':      adh_r,
        'adherence_f1':          adh_f1,
    }


def tune_thresholds_on_val(
    all_logits,
    all_labels,
    all_masks,
    candidates=None,
):
    """Find per-target thresholds that maximise F1 on the provided data.

    Sweeps candidates thresholds independently for relevance, utilization,
    and adherence using the already-collected logits (no extra forward passes).

    Returns:
        {"relevance": t, "utilization": t, "adherence": t}
    """
    if candidates is None:
        candidates = THRESHOLD_CANDIDATES

    def _flat_probs_and_true(logit_key, label_key, mask_key):
        probs_list, true_list = [], []
        for logits, labels, masks in zip(all_logits, all_labels, all_masks):
            msk = masks[mask_key].bool()
            if msk.any():
                probs_list.append(torch.sigmoid(logits[logit_key])[msk].reshape(-1))
                true_list.append((labels[label_key][msk] == 1).reshape(-1))
        if not probs_list:
            return None, None
        return torch.cat(probs_list), torch.cat(true_list)

    targets = {
        "relevance":   ("logits_relevance",   "labels_relevance",   "context_mask"),
        "utilization": ("logits_utilization", "labels_utilization", "context_mask"),
        "adherence":   ("logits_adherence",   "labels_adherence",   "response_mask"),
    }

    best_thresholds = {}
    for tgt_name, (lgt_key, lbl_key, msk_key) in targets.items():
        probs, true = _flat_probs_and_true(lgt_key, lbl_key, msk_key)
        if probs is None:
            best_thresholds[tgt_name] = 0.5
            continue

        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            pred = probs > t
            tp = (pred & true).sum().float()
            fp = (pred & ~true).sum().float()
            fn = (~pred & true).sum().float()
            prec = tp / (tp + fp).clamp(min=1)
            rec  = tp / (tp + fn).clamp(min=1)
            f1   = float(2 * prec * rec / (prec + rec).clamp(min=1))
            if f1 > best_f1:
                best_f1, best_t = f1, t

        print(f"[metrics] Tuned threshold for {tgt_name}: {best_t} (val_f1={best_f1:.4f})")
        best_thresholds[tgt_name] = best_t

    return best_thresholds
