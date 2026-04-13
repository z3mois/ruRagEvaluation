
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import TrainingConfig
from metrics import compute_classification_metrics_stream, compute_trace_metrics_inference


def run_training_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainingConfig,
    device: torch.device,
    use_amp: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Tuple[Optional[Dict], dict, float]:
    """Run the full training + validation loop.

    Returns:
        best_state_dict: state dict of the best model by relevance_f1 (or None if no validation ran)
        final_val_metrics: classification metrics dict from the last validation epoch
        best_val_f1: best relevance_f1 seen on validation
    """
    amp_context = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext

    best_val_f1 = -1.0
    best_state: Optional[Dict] = None
    final_val_metrics: dict = {}

    for epoch in range(train_cfg.num_epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"[train] Epoch {epoch}", total=len(train_loader))
        for batch_idx, batch in enumerate(loop):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            labels = {k: v.to(device) for k, v in batch.items() if k.startswith("labels_")}
            masks  = {k: v.to(device) for k, v in batch.items() if k.endswith("_mask")}
            if batch_idx == 0:
                b_ids = batch["input_ids"]
                ctx_m = batch.get("context_mask")
                lbl_r = batch.get("labels_relevance")
                lbl_u = batch.get("labels_utilization")
                B, T = b_ids.shape
                ctx_tokens = int(ctx_m.sum()) if ctx_m is not None else 0
                pos_rel = int((lbl_r[ctx_m.bool()] == 1).sum()) if (lbl_r is not None and ctx_m is not None) else 0
                pos_util = int((lbl_u[ctx_m.bool()] == 1).sum()) if (lbl_u is not None and ctx_m is not None) else 0
                pos_rate_rel  = pos_rel  / max(ctx_tokens, 1)
                pos_rate_util = pos_util / max(ctx_tokens, 1)
                print(
                    f"[train] Epoch {epoch} | batch 0: shape=({B},{T}) "
                    f"ctx_tokens={ctx_tokens} "
                    f"pos_rel={pos_rel} ({pos_rate_rel:.1%}) "
                    f"pos_util={pos_util} ({pos_rate_util:.1%})"
                )

            optimizer.zero_grad(set_to_none=True)

            with amp_context():
                out = model(**inputs)

                l_rel = F.binary_cross_entropy_with_logits(
                    out["logits_relevance"][masks["context_mask"]],
                    labels["labels_relevance"][masks["context_mask"]],
                )
                l_util = F.binary_cross_entropy_with_logits(
                    out["logits_utilization"][masks["context_mask"]],
                    labels["labels_utilization"][masks["context_mask"]],
                )
                l_adh = F.binary_cross_entropy_with_logits(
                    out["logits_adherence"][masks["response_mask"]],
                    labels["labels_adherence"][masks["response_mask"]],
                )

                w_rel  = train_cfg.loss_weight_relevance
                w_util = train_cfg.loss_weight_utilization
                w_adh  = train_cfg.loss_weight_adherence
                loss = (w_rel * l_rel + w_util * l_util + w_adh * l_adh) / (w_rel + w_util + w_adh)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]["lr"]
            loop.set_postfix(loss=loss.item(), lr=f"{current_lr:.2e}")

        print(f"[train] Epoch {epoch} train loss sum: {total_loss:.4f}")

        model.eval()

        trace_lists = {
            "relevance_rate":   [],
            "utilization_rate": [],
            "adherence_rate":   [],
            "completeness":     [],
        }
        all_logits_batches = []
        all_labels_batches = []
        all_masks_batches  = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[val]"):
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                labels = {k: v for k, v in batch.items() if k.startswith("labels_")}
                masks  = {k: v for k, v in batch.items() if k.endswith("_mask")}

                out = model(**inputs)

                batch_trace = compute_trace_metrics_inference(out, masks)
                for name, values in batch_trace.items():
                    trace_lists[name].extend(values.cpu().tolist())

                all_logits_batches.append({
                    "logits_relevance":   out["logits_relevance"].cpu(),
                    "logits_utilization": out["logits_utilization"].cpu(),
                    "logits_adherence":   out["logits_adherence"].cpu(),
                })
                all_labels_batches.append({
                    "labels_relevance":   labels["labels_relevance"].cpu(),
                    "labels_utilization": labels["labels_utilization"].cpu(),
                    "labels_adherence":   labels["labels_adherence"].cpu(),
                })
                all_masks_batches.append({
                    "context_mask":  masks["context_mask"].cpu(),
                    "response_mask": masks["response_mask"].cpu(),
                })

        all_rel_pos, all_rel_neg = [], []
        all_util_pos, all_util_neg = [], []
        pred_pos_rel, pred_pos_util, pred_pos_adh = [], [], []
        for lgt, lbl, msk in zip(all_logits_batches, all_labels_batches, all_masks_batches):
            ctx  = msk["context_mask"].bool()
            resp = msk["response_mask"].bool()
            p_rel  = torch.sigmoid(lgt["logits_relevance"])
            p_util = torch.sigmoid(lgt["logits_utilization"])
            p_adh  = torch.sigmoid(lgt["logits_adherence"])
            if ctx.any():
                is_pos = (lbl["labels_relevance"][ctx] == 1)
                all_rel_pos.append(p_rel[ctx][is_pos])
                all_rel_neg.append(p_rel[ctx][~is_pos])
                is_pos_u = (lbl["labels_utilization"][ctx] == 1)
                all_util_pos.append(p_util[ctx][is_pos_u])
                all_util_neg.append(p_util[ctx][~is_pos_u])
                pred_pos_rel.append(  (p_rel[ctx]  > 0.5).float().mean())
                pred_pos_util.append( (p_util[ctx] > 0.5).float().mean())
            if resp.any():
                pred_pos_adh.append(  (p_adh[resp] > 0.5).float().mean())

        def _stats(lst):
            if not lst:
                return "n/a"
            v = torch.cat(lst)
            if v.numel() == 0:
                return "empty"
            qs = torch.quantile(v.float(), torch.tensor([0.25, 0.5, 0.75]))
            return f"mean={v.mean():.3f} q25={qs[0]:.3f} median={qs[1]:.3f} q75={qs[2]:.3f}"

        print("\n[val] Sigmoid distribution (relevance):")
        print(f"  positive tokens : {_stats(all_rel_pos)}")
        print(f"  negative tokens : {_stats(all_rel_neg)}")
        print("\n[val] Sigmoid distribution (utilization):")
        print(f"  positive tokens : {_stats(all_util_pos)}")
        print(f"  negative tokens : {_stats(all_util_neg)}")

        def _rate(lst):
            return f"{torch.stack(lst).mean():.3f}" if lst else "n/a"
        print(f"[val] Predicted positive rate @0.5:  rel={_rate(pred_pos_rel)}  util={_rate(pred_pos_util)}  adh={_rate(pred_pos_adh)}")

        final_val_metrics = compute_classification_metrics_stream(
            all_logits_batches, all_labels_batches, all_masks_batches
        )

        print("\n[val] Classification metrics @0.5 (full validation set):")
        for name, val in final_val_metrics.items():
            print(f"  {name}: {val:.4f}")

        print(
            "[val] Trace metrics avg -> "
            f"relevance={np.mean(trace_lists['relevance_rate']):.4f}, "
            f"utilization={np.mean(trace_lists['utilization_rate']):.4f}, "
            f"adherence={np.mean(trace_lists['adherence_rate']):.4f}, "
            f"completeness={np.mean(trace_lists['completeness']):.4f}"
        )

        if final_val_metrics.get("relevance_f1", -1.0) > best_val_f1:
            best_val_f1 = final_val_metrics["relevance_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"[train] New best relevance_f1={best_val_f1:.4f} (saved in RAM)")

    print("[train] Best val relevance_f1:", best_val_f1)
    return best_state, final_val_metrics, best_val_f1
