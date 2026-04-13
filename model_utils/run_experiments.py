"""
run_experiments.py — launch file for training experiments.

Run from the model_utils directory:
    python run_experiments.py

Or import individual functions from the notebook:
    from run_experiments import SWEEP, BASE_CFG, run_single
"""

import sys
from pathlib import Path

# make sure model_utils is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent))
# parent of model_utils (project root) — needed for `from prompt_test.data import ...`
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ProjectConfig, TrainingConfig, ModelConfig, DataConfig
from pipeline import train
from runner import SweepConfig, run_sweep, load_sweep_results

# ──────────────────────────────────────────────────────────────────────────────
# 1. PER-MODEL CONTEXT LIMITS
#    Different architectures support different maximum sequence lengths.
#    List only the lengths that make sense for each model.
# ──────────────────────────────────────────────────────────────────────────────

MODEL_LENGTHS = {
    # DeBERTa — no rotary positional embeddings, tested best at short contexts
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli": [128, 256, 512],

    # BGE-M3 — supports up to 8192
    "BAAI/bge-m3": [512, 1024, 2048, 4096, 8192],

    # RoPE-BERT — limited to 2048 by architecture
    "Tochka-AI/ruRoPEBert-e5-base-2k": [256, 512, 1024, 2048],

    # RuModernBERT — supports up to 8192
    "deepvk/RuModernBERT-base": [512, 1024, 2048, 4096, 8192],
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. SHARED TRAINING HYPERPARAMETERS
#    Edit these to change LR, batch size, epochs, etc. for ALL runs.
# ──────────────────────────────────────────────────────────────────────────────

BASE_CFG = ProjectConfig(
    data=DataConfig(
        max_length=512,         # overridden per run by the sweep
    ),
    model=ModelConfig(
        use_fp16=True,
    ),
    train=TrainingConfig(
        num_epochs=3,
        train_batch_size=16,    # fallback when max_length is not in the map below
        eval_batch_size=16,
        learning_rate=3e-4,
        weight_decay=0.001,
        grad_clip=1.0,
        save_datasets=True,

        # Batch size automatically decreases for longer contexts to fit in GPU memory.
        # If max_length is not in this map, train_batch_size above is used as fallback.
        # Set to None to always use train_batch_size.
        length_batch_size_map={
            128:  64,
            256:  32,
            512:  8,
            1024:  4,
            2048:  2,
            4096:  2,
            8192:  1,
        },

        # Per-target loss weights (normalized sum).
        # Default 1/1/1 is equivalent to (l_rel + l_util + l_adh) / 3.
        # Increase a weight to make the model focus more on that target.
        loss_weight_relevance=1.0,
        loss_weight_utilization=1.0,
        loss_weight_adherence=1.0,
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
# 3. SWEEP DEFINITION
# ──────────────────────────────────────────────────────────────────────────────

SWEEP = SweepConfig(
    models=list(MODEL_LENGTHS.keys()),
    model_lengths=MODEL_LENGTHS,    # per-model length lists
    base_output_dir="sweeps",
    num_epochs=BASE_CFG.train.num_epochs,
    rewrite_dataset=False,
    extra_overrides={
        # dot-path keys map to any field in ProjectConfig
        "train.learning_rate":           BASE_CFG.train.learning_rate,
        "train.train_batch_size":        BASE_CFG.train.train_batch_size,
        "train.eval_batch_size":         BASE_CFG.train.eval_batch_size,
        "train.weight_decay":            BASE_CFG.train.weight_decay,
        "train.grad_clip":               BASE_CFG.train.grad_clip,
        "train.loss_weight_relevance":   BASE_CFG.train.loss_weight_relevance,
        "train.loss_weight_utilization": BASE_CFG.train.loss_weight_utilization,
        "train.loss_weight_adherence":   BASE_CFG.train.loss_weight_adherence,
        "train.length_batch_size_map":   BASE_CFG.train.length_batch_size_map,
    },
)

# ──────────────────────────────────────────────────────────────────────────────
# 4. SINGLE-EXPERIMENT HELPER
#    Use this to quickly run one model + length without touching the sweep.
# ──────────────────────────────────────────────────────────────────────────────

def run_single(
    model_name: str,
    max_length: int,
    *,
    num_epochs: int = BASE_CFG.train.num_epochs,
    learning_rate: float = BASE_CFG.train.learning_rate,
    train_batch_size: int = BASE_CFG.train.train_batch_size,
    eval_batch_size: int = BASE_CFG.train.eval_batch_size,
    weight_decay: float = BASE_CFG.train.weight_decay,
    loss_weight_relevance: float = BASE_CFG.train.loss_weight_relevance,
    loss_weight_utilization: float = BASE_CFG.train.loss_weight_utilization,
    loss_weight_adherence: float = BASE_CFG.train.loss_weight_adherence,
    output_dir: str = None,
    rewrite_dataset: bool = False,
):
    """Run one experiment with explicit hyperparameters.

    Example::

        from run_experiments import run_single
        result = run_single("BAAI/bge-m3", max_length=1024, num_epochs=5, learning_rate=1e-4)
        print(result.best_val_f1)
        print(result.test_metrics)   # test metrics from the best checkpoint

        # Focus more on relevance:
        result = run_single("BAAI/bge-m3", 512, loss_weight_relevance=2.0)
    """
    from data import safe_model_tag

    if output_dir is None:
        tag = safe_model_tag(model_name)
        output_dir = f"sweeps/{tag}/len_{max_length}"

    run_cfg = BASE_CFG.with_overrides(**{
        "model.pretrained_name":         model_name,
        "data.max_length":               max_length,
        "train.output_dir":              output_dir,
        "train.num_epochs":              num_epochs,
        "train.learning_rate":           learning_rate,
        "train.train_batch_size":        train_batch_size,
        "train.eval_batch_size":         eval_batch_size,
        "train.weight_decay":            weight_decay,
        "train.loss_weight_relevance":   loss_weight_relevance,
        "train.loss_weight_utilization": loss_weight_utilization,
        "train.loss_weight_adherence":   loss_weight_adherence,
    })
    return train(run_cfg, rewrite_dataset=rewrite_dataset)


# ──────────────────────────────────────────────────────────────────────────────
# 5. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run training experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # experiment selection
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to run. Defaults to all models in MODEL_LENGTHS.",
    )
    parser.add_argument(
        "--lengths", nargs="+", type=int, default=None,
        help="Override max_lengths for all models (ignores per-model limits).",
    )
    # training hyperparameters
    parser.add_argument("--epochs",          type=int,   default=BASE_CFG.train.num_epochs)
    parser.add_argument("--lr",              type=float, default=BASE_CFG.train.learning_rate)
    parser.add_argument("--batch-size",      type=int,   default=BASE_CFG.train.train_batch_size)
    parser.add_argument("--eval-batch-size", type=int,   default=BASE_CFG.train.eval_batch_size)
    parser.add_argument("--weight-decay",    type=float, default=BASE_CFG.train.weight_decay)
    parser.add_argument("--grad-clip",       type=float, default=BASE_CFG.train.grad_clip)
    # loss weights: --loss-weights rel util adh  (e.g. --loss-weights 2.0 1.0 1.0)
    parser.add_argument(
        "--loss-weights", nargs=3, type=float, metavar=("REL", "UTIL", "ADH"),
        default=None,
        help="Per-target loss weights for relevance / utilization / adherence. "
             "Weights are normalized (sum), so 2 1 1 upweights relevance by 2×.",
    )
    # misc
    parser.add_argument("--output-dir",      default="sweeps")
    parser.add_argument("--rewrite-dataset", action="store_true")
    parser.add_argument("--single", action="store_true",
                        help="Run only the first model × first length (quick smoke test).")
    args = parser.parse_args()

    models = args.models or list(MODEL_LENGTHS.keys())

    extra = {
        "train.learning_rate":           args.lr,
        "train.train_batch_size":        args.batch_size,
        "train.eval_batch_size":         args.eval_batch_size,
        "train.weight_decay":            args.weight_decay,
        "train.grad_clip":               args.grad_clip,
        "train.length_batch_size_map":   BASE_CFG.train.length_batch_size_map,
    }
    if args.loss_weights:
        extra["train.loss_weight_relevance"]   = args.loss_weights[0]
        extra["train.loss_weight_utilization"] = args.loss_weights[1]
        extra["train.loss_weight_adherence"]   = args.loss_weights[2]

    if args.single:
        # quick single-run smoke test
        m = models[0]
        l = (args.lengths or MODEL_LENGTHS.get(m, [512]))[0]
        print(f"[run_experiments] Smoke test: model={m}, max_length={l}")
        result = run_single(
            m, l,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            train_batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            weight_decay=args.weight_decay,
            loss_weight_relevance=extra.get("train.loss_weight_relevance",
                                            BASE_CFG.train.loss_weight_relevance),
            loss_weight_utilization=extra.get("train.loss_weight_utilization",
                                              BASE_CFG.train.loss_weight_utilization),
            loss_weight_adherence=extra.get("train.loss_weight_adherence",
                                            BASE_CFG.train.loss_weight_adherence),
            output_dir=f"{args.output_dir}/{m}/len_{l}",
            rewrite_dataset=args.rewrite_dataset,
        )
        print(f"Done. best_val_f1={result.best_val_f1:.4f}")
        if result.test_metrics:
            print("Test metrics:")
            for k, v in result.test_metrics.items():
                print(f"  {k}: {v:.4f}")
    else:
        if args.lengths:
            # uniform length list — ignore per-model limits
            model_lengths = {m: args.lengths for m in models}
        else:
            model_lengths = {m: MODEL_LENGTHS.get(m, [512]) for m in models}

        sweep = SweepConfig(
            models=models,
            model_lengths=model_lengths,
            base_output_dir=args.output_dir,
            num_epochs=args.epochs,
            rewrite_dataset=args.rewrite_dataset,
            extra_overrides=extra,
        )
        results = run_sweep(sweep, base_cfg=BASE_CFG)

        # summary
        df = load_sweep_results(args.output_dir)
        if not df.empty:
            cols = ["model_name", "max_length", "best_val_f1",
                    "relevance_f1", "utilization_f1", "adherence_f1"]
            # add test columns if present
            test_cols = [c for c in df.columns if c.startswith("test_")]
            print("\n" + df[cols + test_cols]
                      .sort_values("best_val_f1", ascending=False)
                      .to_string(index=False))
