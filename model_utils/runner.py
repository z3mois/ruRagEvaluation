
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import ProjectConfig, cfg as _default_cfg
from data import safe_model_tag
from pipeline import train
from results import ExperimentResult, load_results_from_dirs


@dataclass
class SweepConfig:
    """Defines a grid sweep over models × max_lengths.

    Per-model lengths
    -----------------
    Use ``model_lengths`` to set different length lists per model::

        SweepConfig(
            models=["BAAI/bge-m3", "Tochka-AI/ruRoPEBert-e5-base-2k"],
            max_lengths=[512, 1024],          # fallback for models not in model_lengths
            model_lengths={
                "BAAI/bge-m3":                     [512, 1024, 2048],
                "Tochka-AI/ruRoPEBert-e5-base-2k": [256, 512],        # smaller context
            },
        )

    If a model is not listed in ``model_lengths``, the shared ``max_lengths`` list is used.
    """
    models: List[str]
    max_lengths: List[int] = field(default_factory=lambda: [512])
    base_output_dir: str = "sweeps"
    num_epochs: int = 2
    rewrite_dataset: bool = False
    model_lengths: Dict[str, List[int]] = field(default_factory=dict)
    extra_overrides: Dict = field(default_factory=dict)


def run_sweep(
    sweep: SweepConfig,
    base_cfg: Optional[ProjectConfig] = None,
) -> List[ExperimentResult]:
    """Run a grid of experiments without mutating any shared config.

    For each (model, max_length) combination, builds a fresh ProjectConfig via
    ``with_overrides()``, calls ``pipeline.train()``, and collects results.

    Args:
        sweep: Sweep specification.
        base_cfg: Base config to derive per-run configs from.
                  Defaults to the global ``cfg`` singleton.

    Returns:
        List of ExperimentResult objects, one per completed run.
    """
    if base_cfg is None:
        base_cfg = _default_cfg

    run_plan = []
    for model_name in sweep.models:
        lengths = sweep.model_lengths.get(model_name, sweep.max_lengths)
        for max_length in lengths:
            run_plan.append((model_name, max_length))

    results: List[ExperimentResult] = []
    total = len(run_plan)

    for run_idx, (model_name, max_length) in enumerate(run_plan, 1):
        model_tag = safe_model_tag(model_name)
        output_dir = str(Path(sweep.base_output_dir) / model_tag / f"len_{max_length}")

        print(
            f"\n{'='*70}\n"
            f"[runner] Run {run_idx}/{total}: "
            f"model={model_name}, max_length={max_length}\n"
            f"[runner] output_dir={output_dir}\n"
            f"{'='*70}"
        )

        overrides = {
            "model.pretrained_name": model_name,
            "data.max_length": max_length,
            "train.output_dir": output_dir,
            "train.num_epochs": sweep.num_epochs,
            **sweep.extra_overrides,
        }
        run_cfg = base_cfg.with_overrides(**overrides)

        try:
            result = train(run_cfg, rewrite_dataset=sweep.rewrite_dataset)
            results.append(result)
            print(f"[runner] Run {run_idx}/{total} done — best_val_f1={result.best_val_f1:.4f}")
        except Exception as exc:
            print(f"[runner] Run {run_idx}/{total} FAILED: {exc}")
            traceback.print_exc()

    print(f"\n[runner] Sweep complete. {len(results)}/{total} runs succeeded.")
    return results


def load_sweep_results(sweep_root: str = "sweeps"):
    """Convenience wrapper: load all result.json files under sweep_root into a DataFrame."""
    return load_results_from_dirs(sweep_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a grid sweep of training experiments.")
    parser.add_argument("--models", nargs="+", required=True, help="HuggingFace model names")
    parser.add_argument("--lengths", nargs="+", type=int, required=True, help="max_length values")
    parser.add_argument("--base-output-dir", default="sweeps", help="Root directory for outputs")
    parser.add_argument("--epochs", type=int, default=2, help="num_epochs per run")
    parser.add_argument("--rewrite-dataset", action="store_true", help="Force re-tokenization")
    args = parser.parse_args()

    sweep_cfg = SweepConfig(
        models=args.models,
        max_lengths=args.lengths,
        base_output_dir=args.base_output_dir,
        num_epochs=args.epochs,
        rewrite_dataset=args.rewrite_dataset,
    )
    all_results = run_sweep(sweep_cfg)

    print("\n[runner] Results summary:")
    print(f"{'Model':<45} {'max_len':>8} {'best_val_f1':>12}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: -x.best_val_f1):
        print(f"{r.model_name:<45} {str(r.max_length):>8} {r.best_val_f1:>12.4f}")
