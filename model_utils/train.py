
from config import cfg as _default_cfg, ProjectConfig
from pipeline import train as _pipeline_train


def train(project_cfg: ProjectConfig = None, rewrite_dataset: bool = False, **_ignored):
    """Train a model.

    Args:
        project_cfg: Config for this run.  Defaults to the global ``cfg`` singleton
                     so that existing notebook cells work without modification.
        rewrite_dataset: Force re-tokenization even if a cached version exists.

    Returns:
        dict with keys ``max_length``, ``output_dir``, ``class_metrics`` (backward compat)
        and a full :class:`~results.ExperimentResult` under key ``result``.
    """
    if project_cfg is None:
        project_cfg = _default_cfg

    result = _pipeline_train(project_cfg, rewrite_dataset=rewrite_dataset)

    return {
        "max_length": result.max_length,
        "output_dir": result.output_dir,
        "class_metrics": result.val_metrics,
        "result": result,
    }


if __name__ == "__main__":
    train()
