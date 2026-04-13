
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


@dataclass
class DataConfig:
    question_field: str = "question_ru"
    documents_field: str = "documents_sentences_ru"
    response_field: str = "response_ru"

    relevant_keys_field: str = "all_relevant_sentence_keys"
    utilized_keys_field: str = "all_utilized_sentence_keys"
    adherence_score_field: str = "adherence_score"

    max_length: Optional[int] = 1024

    min_sentences: int = 1


@dataclass
class ModelConfig:
    pretrained_name: str = 'BAAI/bge-m3'
    use_fp16: bool = True  

    # Architecture: False = simple (paper-style: backbone + linear heads),
    # True = complex (shared transform + inner MLP heads).
    use_complex_model: bool = False


@dataclass
class TrainingConfig:
    num_epochs: int = 2

    train_batch_size: int = 64
    eval_batch_size: int = 32

    learning_rate: float = 3e-4
    weight_decay: float = 0.001

    # Optional separate LRs for backbone vs heads.
    # If both are None → single `learning_rate` is used for all parameters (unchanged behavior).
    # If set → AdamW gets two param groups: base.* uses backbone_lr, rest uses head_lr.
    backbone_lr: Optional[float] = None
    head_lr: Optional[float] = None
    warmup_ratio: float = 0.0

    grad_clip: float = 1.0
    log_every: int = 100

    output_dir: str = "./BAAI_bge_m3_2k"

    save_datasets: bool = True
    datasets_subdir: str = "datasets"   # output_dir/datasets
    loss_weight_relevance:   float = 1.0
    loss_weight_utilization: float = 1.0
    loss_weight_adherence:   float = 1.0

    length_batch_size_map: Optional[Dict[int, int]] = None

    def resolve_train_batch_size(self, max_length: int) -> int:
        """Return effective train batch size for a given max_length.

        Falls back to self.train_batch_size if max_length is not in the map
        or if no map is configured.
        """
        if self.length_batch_size_map:
            return self.length_batch_size_map.get(max_length, self.train_batch_size)
        return self.train_batch_size


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

    def with_overrides(self, **kwargs) -> "ProjectConfig":
        """Return a deep copy with dot-path overrides applied.

        Example::

            run_cfg = cfg.with_overrides(**{
                "model.pretrained_name": "deepvk/RuModernBERT-base",
                "data.max_length": 512,
                "train.output_dir": "sweeps/modernbert/len_512",
            })
        """
        new = deepcopy(self)
        for path, value in kwargs.items():
            obj = new
            parts = path.split(".")
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], value)
        return new

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectConfig":
        train_d = dict(d["train"])
        # JSON serializes int keys as strings; restore them.
        if train_d.get("length_batch_size_map"):
            train_d["length_batch_size_map"] = {
                int(k): v for k, v in train_d["length_batch_size_map"].items()
            }
        return cls(
            data=DataConfig(**d["data"]),
            model=ModelConfig(**d["model"]),
            train=TrainingConfig(**train_d),
        )


cfg = ProjectConfig()
