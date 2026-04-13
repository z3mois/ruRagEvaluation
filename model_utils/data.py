from typing import Dict, Any, List, Tuple
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoConfig

from config import DataConfig
from prompt_test.data import load_full_ragbench

from pathlib import Path
from datasets import load_from_disk
import re

def safe_model_tag(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "__", model_name)

def save_dataset(ds: DatasetDict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(path)
    print(f"[data] Saved dataset to: {path}")

def load_dataset(path: str) -> DatasetDict:
    ds = load_from_disk(path)
    print(f"[data] Loaded dataset from: {path}")
    return ds


def raw_cache_path(output_dir: str) -> Path:
    """Canonical location for the raw (pre-tokenization) combined dataset."""
    return Path(output_dir) / "raw" / "combined"


def tokenized_cache_path(output_dir: str, model_name: str, max_length: int = None) -> Path:
    """Canonical location for the tokenized dataset for a given model.

    Includes max_length in the path to prevent stale-cache collisions when
    the same output_dir is reused with different max_length values.
    """
    tag = safe_model_tag(model_name)
    if max_length is not None:
        tag = f"{tag}_len{max_length}"
    return Path(output_dir) / "tokenized" / tag

def has_sentences(example: Dict[str, Any], data_cfg: DataConfig) -> bool:
    docs = example.get(data_cfg.documents_field, [])
    return len(docs) >= data_cfg.min_sentences


def count_tokens(example: Dict[str, Any], tokenizer, data_cfg: DataConfig) -> int:
    q_ids = tokenizer.encode(example[data_cfg.question_field], add_special_tokens=False)

    doc_ids = []
    for doc in example[data_cfg.documents_field]:
        for key, sent in doc:
            doc_ids.extend(tokenizer.encode(sent, add_special_tokens=False))

    r_ids = tokenizer.encode(example[data_cfg.response_field], add_special_tokens=False)
    # + два [SEP]
    return len(q_ids) + 1 + len(doc_ids) + 1 + len(r_ids)


_BUILD_EXAMPLE_DEBUG_COUNT = 0
_BUILD_EXAMPLE_DEBUG_FILE = None


def build_example(
    example: Dict[str, Any],
    tokenizer,
    data_cfg: DataConfig,
    max_length: int,
    debug_dir: str = None,
) -> Dict[str, Any]:
    """
    Собираем:
      - input_ids, attention_mask
      - context_mask, response_mask
      - labels_relevance, labels_utilization, labels_adherence
    Под твою схему: [question] [SEP] [docs] [SEP] [response]
    """
    global _BUILD_EXAMPLE_DEBUG_COUNT, _BUILD_EXAMPLE_DEBUG_FILE

    question_text = example[data_cfg.question_field]
    question_ids = tokenizer.encode(question_text, add_special_tokens=False)

    rel_keys = set(example[data_cfg.relevant_keys_field])
    util_keys = set(example[data_cfg.utilized_keys_field])

    doc_ids: List[int] = []
    rel_labels: List[float] = []
    util_labels: List[float] = []

    doc_keys_found: List[str] = []
    for doc in example[data_cfg.documents_field]:
        for key, sent in doc:
            doc_keys_found.append(key)
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            doc_ids += tokens
            # Prefix match: relevant_keys like '0ac' should match document_key '0a'.
            # A document sentence is relevant if ANY relevant_key starts with its key.
            is_rel = key in rel_keys or any(rk.startswith(key) for rk in rel_keys)
            is_util = key in util_keys or any(uk.startswith(key) for uk in util_keys)
            rel_labels += [float(is_rel)] * len(tokens)
            util_labels += [float(is_util)] * len(tokens)
    if _BUILD_EXAMPLE_DEBUG_COUNT < 5:
        _BUILD_EXAMPLE_DEBUG_COUNT += 1
        if debug_dir and _BUILD_EXAMPLE_DEBUG_FILE is None:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
            _BUILD_EXAMPLE_DEBUG_FILE = str(Path(debug_dir) / "build_example_debug.txt")
        if _BUILD_EXAMPLE_DEBUG_FILE:
            with open(_BUILD_EXAMPLE_DEBUG_FILE, "a", encoding="utf-8") as f:
                def _match(k, keys):
                    return k in keys or any(x.startswith(k) for x in keys)
                n_rel_match = sum(1 for k in doc_keys_found if _match(k, rel_keys))
                n_util_match = sum(1 for k in doc_keys_found if _match(k, util_keys))
                f.write(f"\n=== build_example #{_BUILD_EXAMPLE_DEBUG_COUNT} ===\n")
                f.write(f"  rel_keys ({len(rel_keys)}): {sorted(rel_keys)[:10]}\n")
                f.write(f"  util_keys ({len(util_keys)}): {sorted(util_keys)[:10]}\n")
                f.write(f"  doc_keys ({len(doc_keys_found)}): {doc_keys_found[:10]}\n")
                f.write(f"  rel  matched: {n_rel_match}/{len(doc_keys_found)}\n")
                f.write(f"  util matched: {n_util_match}/{len(doc_keys_found)}\n")
                f.write(f"  pos in rel_labels: {sum(1 for x in rel_labels if x == 1.0)}/{len(rel_labels)}\n")
                f.write(f"  pos in util_labels: {sum(1 for x in util_labels if x == 1.0)}/{len(util_labels)}\n")

    response_text = example[data_cfg.response_field]
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    adh_score = float(example[data_cfg.adherence_score_field])
    adh_labels = [adh_score] * len(response_ids)

    sep_id = tokenizer.sep_token_id

    input_ids = question_ids + [sep_id] + doc_ids + [sep_id] + response_ids

    q_len = len(question_ids)
    d_len = len(doc_ids)
    r_len = len(response_ids)

    # маски:
    context_mask = (
        [0] * (q_len + 1) +
        [1] * d_len +
        [0] +
        [0] * r_len
    )
    response_mask = (
        [0] * (q_len + 1 + d_len + 1) +
        [1] * r_len
    )

    input_ids = input_ids[:max_length]
    context_mask = context_mask[:max_length]
    response_mask = response_mask[:max_length]

    full_rel_labels = [0.0] * len(input_ids)
    full_util_labels = [0.0] * len(input_ids)
    full_adh_labels = [0.0] * len(input_ids)

    ctx_start = q_len + 1
    ctx_end = min(ctx_start + d_len, max_length)
    rl_slice_len = ctx_end - ctx_start
    full_rel_labels[ctx_start:ctx_end] = rel_labels[:rl_slice_len]
    full_util_labels[ctx_start:ctx_end] = util_labels[:rl_slice_len]

    # question + sep + docs + sep
    resp_start = q_len + 1 + d_len + 1
    if resp_start < max_length:
        resp_end = min(resp_start + r_len, max_length)
        adh_slice_len = resp_end - resp_start
        full_adh_labels[resp_start:resp_end] = adh_labels[:adh_slice_len]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
        "context_mask": torch.tensor(context_mask, dtype=torch.bool),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
        "labels_relevance": torch.tensor(full_rel_labels, dtype=torch.float),
        "labels_utilization": torch.tensor(full_util_labels, dtype=torch.float),
        "labels_adherence": torch.tensor(full_adh_labels, dtype=torch.float),
    }


def make_raw_combined(data_cfg: DataConfig) -> DatasetDict:
    raw = load_full_ragbench()

    raw = {
        name: ds.filter(lambda ex, cfg_local=data_cfg: has_sentences(ex, cfg_local))
        for name, ds in raw.items()
    }

    train_splits = [ds["train"] for ds in raw.values() if "train" in ds]
    val_splits   = [ds["validation"] for ds in raw.values() if "validation" in ds]
    test_splits  = [ds["test"] for ds in raw.values() if "test" in ds]

    combined = DatasetDict({
        "train":      concatenate_datasets(train_splits),
        "validation": concatenate_datasets(val_splits),
        "test":       concatenate_datasets(test_splits),
    })

    return combined
def tokenize_from_raw(
    combined_raw: DatasetDict,
    data_cfg: DataConfig,
    model_name: str,
    debug_dir: str = None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    if data_cfg.max_length is not None:
        effective_max_length = data_cfg.max_length
    else:
        model_max = getattr(model_cfg, "max_position_embeddings", None)
        tok_max = tokenizer.model_max_length
        candidates = [x for x in (model_max, tok_max) if x is not None and x > 0]
        effective_max_length = min(candidates) if candidates else 512
        
    print(f"[data] Tokenize for {model_name}, max_length={effective_max_length}")

    filtered = DatasetDict({
        split: ds.filter(
            lambda ex, tok=tokenizer, cfg_local=data_cfg, ml=effective_max_length:
            count_tokens(ex, tok, cfg_local) <= ml
        )
        for split, ds in combined_raw.items()
    })

    global _BUILD_EXAMPLE_DEBUG_COUNT, _BUILD_EXAMPLE_DEBUG_FILE
    _BUILD_EXAMPLE_DEBUG_COUNT = 0
    _BUILD_EXAMPLE_DEBUG_FILE = None

    def _map_fn(example, tok=tokenizer, cfg_local=data_cfg, ml=effective_max_length, dd=debug_dir):
        return build_example(example, tok, cfg_local, max_length=ml, debug_dir=dd)

    tokenized = filtered.map(
        _map_fn,
        batched=False,
        remove_columns=filtered["train"].column_names,
    )
    print(f"[data] Tokenize for {model_name}, max_length={effective_max_length}")
    return tokenized, tokenizer, effective_max_length



def collate_fn(batch, pad_token_id: int):
    """
    Преобразуем все элементы батча в torch.Tensor и паддим
    по максимальной длине в батче.
    """
    keys = batch[0].keys()
    out: Dict[str, torch.Tensor] = {}

    for k in keys:

        vals = [torch.as_tensor(sample[k]) for sample in batch]

        if k in (
            "input_ids", "attention_mask",
            "context_mask", "response_mask",
            "labels_relevance", "labels_utilization", "labels_adherence",
        ):
            if k == "input_ids":
                padding_value = pad_token_id
            else:
                padding_value = 0

            out[k] = pad_sequence(
                vals,
                batch_first=True,
                padding_value=padding_value,
            )
        else:
            out[k] = torch.stack(vals)

    return out


def make_dataloaders(tokenized, tokenizer, train_cfg):
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=train_cfg.train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    test_loader = DataLoader(
        tokenized["test"],
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    return train_loader, val_loader, test_loader
def log_one_processed_example(tokenized, tokenizer):
    """Печать одного примера из tokenized['train'] в удобном виде."""
    if "train" not in tokenized or len(tokenized["train"]) == 0:
        print("[data] Нет примеров в split='train', пример не показать.")
        return

    processed = tokenized["train"][0]  

    input_ids = processed["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    labels_rel = processed["labels_relevance"]
    labels_util = processed["labels_utilization"]
    labels_adh = processed["labels_adherence"]

    if isinstance(labels_rel, torch.Tensor):
        labels_rel = labels_rel.tolist()
    if isinstance(labels_util, torch.Tensor):
        labels_util = labels_util.tolist()
    if isinstance(labels_adh, torch.Tensor):
        labels_adh = labels_adh.tolist()

    print("\n[data] Пример токенизации и разметки из tokenized['train'][0]:")
    print(f"{'Token':20} | {'Relevance':9} | {'Utilization':11} | {'Adherence':9}")
    print("-" * 60)

    for i, (token, rel, util, adh) in enumerate(
        zip(tokens, labels_rel, labels_util, labels_adh), 1
    ):

        if token == tokenizer.pad_token or token == "[PAD]":
            break
        print(f"{token:20} | {rel:9.1f} | {util:11.1f} | {adh:9.1f}")

    print("-" * 60)
    print("[data] Конец примера\n")
