import os
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModel


DATASETS = ['cuad', 'delucionqa'] 
def load_full_ragbench() -> Dict[str, DatasetDict]:
    ragbench = {}
    for dataset in DATASETS:
        ragbench[dataset] = load_dataset("CMCenjoyer/ragbench-ru", dataset)
    return ragbench


class DebertaTrace(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        hid = base_model.config.hidden_size
        self.rel_head = nn.Linear(hid, 1)
        self.util_head = nn.Linear(hid, 1)
        self.adh_head = nn.Linear(hid, 1)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        return {
            "logits_relevance": self.rel_head(hs).squeeze(-1),
            "logits_utilization": self.util_head(hs).squeeze(-1),
            "logits_adherence": self.adh_head(hs).squeeze(-1),
        }

def compute_trace_metrics_inference(logits: Dict[str, torch.Tensor],
                                    masks: Dict[str, torch.Tensor],
                                    threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    rel_pred  = (torch.sigmoid(logits['logits_relevance'].cpu())  > threshold)
    util_pred = (torch.sigmoid(logits['logits_utilization'].cpu())> threshold)
    adh_pred  = (torch.sigmoid(logits['logits_adherence'].cpu())   > threshold)

    ctx_m  = masks['context_mask'].cpu()
    resp_m = masks['response_mask'].cpu()

    def rate(pred, mask):
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
        "relevance_rate":   relevance_rate,
        "utilization_rate": utilization_rate,
        "adherence_rate":   adherence_rate,
        "completeness":     completeness
    }


def preprocess_one(example: Dict[str, Any], tokenizer, max_length: int = 1024) -> Dict[str, torch.Tensor]:
    question_ids = tokenizer.encode(example["question_ru"], add_special_tokens=False)
    doc_ids, rel_labels, util_labels = [], [], []

    for doc in example["documents_sentences_ru"]:
        for key, sent in doc:
            tokens = tokenizer.encode(sent, add_special_tokens=False)
            doc_ids += tokens
            rel_labels += [float(key in example["all_relevant_sentence_keys"])] * len(tokens)
            util_labels += [float(key in example["all_utilized_sentence_keys"])] * len(tokens)

    response_ids = tokenizer.encode(example["response_ru"], add_special_tokens=False)
    adh_labels = [float(example.get("adherence_score", 0.0))] * len(response_ids)

    sep_id = tokenizer.sep_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    if sep_id is None:
        sep_id = 102

    input_ids = question_ids + [sep_id] + doc_ids + [sep_id] + response_ids

    context_mask  = [0] * (len(question_ids)+1) + [1]*len(doc_ids) + [0] + [0]*len(response_ids)
    response_mask = [0]*(len(question_ids)+len(doc_ids)+2) + [1]*len(response_ids)

    rel_labels  = [0.0]*(len(question_ids)+1) + rel_labels + [0.0]*(len(response_ids)+1)
    util_labels = [0.0]*(len(question_ids)+1) + util_labels + [0.0]*(len(response_ids)+1)
    adh_labels  = [0.0]*(len(question_ids)+len(doc_ids)+2) + adh_labels

    if len(input_ids) > max_length:
        return {} 

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor([1]*len(input_ids), dtype=torch.long),
        "context_mask": torch.tensor(context_mask, dtype=torch.bool),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
        "labels_relevance": torch.tensor(rel_labels, dtype=torch.float),
        "labels_utilization": torch.tensor(util_labels, dtype=torch.float),
        "labels_adherence": torch.tensor(adh_labels, dtype=torch.float),
    }


@st.cache_resource(show_spinner=False)
def load_tok_and_base_from_checkpoint(checkpoint_dir: str = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'):
    tok = AutoTokenizer.from_pretrained(checkpoint_dir)
    base = AutoModel.from_pretrained(checkpoint_dir)
    return tok, base

@st.cache_resource(show_spinner=True)
def build_model_from_state(checkpoint_dir: str, state_dict_path: str, device: str = "cpu"):
    tokenizer, base = load_tok_and_base_from_checkpoint(checkpoint_dir)
    model = DebertaTrace(base)
    sd = torch.load(state_dict_path, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model

@st.cache_resource(show_spinner=True)
def load_all_datasets():
    return load_full_ragbench()


def collect_fit_indices(ds: Dataset, tokenizer, max_length: int) -> List[int]:
    """Вернёт индексы исходного сплита, которые проходят по длине."""
    kept = []
    for i, ex in enumerate(ds):
        processed = preprocess_one(ex, tokenizer, max_length)
        if processed: 
            kept.append(i)
    return kept


st.set_page_config(page_title="Trace Inference (single item)", layout="wide")
st.title("🧪 Trace Inference (1 элемент)")

with st.sidebar:
    st.header("⚙️ Настройки")
    checkpoint_dir = st.text_input("Папка чекпоинта (save_pretrained)", value='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
    state_path     = st.text_input("Файл state_dict голов (*.pt|*.bin)", value='../tmp/model_state_dict_ru_on_ru.pth')
    max_length     = st.number_input("max_length", min_value=128, max_value=8192, value=1024, step=64)
    threshold      = st.slider("Порог sigmoid", 0.05, 0.95, 0.5, 0.05)
    device_choice  = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device         = st.selectbox("Девайс", options=["cpu","cuda","mps"], index=["cpu","cuda","mps"].index(device_choice))

    st.divider()
    st.subheader("📚 Датасет")
    subds = st.selectbox("Поддатасет", options=DATASETS, index=0)
    split = st.selectbox("Сплит", options=["train","validation","test"], index=0)

    prepare_btn = st.button("Загрузить/подготовить")

if prepare_btn:
    st.experimental_rerun()


if not (os.path.exists(state_path)):
    st.info("Укажи корректные `state_dict` (в сайдбаре), затем жми «Загрузить/подготовить».")
    st.stop()


with st.spinner("Загружаю модель из hf чекпоинта и state_dict..."):
    tokenizer, model = build_model_from_state(checkpoint_dir, state_path, device=device)


with st.spinner("Гружу RAGBench-ru..."):
    ragbench = load_all_datasets()

if subds not in ragbench:
    st.error(f"Поддатасет '{subds}' не найден.")
    st.stop()

if split not in ragbench[subds]:
    st.error(f"Сплит '{split}' отсутствует в '{subds}'. Доступные: {list(ragbench[subds].keys())}")
    st.stop()

current_split: Dataset = ragbench[subds][split]

with st.spinner("Фильтрую элементы, которые влазят по длине..."):
    fit_indices = collect_fit_indices(current_split, tokenizer, max_length)

if len(fit_indices) == 0:
    st.error("Ни один элемент этого сплита не прошёл по длине. Увеличь max_length или выбери другой сплит/датасет.")
    st.stop()

left, right = st.columns([1, 3], vertical_alignment="top")

with left:
    st.subheader("Выбор элемента")
    st.caption(f"Поддатасет: `{subds}` | сплит: `{split}` | подходит: {len(fit_indices)}")
    chosen_original_idx = st.selectbox(
        "Индекс (в исходном сплите)",
        options=fit_indices,
        index=0
    )
    src_ex = current_split[int(chosen_original_idx)]
    with st.expander("Сырой пример (усечённо)"):
        st.json({
            "question": src_ex.get("question_ru","")[:500],
            "response": src_ex.get("response_ru","")[:500],
            "adherence_score": src_ex.get("adherence_score", None),
            "relevant_keys_cnt": len(src_ex.get("all_relevant_sentence_keys", [])),
            "utilized_keys_cnt": len(src_ex.get("all_utilized_sentence_keys", [])),
            "documents_sentences_len": sum(len(doc) for doc in src_ex.get("documents_sentences", [])),
        })

run = st.button("🚀 Запустить инференс для выбранного элемента")

with right:
    st.subheader("Результаты")
    if not run:
        st.info("Нажми «Запустить инференс…»")
        st.stop()

    batch = preprocess_one(src_ex, tokenizer, max_length)
    if not batch:
        st.error("Неожиданно не влазит после препроцесса. Попробуй другой индекс.")
        st.stop()


    input_ids = batch["input_ids"].unsqueeze(0).to(device)
    attn      = batch["attention_mask"].unsqueeze(0).to(device)
    ctx_mask  = batch["context_mask"].unsqueeze(0)
    resp_mask = batch["response_mask"].unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn)

    metrics = compute_trace_metrics_inference(
        logits,
        {"context_mask": ctx_mask, "response_mask": resp_mask},
        threshold=threshold
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Relevance (ctx)", f"{metrics['relevance_rate'][0].item():.3f}")
    m2.metric("Utilization (ctx)", f"{metrics['utilization_rate'][0].item():.3f}")
    m3.metric("Adherence (resp)", f"{metrics['adherence_rate'][0].item():.3f}")
    m4.metric("Completeness", f"{metrics['completeness'][0].item():.3f}")

    tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"].tolist())
    rel_p  = torch.sigmoid(logits["logits_relevance"][0].cpu())
    util_p = torch.sigmoid(logits["logits_utilization"][0].cpu())
    adh_p  = torch.sigmoid(logits["logits_adherence"][0].cpu())

    df = pd.DataFrame({
        "token": tokens,
        "is_ctx": batch["context_mask"].cpu().numpy().astype(int),
        "is_resp": batch["response_mask"].cpu().numpy().astype(int),
        "rel_prob": rel_p.numpy(),
        "util_prob": util_p.numpy(),
        "adh_prob": adh_p.numpy(),
        "rel_true": batch["labels_relevance"].cpu().numpy(),
        "util_true": batch["labels_utilization"].cpu().numpy(),
        "adh_true": batch["labels_adherence"].cpu().numpy(),
    })

    prob_round = st.slider("Округление вероятностей", 2, 4, 3)
    only_ctx   = st.checkbox("Показать только контекстные токены", value=False)
    only_resp  = st.checkbox("Показать только токены ответа", value=False)

    view = df.copy()
    if only_ctx:
        view = view[view["is_ctx"] == 1]
    if only_resp:
        view = view[view["is_resp"] == 1]
    for c in ["rel_prob", "util_prob", "adh_prob"]:
        view[c] = view[c].round(prob_round)

    st.dataframe(view, use_container_width=True)

st.caption("Модель и токенайзер всегда грузятся из локального чекпоинта (`save_pretrained`). "
           "Веса голов — из `state_dict`. Инференс строго по одному элементу.")
