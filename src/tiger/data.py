"""Mention / candidate tensorization.

Graph-TempEL ships pre-tokenized fields (lists of WordPiece tokens), so
by default no tokenizer.tokenize call is needed for contexts and
candidate descriptions — only vocabulary lookups. Raw-text datasets
(e.g. ZESHEL) set data.pretokenized: false and go through the
tokenizer, matching BLINK's data_process_zeshel.py path.
"""

import hashlib
import json
import os

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

ENT_START_TAG = "[unused0]"
ENT_END_TAG = "[unused1]"
ENT_TITLE_TAG = "[unused2]"


def _cache_key(source_path: str, data_cfg: dict, model_name: str,
               extra: dict | None = None) -> str:
    """Cache key for tokenized tensors: source file identity (path,
    size, mtime) plus every setting that changes the tokenization."""
    stat = os.stat(source_path)
    payload = {
        "path": os.path.abspath(source_path),
        "size": stat.st_size,
        "mtime": int(stat.st_mtime),
        "max_context_length": data_cfg["max_context_length"],
        "max_cand_length": data_cfg["max_cand_length"],
        "pretokenized": data_cfg.get("pretokenized", True),
        "model": model_name,
        **(extra or {}),
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def get_context_ids(sample: dict, tokenizer, max_seq_length: int,
                    pretokenized: bool = True) -> list[int]:
    """[CLS] left [Ms] mention [Me] right [SEP], padded to max_seq_length.

    Follows BLINK's get_context_representation with one fix: a plain
    context_left[-left_quota:] slice grabs the WHOLE left context when
    left_quota == 0, so that case is guarded here.
    """
    mention_tokens = []
    if sample.get("mention"):
        mention_tokens = tokenizer.tokenize(sample["mention"])
        mention_tokens = [ENT_START_TAG] + mention_tokens + [ENT_END_TAG]

    if pretokenized:
        context_left = sample["context_left"]
        context_right = sample["context_right"]
    else:
        context_left = tokenizer.tokenize(sample["context_left"])
        context_right = tokenizer.tokenize(sample["context_right"])

    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    if len(context_left) <= left_quota:
        if len(context_right) > right_quota:
            right_quota += left_quota - len(context_left)
    else:
        if len(context_right) <= right_quota:
            left_quota += right_quota - len(context_right)
    left_quota = max(0, left_quota)
    right_quota = max(0, right_quota)

    left_part = context_left[-left_quota:] if left_quota > 0 else []
    context_tokens = left_part + mention_tokens + context_right[:right_quota]
    context_tokens = ["[CLS]"] + context_tokens[: max_seq_length - 2] + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    input_ids += [0] * (max_seq_length - len(input_ids))
    assert len(input_ids) == max_seq_length
    return input_ids


def get_candidate_ids(candidate_desc, tokenizer, max_seq_length: int,
                      candidate_title: str | None = None,
                      pretokenized: bool = True) -> list[int]:
    """[CLS] title [unused2] description [SEP], padded to max_seq_length."""
    cand_tokens = candidate_desc if pretokenized else tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        cand_tokens = tokenizer.tokenize(candidate_title) + [ENT_TITLE_TAG] + list(cand_tokens)
    cand_tokens = list(cand_tokens)[: max_seq_length - 2]
    cand_tokens = [tokenizer.cls_token] + cand_tokens + [tokenizer.sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    input_ids += [0] * (max_seq_length - len(input_ids))
    assert len(input_ids) == max_seq_length
    return input_ids


def process_mention_data(samples: list[dict], tokenizer, data_cfg: dict,
                         qid_to_node: dict | None = None,
                         silent: bool = False) -> TensorDataset:
    """Returns TensorDataset(context_ids, cand_ids, label_idx, node_ids).

    node_ids maps each sample's gold entity to its graph node (or -1);
    all -1 when no graph is in use.
    """
    max_ctx = data_cfg["max_context_length"]
    max_cand = data_cfg["max_cand_length"]
    pretokenized = data_cfg.get("pretokenized", True)

    contexts, cands, labels, nodes = [], [], [], []
    iter_ = samples if silent else tqdm(samples, desc="tensorize", leave=False)
    for sample in iter_:
        contexts.append(get_context_ids(sample, tokenizer, max_ctx, pretokenized))
        cands.append(get_candidate_ids(
            sample["label"], tokenizer, max_cand,
            sample.get("label_title"), pretokenized,
        ))
        labels.append(int(sample["label_id"]))
        if qid_to_node is not None:
            nodes.append(qid_to_node.get(sample.get("qid"), -1))
        else:
            nodes.append(-1)

    return TensorDataset(
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(cands, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(nodes, dtype=torch.long),
    )


def tensorize_file_cached(jsonl_path: str, tokenizer, data_cfg: dict,
                          model_name: str, cache_dir: str | None,
                          include_candidates: bool = True) -> tuple:
    """Tokenize a full mention file into tensors, caching the result.
    Tokenization is identical across models/seeds/configs that share the
    same lengths, so the evaluation matrix reuses one cache per file.

    Returns (context_ids, cand_ids or None, label_idx).
    """
    from .utils import read_jsonl

    if cache_dir is not None:
        key = _cache_key(jsonl_path, data_cfg, model_name,
                         {"include_candidates": include_candidates})
        cache_path = os.path.join(cache_dir, f"tensors_{key}.pt")
        if os.path.exists(cache_path):
            payload = torch.load(cache_path, weights_only=True)
            return payload["context"], payload.get("cand"), payload["label"]

    samples = read_jsonl(jsonl_path)
    max_ctx = data_cfg["max_context_length"]
    max_cand = data_cfg["max_cand_length"]
    pretokenized = data_cfg.get("pretokenized", True)

    contexts, cands, labels = [], [], []
    for sample in tqdm(samples, desc="tensorize", leave=False):
        contexts.append(get_context_ids(sample, tokenizer, max_ctx, pretokenized))
        if include_candidates:
            cands.append(get_candidate_ids(
                sample["label"], tokenizer, max_cand,
                sample.get("label_title"), pretokenized))
        labels.append(int(sample["label_id"]))

    context_ids = torch.tensor(contexts, dtype=torch.long)
    cand_ids = torch.tensor(cands, dtype=torch.long) if include_candidates else None
    label_idx = torch.tensor(labels, dtype=torch.long)

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        payload = {"context": context_ids, "label": label_idx}
        if cand_ids is not None:
            payload["cand"] = cand_ids
        torch.save(payload, cache_path)
    return context_ids, cand_ids, label_idx


def build_candidate_pool(documents_path: str, tokenizer, data_cfg: dict,
                         max_pool: int | None = None,
                         model_name: str = "",
                         cache_dir: str | None = None,
                         silent: bool = False) -> torch.Tensor:
    """Tokenized candidate pool from a per-year documents file.

    Default eval protocol: description truncated to pool_text_limit
    tokens, and (by default) NO title — the candidate pool uses text
    only, unlike training candidates.
    Set data.use_title_in_pool: true for the consistent representation.
    """
    use_title = data_cfg.get("use_title_in_pool", False)
    text_limit = data_cfg.get("pool_text_limit", 128)
    max_cand = data_cfg["max_cand_length"]
    pretokenized = data_cfg.get("pretokenized", True)

    cache_path = None
    if cache_dir is not None:
        key = _cache_key(documents_path, data_cfg, model_name,
                         {"use_title": use_title, "text_limit": text_limit})
        cache_path = os.path.join(cache_dir, f"pool_tokens_{key}.pt")
        if os.path.exists(cache_path):
            tokens = torch.load(cache_path, weights_only=True)
            return tokens[:max_pool] if max_pool is not None else tokens

    pool = []
    with open(documents_path, "r", encoding="utf-8") as fh:
        iter_ = fh if silent else tqdm(fh, desc="candidate pool", leave=False)
        for line in iter_:
            doc = json.loads(line)
            text = doc["text"][:text_limit]
            title = doc.get("title") if use_title else None
            pool.append(get_candidate_ids(text, tokenizer, max_cand, title, pretokenized))

    tokens = torch.tensor(pool, dtype=torch.long)
    if cache_path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(tokens, cache_path)
    return tokens[:max_pool] if max_pool is not None else tokens
