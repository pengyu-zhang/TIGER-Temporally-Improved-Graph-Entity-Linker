"""Evaluate a trained checkpoint on one or more test years.

For each test year: encode that year's candidate pool once (cached next
to the checkpoint), encode all test mentions, compute recall@k over the
full pool. Inference uses text only — the graph channel never runs here,
matching the paper.

Example:
  python -m tiger.evaluate --config configs/default.yaml \
      --model-dir outputs/default/new_entities/2019 \
      --test-years 2019,2020,2021,2022
"""

import argparse
import hashlib
import json
import os
import time

import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from .config import load_config
from .data import build_candidate_pool, tensorize_file_cached
from .metrics import Stats
from .model import TigerModel, load_tokenizer
from .utils import JsonlWriter, configure_backends, get_device, get_logger, set_seed


@torch.no_grad()
def encode_pool(model, pool_tokens, device, batch_size, amp_dtype=None):
    encodings = []
    loader = DataLoader(pool_tokens, batch_size=batch_size)
    for batch in tqdm(loader, desc="encode pool", leave=False):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            enc = model.encode_candidates(batch.to(device))
        encodings.append(enc.float().cpu())
    return torch.cat(encodings)


@torch.no_grad()
def evaluate_year(model, tokenizer, cfg, test_path, documents_path, device,
                  enc_cache_dir, data_cache_dir,
                  max_eval_samples=None, max_pool=None):
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg["data"]
    model_name = cfg["model"]["bert_model"]
    top_k = eval_cfg.get("top_k", 64)
    encode_bs = eval_cfg.get("encode_batch_size", 256)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(eval_cfg.get("amp"))

    # ---- candidate pool: tokens cached per (file, settings); encodings
    #      cached per checkpoint ----
    key = json.dumps([documents_path, data_cfg.get("use_title_in_pool", False),
                      data_cfg.get("pool_text_limit", 128),
                      data_cfg["max_cand_length"], max_pool,
                      eval_cfg.get("amp"), cfg.get("tf32", False)], sort_keys=True)
    cache_tag = hashlib.md5(key.encode()).hexdigest()[:10]
    cache_path = os.path.join(enc_cache_dir, f"pool_{cache_tag}.pt")
    if eval_cfg.get("cache_pool_encodings", True) and os.path.exists(cache_path):
        pool_enc = torch.load(cache_path, weights_only=True)
    else:
        pool_tokens = build_candidate_pool(documents_path, tokenizer, data_cfg,
                                           max_pool=max_pool, model_name=model_name,
                                           cache_dir=data_cache_dir)
        pool_enc = encode_pool(model, pool_tokens, device, encode_bs, amp_dtype)
        if eval_cfg.get("cache_pool_encodings", True):
            os.makedirs(enc_cache_dir, exist_ok=True)
            torch.save(pool_enc, cache_path)
    pool_size = pool_enc.size(0)
    pool_enc = pool_enc.to(device)

    # ---- mentions (tokenized tensors cached per file+settings) ----
    context_ids, _, label_idx = tensorize_file_cached(
        test_path, tokenizer, data_cfg, model_name, data_cache_dir,
        include_candidates=False)
    if max_eval_samples is not None:
        context_ids, label_idx = context_ids[:max_eval_samples], label_idx[:max_eval_samples]
    if max_pool is not None:
        keep = label_idx < max_pool
        context_ids, label_idx = context_ids[keep], label_idx[keep]
    dataset = TensorDataset(context_ids, label_idx)
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset),
                        batch_size=encode_bs)

    stats = Stats(top_k)
    for context_input, label_ids in tqdm(loader, desc="score mentions", leave=False):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            ctxt = model.encode_context(context_input.to(device))
        scores = ctxt.float() @ pool_enc.t()
        _, top_ids = scores.topk(min(top_k, pool_size), dim=1)
        # gold rank per row, -1 if absent (vectorized)
        matches = top_ids == label_ids.to(device).unsqueeze(1)
        found = matches.any(dim=1)
        ranks = torch.where(found, matches.float().argmax(dim=1),
                            torch.full_like(label_ids.to(device), -1))
        for rank in ranks.cpu().tolist():
            stats.add(int(rank))
    return stats, pool_size, len(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="Directory containing pytorch_model.bin")
    parser.add_argument("--test-years", required=True,
                        help="Comma-separated, e.g. 2019,2020,2021,2022")
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-pool", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    logger = get_logger(args.model_dir)
    set_seed(cfg["seed"])
    configure_backends(cfg.get("tf32", False))
    device = get_device(cfg.get("no_cuda", False))

    tokenizer = load_tokenizer(cfg)
    # Inference is text-only: build the model without the graph channel.
    eval_model_cfg = dict(cfg)
    eval_model_cfg["graph"] = {"enabled": False}
    model = TigerModel(eval_model_cfg).to(device)
    model.load(os.path.join(args.model_dir, "pytorch_model.bin"), device)
    model.eval()

    root = cfg["data"]["root"]
    results_path = os.path.join(args.model_dir, f"results_{args.split}.jsonl")
    metrics = JsonlWriter(results_path)
    for year in [int(y) for y in args.test_years.split(",")]:
        test_path = os.path.join(root, "mentions", "all", str(year),
                                 f"{args.split}.jsonl")
        # label_id indexes into the SPLIT-specific document order
        documents_path = os.path.join(root, "documents", f"{year}_{args.split}.json")
        t0 = time.time()
        stats, pool_size, n = evaluate_year(
            model, tokenizer, cfg, test_path, documents_path, device,
            enc_cache_dir=os.path.join(args.model_dir, "pool_cache"),
            data_cache_dir=cfg["data"].get("cache_dir", "data/cache"),
            max_eval_samples=args.max_eval_samples, max_pool=args.max_pool,
        )
        logger.info("test year %d (pool %d, mentions %d): %s [%.1f min]",
                    year, pool_size, n, stats.output(), (time.time() - t0) / 60)
        metrics.write({"event": "result", "split": args.split, "test_year": year,
                       "pool_size": pool_size, "num_mentions": n,
                       "minutes": (time.time() - t0) / 60, **stats.recalls()})
    metrics.close()
    logger.info("results written to %s", results_path)


if __name__ == "__main__":
    main()
