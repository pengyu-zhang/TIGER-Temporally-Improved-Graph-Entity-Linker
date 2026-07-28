"""Train one (config, variant, year) combination.

Example:
  python -m tiger.train --config configs/default.yaml \
      --variant new_entities --year 2019 --output-dir outputs/default/new_entities/2019
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from torch.utils.data import TensorDataset

from .config import load_config
from .data import process_mention_data, tensorize_file_cached
from .graph import GraphBundle
from .model import TigerModel, load_tokenizer
from .optim import build_optimizer, build_scheduler
from .utils import (JsonlWriter, configure_backends, get_device, get_logger,
                    read_jsonl, set_seed)


def paths_for(cfg: dict, variant: str, year: int) -> dict:
    root = cfg["data"]["root"]
    return {
        "train": os.path.join(root, "mentions", variant, str(year), "train.jsonl"),
        "validation": os.path.join(root, "mentions", "all", str(year), "validation.jsonl"),
        # graph node order follows the TRAIN document file of the year
        "documents": os.path.join(root, "documents", f"{year}_train.json"),
        "graph_dir": os.path.join(root, "graphs", str(year)),
    }


@torch.no_grad()
def in_batch_eval(model, dataset, device, group_size, encode_batch_size,
                  amp_dtype=None):
    """In-batch-negative accuracy on the validation set (a
    training-time sanity metric): each consecutive group of `group_size`
    samples scores its contexts against its own candidates.

    Encoding runs at `encode_batch_size` for throughput; scoring then
    happens per group, which is numerically identical to encoding group
    by group.
    """
    model.eval()
    if device.type == "cuda":
        # release blocks cached during training before the workload changes
        torch.cuda.empty_cache()
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset),
                        batch_size=encode_batch_size)
    ctxt_parts, cand_parts = [], []
    for batch in loader:
        with torch.autocast(device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            ctxt_parts.append(model.encode_context(batch[0].to(device)).float().cpu())
            cand_parts.append(model.encode_candidates(batch[1].to(device)).float().cpu())
    ctxt = torch.cat(ctxt_parts).to(device)
    cands = torch.cat(cand_parts).to(device)

    correct, total = 0, 0
    for start in range(0, ctxt.size(0), group_size):
        group = slice(start, min(start + group_size, ctxt.size(0)))
        scores = ctxt[group].mm(cands[group].t())
        preds = scores.argmax(dim=1)
        labels = torch.arange(scores.size(0), device=scores.device)
        correct += int((preds == labels).sum())
        total += scores.size(0)
    model.train()
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default=None,
                        help="new_entities | continual_entities (default from config)")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if a finished checkpoint already exists")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="Config override, e.g. --set train.batch_size=32")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    variant = args.variant or cfg["data"].get("variant", "new_entities")
    out_dir = args.output_dir
    ckpt_path = os.path.join(out_dir, "pytorch_model.bin")
    done_path = os.path.join(out_dir, "DONE")
    if args.skip_existing and os.path.exists(done_path):
        print(f"[tiger] {done_path} exists, skipping")
        return

    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(out_dir)
    set_seed(cfg["seed"])
    configure_backends(cfg.get("tf32", False))
    device = get_device(cfg.get("no_cuda", False))

    train_cfg = cfg["train"]
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(train_cfg.get("amp"))

    paths = paths_for(cfg, variant, args.year)
    tokenizer = load_tokenizer(cfg)

    graph_bundle = None
    if (cfg.get("graph") or {}).get("enabled"):
        graph_bundle = GraphBundle(paths["graph_dir"], paths["documents"],
                                   cfg["graph"], device)
        logger.info("graph: %d nodes, nfeat=%d", graph_bundle.num_nodes, graph_bundle.nfeat)

    train_samples = read_jsonl(paths["train"], args.max_train_samples)
    logger.info("read %d train samples (%s %d)", len(train_samples), variant, args.year)
    train_data = process_mention_data(
        train_samples, tokenizer, cfg["data"],
        qid_to_node=graph_bundle.qid_to_node if graph_bundle else None,
    )
    sampler = (RandomSampler(train_data) if train_cfg.get("shuffle")
               else SequentialSampler(train_data))
    train_loader = DataLoader(train_data, sampler=sampler,
                              batch_size=train_cfg["batch_size"], drop_last=False)

    valid_data = None
    if train_cfg.get("eval_on_validation", True):
        # The in-batch validation metric is diagnostic only (it never
        # affects the trained weights), so it may run on a subsample;
        # CLI --max-valid-samples overrides train.validation_max_samples.
        max_valid = args.max_valid_samples
        if max_valid is None:
            max_valid = train_cfg.get("validation_max_samples")
        if max_valid is not None:
            valid_samples = read_jsonl(paths["validation"], max_valid)
            valid_data = process_mention_data(valid_samples, tokenizer, cfg["data"],
                                              qid_to_node=None)
            valid_data = TensorDataset(valid_data.tensors[0], valid_data.tensors[1])
        else:
            ctx, cand, _ = tensorize_file_cached(
                paths["validation"], tokenizer, cfg["data"],
                cfg["model"]["bert_model"],
                cfg["data"].get("cache_dir", "data/cache"),
                include_candidates=True)
            valid_data = TensorDataset(ctx, cand)
        logger.info("validation set: %d samples", len(valid_data))

    model = TigerModel(cfg, graph_bundle).to(device)
    model.train()

    grad_acc = train_cfg.get("gradient_accumulation_steps", 1)
    epochs = train_cfg["num_epochs"]
    steps_per_epoch = -(-len(train_loader) // grad_acc)
    num_train_steps = steps_per_epoch * epochs
    optimizer = build_optimizer(model, train_cfg, cfg.get("graph"))
    scheduler = build_scheduler(optimizer, num_train_steps,
                                train_cfg.get("warmup_proportion", 0.1))

    metrics = JsonlWriter(os.path.join(out_dir, "metrics.jsonl"))
    metrics.write({"event": "config", "config": cfg, "variant": variant,
                   "year": args.year})

    eval_encode_bs = train_cfg.get("eval_encode_batch_size", 256)
    if train_cfg.get("eval_before_training", False) and valid_data is not None:
        acc = in_batch_eval(model, valid_data, device,
                            train_cfg["eval_batch_size"], eval_encode_bs, amp_dtype)
        logger.info("in-batch validation accuracy before training: %.5f", acc)
        metrics.write({"event": "eval", "when": "before", "in_batch_accuracy": acc})

    start = time.time()
    global_step = 0
    for epoch in range(epochs):
        running = 0.0
        for step, batch in enumerate(train_loader):
            context_input = batch[0].to(device)
            cand_input = batch[1].to(device)
            node_ids = batch[3].to(device)
            with torch.autocast(device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                loss, _, parts = model(context_input, cand_input, node_ids)
            if grad_acc > 1:
                loss = loss / grad_acc
            loss.backward()
            running += float(loss.detach())

            if (step + 1) % grad_acc == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               train_cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                metrics.write({"event": "step", "epoch": epoch, "step": global_step,
                               "loss": float(loss.detach()) * grad_acc,
                               "lr": scheduler.get_last_lr()[0], **parts})

        logger.info("epoch %d mean loss %.5f", epoch, running / max(1, len(train_loader)))

    if valid_data is not None:
        acc = in_batch_eval(model, valid_data, device,
                            train_cfg["eval_batch_size"], eval_encode_bs, amp_dtype)
        logger.info("in-batch validation accuracy after training: %.5f", acc)
        metrics.write({"event": "eval", "when": "after", "in_batch_accuracy": acc})

    model.save(ckpt_path)
    with open(os.path.join(out_dir, "training_params.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": cfg, "variant": variant, "year": args.year,
                   "train_minutes": (time.time() - start) / 60}, fh, indent=2)
    with open(done_path, "w") as fh:
        fh.write("ok\n")
    metrics.write({"event": "done", "minutes": (time.time() - start) / 60})
    metrics.close()
    logger.info("saved %s (%.1f min)", ckpt_path, (time.time() - start) / 60)


if __name__ == "__main__":
    main()
