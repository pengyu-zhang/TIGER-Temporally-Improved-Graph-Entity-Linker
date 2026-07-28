"""TIGER model: BLINK bi-encoder plus (optionally) the graph channel.

Design notes:
  - The SFGCN is constructed ONCE and trained jointly with the
    bi-encoder.
  - Graph embeddings are aligned to batch entities BY QID via
    node_ids, so sample order never needs to match graph node order.
  - The fusion score `tanh(y_e @ Z^T) * scale` needs matching
    dimensions; a linear projection maps the GCN output (nhid2) to the
    encoder width.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .gcn import SFGCN, common_loss_chunked, hsic_loss_chunked


def to_bert_input(token_idx: torch.Tensor, null_idx: int = 0):
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask


class BertEncoder(nn.Module):
    """CLS-vector encoder (out_dim projection unused in all TIGER
    configurations, kept for BLINK compatibility)."""

    def __init__(self, bert_model, output_dim: int, add_linear: bool = False):
        super().__init__()
        self.bert_model = bert_model
        bert_dim = bert_model.embeddings.word_embeddings.weight.size(1)
        self.out_dim = output_dim if add_linear else bert_dim
        if add_linear:
            self.additional_linear = nn.Linear(bert_dim, output_dim)
            self.dropout = nn.Dropout(0.1)
        else:
            self.additional_linear = None

    def forward(self, token_ids, segment_ids, attention_mask):
        output = self.bert_model(
            input_ids=token_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask,
        )
        if self.additional_linear is not None:
            return self.additional_linear(self.dropout(output.pooler_output))
        # .contiguous() detaches the CLS slice from the full [B, L, H]
        # hidden-state buffer; returning the raw view would keep that
        # whole buffer alive whenever callers accumulate encodings.
        return output.last_hidden_state[:, 0, :].contiguous()


class TigerModel(nn.Module):
    def __init__(self, cfg: dict, graph_bundle=None):
        super().__init__()
        model_cfg = cfg["model"]
        self.graph_cfg = cfg.get("graph", {}) or {}
        self.graph_enabled = bool(self.graph_cfg.get("enabled")) and graph_bundle is not None

        ctxt_bert = AutoModel.from_pretrained(model_cfg["bert_model"])
        cand_bert = AutoModel.from_pretrained(model_cfg["bert_model"])
        add_linear = model_cfg.get("add_linear", False)
        self.context_encoder = BertEncoder(ctxt_bert, model_cfg.get("out_dim", 1), add_linear)
        self.cand_encoder = BertEncoder(cand_bert, model_cfg.get("out_dim", 1), add_linear)

        if self.graph_enabled:
            self.graph = graph_bundle  # holds features/adjacency on device
            self.gcn = SFGCN(
                nfeat=graph_bundle.nfeat,
                nhid1=self.graph_cfg.get("nhid1", 768),
                nhid2=self.graph_cfg.get("nhid2", 256),
                dropout=self.graph_cfg.get("dropout", 0.5),
                combine=self.graph_cfg.get("combine", "sum"),
            )
            fusion = self.graph_cfg.get("fusion", {}) or {}
            self.fusion_mode = fusion.get("mode", "score_add")
            self.fusion_scale = float(fusion.get("scale", 40.0))
            self.fusion_tanh = bool(fusion.get("tanh", True))
            if self.fusion_mode == "score_add":
                self.fusion_proj = nn.Linear(
                    self.graph_cfg.get("nhid2", 256),
                    self.cand_encoder.out_dim,
                    bias=False,
                )
        else:
            self.graph = None

    # ---------------- encoding ----------------

    def encode_context(self, token_idx: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(*to_bert_input(token_idx))

    def encode_candidates(self, token_idx: torch.Tensor) -> torch.Tensor:
        return self.cand_encoder(*to_bert_input(token_idx))

    # ---------------- training ----------------

    def gcn_forward(self):
        return self.gcn(self.graph.features, self.graph.sadj, self.graph.fadj)

    def forward(self, context_input, cand_input, node_ids=None):
        """In-batch-negative training step. Returns (loss, scores, parts)
        where parts is a dict of individual loss terms for logging."""
        embedding_ctxt = self.encode_context(context_input)
        embedding_cands = self.encode_candidates(cand_input)
        scores = embedding_ctxt.mm(embedding_cands.t())

        parts = {}
        aux_loss = scores.new_zeros((), dtype=torch.float32)
        if self.graph_enabled:
            # Sparse matmul has no half/bf16 CUDA kernels and the n x n
            # loss terms want full precision: run the whole graph channel
            # in fp32, outside any surrounding autocast region.
            with torch.autocast(device_type=scores.device.type, enabled=False):
                emb, emb1, com1, com2, emb2 = self.gcn_forward()

                if self.fusion_mode == "score_add" and node_ids is not None:
                    valid = node_ids >= 0
                    z = emb.new_zeros(node_ids.size(0), emb.size(1))
                    if valid.any():
                        z[valid] = emb[node_ids[valid]]
                    z = self.fusion_proj(z)
                    scores2 = embedding_cands.float().mm(z.t())
                    if self.fusion_tanh:
                        scores2 = torch.tanh(scores2) * self.fusion_scale
                    scores = scores.float() + scores2

                loss_cfg = self.graph_cfg.get("loss", {}) or {}
                a = float(loss_cfg.get("a_common", 0.5))
                b = float(loss_cfg.get("b_hsic", 0.0))
                chunk = int(loss_cfg.get("chunk_rows", 2048))
                scaling = bool(loss_cfg.get("hsic_paper_scaling", False))
                if a != 0.0:
                    loss_common = common_loss_chunked(com1, com2, chunk)
                    parts["loss_common"] = float(loss_common.detach())
                    aux_loss = aux_loss + a * loss_common
                if b != 0.0:
                    loss_hsic = 0.5 * (
                        hsic_loss_chunked(emb1, com1, scaling, chunk)
                        + hsic_loss_chunked(emb2, com2, scaling, chunk)
                    )
                    parts["loss_hsic"] = float(loss_hsic.detach())
                    aux_loss = aux_loss + b * loss_hsic

        target = torch.arange(scores.size(0), device=scores.device)
        loss_el = F.cross_entropy(scores, target, reduction="mean")
        parts["loss_el"] = float(loss_el.detach())
        return loss_el + aux_loss, scores, parts

    # ---------------- persistence ----------------

    def trained_state_dict(self) -> dict:
        """Everything except the (non-parameter) graph tensors."""
        return {k: v for k, v in self.state_dict().items()}

    def save(self, path: str) -> None:
        torch.save(self.trained_state_dict(), path)

    def load(self, path: str, device) -> None:
        state = torch.load(path, map_location=device, weights_only=True)
        missing, unexpected = self.load_state_dict(state, strict=False)
        # The bi-encoder weights must always load; GCN weights are absent
        # when evaluating a graph-enabled checkpoint with graph disabled.
        critical = [k for k in missing if k.startswith(("context_encoder", "cand_encoder"))]
        if critical or unexpected and any(
            k.startswith(("context_encoder", "cand_encoder")) for k in unexpected
        ):
            raise RuntimeError(f"Bad checkpoint: missing={critical} unexpected={unexpected}")


def load_tokenizer(cfg: dict):
    model_cfg = cfg["model"]
    return AutoTokenizer.from_pretrained(
        model_cfg["bert_model"], do_lower_case=model_cfg.get("lowercase", True)
    )
