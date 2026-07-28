"""Graph and feature-matrix loading.

Everything here is loaded ONCE per run and cached:
  - edge lists -> normalized sparse adjacency (torch.sparse, on device)
  - the 0/1 keyword feature matrix (text file) -> .pt sparse cache
  - qid -> node-id mapping used to align mentions with graph nodes
"""

import json
import os

import numpy as np
import torch


def load_qid_mapping(path: str) -> dict:
    """qid_to_node.txt lines: '<qid> <node_id>'."""
    mapping = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 2:
                mapping[parts[0]] = int(parts[1])
    return mapping


def load_edges(path: str) -> np.ndarray:
    edges = np.loadtxt(path, dtype=np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    return edges


def build_norm_adj(edges: np.ndarray, num_nodes: int,
                   normalization: str = "row") -> torch.Tensor:
    """Symmetrize, add self-loops, normalize; returns sparse COO tensor.

    normalization:
      - 'row':       D^-1 (A + I)
      - 'symmetric': D^-1/2 (A + I) D^-1/2  (paper Eq. 6/7)
    """
    src = np.concatenate([edges[:, 0], edges[:, 1], np.arange(num_nodes)])
    dst = np.concatenate([edges[:, 1], edges[:, 0], np.arange(num_nodes)])
    pairs = np.unique(np.stack([src, dst], axis=1), axis=0)
    src, dst = pairs[:, 0], pairs[:, 1]
    values = np.ones(len(src), dtype=np.float32)

    degree = np.zeros(num_nodes, dtype=np.float32)
    np.add.at(degree, src, values)
    inv = np.zeros_like(degree)
    nz = degree > 0
    if normalization == "symmetric":
        inv[nz] = degree[nz] ** -0.5
        values = values * inv[src] * inv[dst]
    elif normalization == "row":
        inv[nz] = degree[nz] ** -1.0
        values = values * inv[src]
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    indices = torch.from_numpy(np.stack([src, dst]))
    return torch.sparse_coo_tensor(
        indices, torch.from_numpy(values), (num_nodes, num_nodes)
    ).coalesce()


def load_feature_matrix(txt_path: str, cache_path: str | None = None) -> torch.Tensor:
    """Parse the n x m 0/1 keyword matrix into a sparse tensor (~2%
    density, so sparse saves both memory and the first GCN matmul). The
    text file is ~60 MB; the parsed indices are cached to a small .pt."""
    if cache_path is None:
        cache_path = txt_path + ".cache.pt"
    src_size = os.path.getsize(txt_path)
    indices = None
    if os.path.exists(cache_path):
        payload = torch.load(cache_path, weights_only=True)
        if int(payload.get("src_size", -1)) == src_size:
            indices, shape = payload["indices"], payload["shape"]

    if indices is None:
        rows, cols = [], []
        n_cols = None
        with open(txt_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                row = np.frombuffer(bytearray(line.replace(" ", ""), "ascii"),
                                    dtype=np.uint8) - ord("0")
                if n_cols is None:
                    n_cols = len(row)
                nz = np.flatnonzero(row)
                rows.extend([i] * len(nz))
                cols.extend(nz.tolist())
        shape = torch.tensor([i + 1, n_cols])
        indices = torch.tensor([rows, cols], dtype=torch.long)
        torch.save({"shape": shape, "indices": indices,
                    "src_size": torch.tensor(src_size)}, cache_path)

    values = torch.ones(indices.size(1), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, shape.tolist()).coalesce()


def validate_alignment(documents_path: str, qid_mapping: dict,
                       num_check: int | None = None) -> None:
    """The structure graph indexes nodes by qid_to_node while the feature
    matrix / kNN graph follow the document line order. These must agree;
    fail loudly if they do not."""
    with open(documents_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if num_check is not None and i >= num_check:
                break
            qid = json.loads(line)["qid"]
            node = qid_mapping.get(qid)
            if node != i:
                raise RuntimeError(
                    f"Graph/document alignment broken at line {i}: "
                    f"document qid {qid} maps to node {node}. "
                    "The structure-graph node ids do not follow the "
                    "document order; regenerate the processed data."
                )


class GraphBundle:
    """Everything the model needs about one training year's graphs."""

    def __init__(self, year_dir: str, documents_path: str, graph_cfg: dict,
                 device: torch.device):
        knn_k = graph_cfg.get("knn_k", 3)
        normalization = graph_cfg.get("normalization", "row")

        self.qid_to_node = load_qid_mapping(os.path.join(year_dir, "qid_to_node.txt"))
        validate_alignment(documents_path, self.qid_to_node)

        features = load_feature_matrix(os.path.join(year_dir, "feature_matrix.txt"))
        num_nodes = features.size(0)
        if len(self.qid_to_node) != num_nodes:
            raise RuntimeError(
                f"{len(self.qid_to_node)} qids in mapping but feature matrix "
                f"has {num_nodes} rows"
            )

        struct_edges = load_edges(os.path.join(year_dir, "structure_edges.txt"))
        knn_edges = load_edges(os.path.join(year_dir, f"knn_edges_k{knn_k}.txt"))

        self.features = features.to(device)
        self.sadj = build_norm_adj(struct_edges, num_nodes, normalization).to(device)
        self.fadj = build_norm_adj(knn_edges, num_nodes, normalization).to(device)
        self.num_nodes = num_nodes
        self.nfeat = features.size(1)

    def node_ids_for(self, qids: list[str]) -> torch.Tensor:
        """-1 for qids that are not graph nodes."""
        return torch.tensor([self.qid_to_node.get(q, -1) for q in qids],
                            dtype=torch.long)
