"""Graph convolution modules (following AM-GCN) plus memory-bounded
implementations of the consistency (common) and HSIC (distinct) losses.

A naive implementation materializes full n x n Gram matrices for both
losses; with n ~ 10k nodes that is ~430 MB per matrix. The chunked
versions below compute identical values while only ever holding a
(chunk x n) block.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Single GCN layer with AM-GCN's uniform initialization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            support = torch.sparse.mm(x, self.weight)
        else:
            support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GCN(nn.Module):
    def __init__(self, nfeat: int, nhid: int, nout: int, dropout: float):
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


class Attention(nn.Module):
    """AM-GCN attention over the three channel embeddings."""

    def __init__(self, in_size: int, hidden_size: int = 16):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class SFGCN(nn.Module):
    """Two Distinct Convolution Modules (SGCN1 on the structure graph,
    SGCN2 on the feature graph) and one Shared Convolution Module (CGCN,
    parameters shared across both graphs)."""

    def __init__(self, nfeat: int, nhid1: int, nhid2: int, dropout: float,
                 combine: str = "sum"):
        super().__init__()
        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.combine = combine
        self.attention = Attention(nhid2)

    def forward(self, x, sadj, fadj):
        emb1 = self.SGCN1(x, sadj)   # distinct, structure graph
        com1 = self.CGCN(x, sadj)    # shared, structure graph
        com2 = self.CGCN(x, fadj)    # shared, feature graph
        emb2 = self.SGCN2(x, fadj)   # distinct, feature graph
        if self.combine == "attention":
            xcom = (com1 + com2) / 2
            emb, _ = self.attention(torch.stack([emb1, emb2, xcom], dim=1))
        else:  # "sum": plain elementwise combination
            emb = emb1 + emb2 + com1 + com2
        return emb, emb1, com1, com2, emb2


def common_loss_chunked(emb1: torch.Tensor, emb2: torch.Tensor,
                        chunk_rows: int = 2048) -> torch.Tensor:
    """mean((cov1 - cov2)^2) over the n x n covariance matrices of the
    centered, L2-normalized embeddings, computed in row blocks."""
    emb1 = F.normalize(emb1 - emb1.mean(dim=0, keepdim=True), p=2, dim=1)
    emb2 = F.normalize(emb2 - emb2.mean(dim=0, keepdim=True), p=2, dim=1)
    n = emb1.size(0)
    total = emb1.new_zeros(())
    for start in range(0, n, chunk_rows):
        rows = slice(start, min(start + chunk_rows, n))
        diff = emb1[rows] @ emb1.T - emb2[rows] @ emb2.T
        total = total + (diff * diff).sum()
    return total / (n * n)


def hsic_loss_chunked(emb1: torch.Tensor, emb2: torch.Tensor,
                      paper_scaling: bool = False,
                      chunk_rows: int = 2048) -> torch.Tensor:
    """HSIC with inner-product kernels: tr(R K1 R K2) with
    R = I - (1/n) 11^T, computed as sum(center(K1) * K2) in row blocks.

    paper_scaling adds the (n-1)^-2 factor from the paper's Eq. 10/11;
    when disabled, the factor is absorbed into the loss weight.
    """
    n = emb1.size(0)
    col_sum1 = emb1.sum(dim=0)                       # K1 column sums = Z1 @ col_sum1
    row_means1 = (emb1 @ col_sum1) / n               # per-row mean of K1
    grand_mean1 = row_means1.sum() / n
    total = emb1.new_zeros(())
    for start in range(0, n, chunk_rows):
        rows = slice(start, min(start + chunk_rows, n))
        k1 = emb1[rows] @ emb1.T
        k1 = k1 - row_means1[rows].unsqueeze(1) - row_means1.unsqueeze(0) + grand_mean1
        k2 = emb2[rows] @ emb2.T
        total = total + (k1 * k2).sum()
    if paper_scaling:
        total = total / float((n - 1) ** 2)
    return total
