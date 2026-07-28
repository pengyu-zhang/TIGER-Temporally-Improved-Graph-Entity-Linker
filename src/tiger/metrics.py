"""Recall@k accounting for the full-pool evaluation protocol."""

RANKS = (1, 2, 4, 8, 16, 32, 64)


class Stats:
    """Counts how often the gold entity appears within the top-k
    predictions, for each k in RANKS (bounded by top_k)."""

    def __init__(self, top_k: int = 64):
        self.cnt = 0
        self.top_k = top_k
        self.ranks = [r for r in RANKS if r <= top_k]
        self.hits = [0] * len(self.ranks)

    def add(self, gold_position: int) -> None:
        """gold_position is the 0-based rank of the gold entity in the
        prediction list, or -1 if it is not in the top_k."""
        self.cnt += 1
        if gold_position == -1:
            return
        for i, rank in enumerate(self.ranks):
            if gold_position < rank:
                self.hits[i] += 1

    def extend(self, other: "Stats") -> None:
        self.cnt += other.cnt
        for i in range(len(self.ranks)):
            self.hits[i] += other.hits[i]

    def recalls(self) -> dict:
        if self.cnt == 0:
            return {f"recall@{r}": 0.0 for r in self.ranks}
        return {f"recall@{r}": self.hits[i] / self.cnt for i, r in enumerate(self.ranks)}

    def output(self) -> str:
        parts = [f"Total: {self.cnt} examples."]
        for name, value in self.recalls().items():
            parts.append(f"{name.replace('recall@', 'r@')}: {value:.4f}")
        return " ".join(parts)
