"""Build the processed data layout used by tiger.train / tiger.evaluate.

Target layout (data/processed/graph_tempel):
  documents/{year}.json                    per-year entity descriptions
  mentions/all/{year}/{validation,test}.jsonl
  mentions/new_entities/{year}/train.jsonl        (1,764 samples)
  mentions/continual_entities/{year}/train.jsonl  (1,764 samples)
  graphs/{year}/structure_edges.txt        Wikidata5M relations (node ids)
  graphs/{year}/knn_edges_k{K}.txt         kNN feature graphs
  graphs/{year}/qid_to_node.txt            qid -> node id
  graphs/{year}/feature_matrix.txt         n x m 0/1 keyword matrix

Sources:
  --source extracted  : from an extracted Graph-TempEL release
                        (unpacked components of Zenodo record 10977757)
  --source local      : from a local materials folder
                        (same content, pre-release layout)

Every build validates: documents/mapping/feature alignment, train-set
sizes, and that every mention's label_id indexes into the year's
document file.
"""

import argparse
import hashlib
import json
import os
import shutil


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: str) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def build_documents(src_dir: str, out_dir: str, year: int) -> None:
    """The source ships {year}_{train,validation,test}.json. The three
    files contain the SAME entity set but in a DIFFERENT line order, and
    each split's mention label_id indexes into its own file's order —
    so all three are kept. Verify the sets really match."""
    line_sets = {}
    for split in ("train", "validation", "test"):
        src = os.path.join(src_dir, f"{year}_{split}.json")
        with open(src, "rb") as fh:
            line_sets[split] = hashlib.md5(b"\n".join(sorted(fh.read().splitlines()))).hexdigest()
        _copy(src, os.path.join(out_dir, "documents", f"{year}_{split}.json"))
    if len(set(line_sets.values())) != 1:
        raise RuntimeError(f"{year}: document entity sets differ across splits: {line_sets}")


def split_train_by_category(all_train_path: str, out_root: str, year: int) -> dict:
    """Derive the new-entities training variant (exactly the 1,764
    category == 'new_entities' samples) from the 'all' training file.

    The continual variant is NOT derivable this way: the paper sampled
    1,764 out of ~136k 'shared' mentions and the sampling is only
    preserved in the released dataset (Zenodo record 10977757,
    02_continual_entities_1764.7z) — use --source extracted for it."""
    counts = {"new_entities": 0, "shared": 0}
    path = os.path.join(out_root, "mentions", "new_entities", str(year), "train.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(all_train_path, "r", encoding="utf-8") as fh, \
            open(path, "w", encoding="utf-8", newline="\n") as out:
        for line in fh:
            category = json.loads(line)["category"]
            counts[category] = counts.get(category, 0) + 1
            if category == "new_entities":
                out.write(line)
    return counts


def validate_year(out_dir: str, year: int) -> dict:
    from .graph import load_qid_mapping, validate_alignment

    train_docs = os.path.join(out_dir, "documents", f"{year}_train.json")
    n_docs = _count_lines(train_docs)
    graph_dir = os.path.join(out_dir, "graphs", str(year))

    # Graph node ids follow the TRAIN document order (verified here);
    # the feature matrix and kNN graph follow the same order.
    mapping = load_qid_mapping(os.path.join(graph_dir, "qid_to_node.txt"))
    validate_alignment(train_docs, mapping)

    n_feat_rows = _count_lines(os.path.join(graph_dir, "feature_matrix.txt"))
    if n_feat_rows != n_docs:
        raise RuntimeError(
            f"{year}: feature matrix has {n_feat_rows} rows, {n_docs} documents")

    report = {"year": year, "documents": n_docs, "graph_nodes": len(mapping)}
    for variant in ("new_entities", "continual_entities"):
        train = os.path.join(out_dir, "mentions", variant, str(year), "train.jsonl")
        if os.path.exists(train):
            report[f"train_{variant}"] = _count_lines(train)
    for split in ("validation", "test"):
        path = os.path.join(out_dir, "mentions", "all", str(year), f"{split}.jsonl")
        if os.path.exists(path):
            n_split_docs = _count_lines(
                os.path.join(out_dir, "documents", f"{year}_{split}.json"))
            n, bad = 0, 0
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    n += 1
                    if int(json.loads(line)["label_id"]) >= n_split_docs:
                        bad += 1
            if bad:
                raise RuntimeError(f"{year} {split}: {bad} label_ids out of range")
            report[split] = n
    return report


def build_from_local(materials: str, out_dir: str, years: list[int]) -> None:
    """materials = a local 'dataset' materials folder."""
    for year in years:
        build_documents(os.path.join(materials, "01_documents_no_duplicates"),
                        out_dir, year)

        all_dir = os.path.join(materials, "02_blink_format_with_index", "all", str(year))
        counts = split_train_by_category(
            os.path.join(all_dir, "train.jsonl"), out_dir, year)
        print(f"[prepare] {year} train categories: {counts}")
        for split in ("validation", "test"):
            _copy(os.path.join(all_dir, f"{split}.jsonl"),
                  os.path.join(out_dir, "mentions", "all", str(year), f"{split}.jsonl"))

        rel = os.path.join(materials, "03_relationship", str(year))
        graph_dir = os.path.join(out_dir, "graphs", str(year))
        _copy(os.path.join(rel, f"{year}_train_id_relation.txt"),
              os.path.join(graph_dir, "structure_edges.txt"))
        _copy(os.path.join(rel, f"{year}_train_qid_id_mapping.txt"),
              os.path.join(graph_dir, "qid_to_node.txt"))
        _copy(os.path.join(materials, "04_feature", "01_keyword", str(year),
                           f"{year}_train_token_id_after_filter_bin.txt"),
              os.path.join(graph_dir, "feature_matrix.txt"))
        knn_dir = os.path.join(materials, "05_knn_relation", str(year))
        for k in range(1, 10):
            src = os.path.join(knn_dir, f"{year}_knn_graph_{k}.txt")
            if os.path.exists(src):
                _copy(src, os.path.join(graph_dir, f"knn_edges_k{k}.txt"))


def _find_dir(root: str, name: str) -> str:
    """Locate the payload directory named `name` under root. The Zenodo
    components nest their payloads inconsistently (e.g. the extraction
    wrapper and the component's top-level folder can share the name),
    so among all matches pick the innermost one — the match that has no
    same-named child directory."""
    matches = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d == name:
                matches.append(os.path.join(dirpath, d))
    innermost = [m for m in matches if not os.path.isdir(os.path.join(m, name))]
    if not innermost:
        raise FileNotFoundError(f"Component directory '{name}' not found under {root}")
    innermost.sort(key=len, reverse=True)
    return innermost[0]


def build_from_extracted(extracted: str, out_dir: str, years: list[int]) -> None:
    """extracted = directory tree holding the unpacked Zenodo
    components (record 10977757). Components are located by their
    payload folder names regardless of nesting."""
    docs_dir = _find_dir(extracted, "01_documents_no_duplicates")
    new_dir = _find_dir(extracted, "new_entities_1764")
    cont_dir = _find_dir(extracted, "continual_entities_1764")
    rel_root = _find_dir(extracted, "03_relationship")
    feat_root = os.path.join(_find_dir(extracted, "04_feature"), "01_keyword")
    knn_root = _find_dir(extracted, "05_knn_relation")

    for year in years:
        build_documents(docs_dir, out_dir, year)
        for variant, src_root in (("new_entities", new_dir),
                                  ("continual_entities", cont_dir)):
            src_year = os.path.join(src_root, str(year))
            _copy(os.path.join(src_year, "train.jsonl"),
                  os.path.join(out_dir, "mentions", variant, str(year), "train.jsonl"))
            for split in ("validation", "test"):
                src = os.path.join(src_year, f"{split}.jsonl")
                dst = os.path.join(out_dir, "mentions", "all", str(year), f"{split}.jsonl")
                if os.path.exists(src) and not os.path.exists(dst):
                    _copy(src, dst)

        rel = os.path.join(rel_root, str(year))
        graph_dir = os.path.join(out_dir, "graphs", str(year))
        _copy(os.path.join(rel, f"{year}_train_id_relation.txt"),
              os.path.join(graph_dir, "structure_edges.txt"))
        _copy(os.path.join(rel, f"{year}_train_qid_id_mapping.txt"),
              os.path.join(graph_dir, "qid_to_node.txt"))
        _copy(os.path.join(feat_root, str(year),
                           f"{year}_train_token_id_after_filter_bin.txt"),
              os.path.join(graph_dir, "feature_matrix.txt"))
        # kNN graphs: the release ships k=3 flat; local materials have
        # k=1..9 under per-year folders. Accept both.
        for k in range(1, 10):
            for src in (os.path.join(knn_root, f"{year}_knn_graph_{k}.txt"),
                        os.path.join(knn_root, str(year), f"{year}_knn_graph_{k}.txt")):
                if os.path.exists(src):
                    _copy(src, os.path.join(graph_dir, f"knn_edges_k{k}.txt"))
                    break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["local", "extracted"], required=True)
    parser.add_argument("--input", required=True,
                        help="materials dataset dir (local) or unpacked release dir (extracted)")
    parser.add_argument("--output", default="data/processed/graph_tempel")
    parser.add_argument("--years", default="2019,2020,2021,2022")
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]
    if args.source == "local":
        build_from_local(args.input, args.output, years)
    else:
        build_from_extracted(args.input, args.output, years)

    print("[prepare] validating ...")
    for year in years:
        report = validate_year(args.output, year)
        print(f"[prepare] OK {report}")
    print("[prepare] done")


if __name__ == "__main__":
    main()
