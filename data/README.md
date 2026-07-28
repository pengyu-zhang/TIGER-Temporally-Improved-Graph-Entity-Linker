# Data

All datasets stay out of git. `scripts/prepare_data.sh` downloads and
builds everything automatically; this page documents what it does and
how to do it manually.

## Graph-TempEL (main dataset)

Graph-TempEL combines [TempEL](https://cloud.ilabt.imec.be/index.php/s/RinXy8NgqdW58RW)
(yearly Wikipedia/Wikidata snapshots for temporal entity linking) with
entity relationships from
[Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m).
It is published on Zenodo under CC-BY-4.0:

- **Official source**: [zenodo.org/records/10977757](https://zenodo.org/records/10977757)
  (component files, ~911 MB total; MD5 list in `md5sums.txt`)
- **Mirror**: the same files are attached to this repository's GitHub
  Release as a fallback if Zenodo is unreachable.

`scripts/prepare_data.sh` tries Zenodo first, falls back to the mirror,
verifies MD5s, extracts, and builds this layout:

```
data/processed/graph_tempel/
├── documents/{year}_{train,validation,test}.json   entity descriptions
│     (same 10,373 entities per year; line ORDER differs per split and
│      each split's mention label_id indexes its own file's order)
├── mentions/
│   ├── all/{year}/{validation,test}.jsonl          43,327 / 48,215 mentions
│   ├── new_entities/{year}/train.jsonl             1,764 samples
│   └── continual_entities/{year}/train.jsonl       1,764 samples
└── graphs/{year}/
    ├── structure_edges.txt      Wikidata5M relations between the year's entities
    ├── knn_edges_k{1..9}.txt    kNN graphs over entity-description embeddings
    ├── qid_to_node.txt          qid -> node id (follows the TRAIN document order)
    └── feature_matrix.txt       10,373 x ~2,900 binary keyword matrix
```

Default years are 2019–2022 (the paper's main experimental window,
chosen there to avoid temporal leakage from Wikidata5M's July 2019
cutoff). Pass `YEARS=2013,...,2022` for the full range.

All fields are pre-tokenized WordPiece tokens (bert-base-uncased
vocabulary), so training does not need to run a tokenizer over text.

## Other datasets from the paper

The paper also reports non-temporal results on:

- **ZESHEL**: download via the
  [BLINK repository](https://github.com/facebookresearch/BLINK/tree/main/examples/zeshel)
  (`examples/zeshel/get_zeshel_data.sh`). Raw text format; set
  `data.pretokenized: false` in the config.
- **WikiLinksNED (Unseen-Mentions)**: Google Drive link in the
  [ET4EL repository](https://github.com/yasumasaonoe/ET4EL).

Neither is required for the Graph-TempEL experiments.
