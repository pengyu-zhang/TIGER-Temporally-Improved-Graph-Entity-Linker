#!/bin/bash

PYTHONPATH=. python blink/biencoder/train_biencoder.py \
  --data_path data/01_blink_baseline/blink_format/random_sample/2013/ \
  --output_path /gpfs/scratch1/nodespecific/int5/pzhang/models/01_blink_baseline_2013_train_random_test_new/biencoder \
  --learning_rate 1e-05 --num_train_epochs 1 --max_context_length 128 --max_cand_length 128 \
  --train_batch_size 64 --eval_batch_size 16 --bert_model google/bert_uncased_L-8_H-512_A-8 \
  --type_optimization all_encoder_layers --data_parallel \
  --structgraph_path data/02_graph/2013/2013_train_id_relation.txt \
  --structgraph_size 10373 \
  --featuregraph_path data/02_graph/2013/2013_knn_graph_3.txt \
  --featuregraph_size 10373 \
  --feature_path data/02_graph/2013/2013_train_token_id_after_filter_bin.txt \
  --nfeat 2889 \
  --nhid1 768 \
  --nhid2 256 \
  --beta 0 \
  --theta 0.5 \
  --dropout 0.5 \

# Get top-64 predictions from Biencoder model on train, valid and test dataset
PYTHONPATH=. python blink/biencoder/eval_biencoder.py \
  --path_to_model /gpfs/scratch1/nodespecific/int5/pzhang/models/01_blink_baseline_2013_train_random_test_new/biencoder/pytorch_model.bin \
  --data_path data/01_blink_baseline/blink_format/new_entities/ \
  --output_path /gpfs/scratch1/nodespecific/int5/pzhang/models/01_blink_baseline_2013_train_random_test_new/ \
  --encode_batch_size 8 --eval_batch_size 2 --top_k 64 --save_topk_result \
  --bert_model google/bert_uncased_L-8_H-512_A-8 --mode 2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 \
  --zeshel True --data_parallel \
