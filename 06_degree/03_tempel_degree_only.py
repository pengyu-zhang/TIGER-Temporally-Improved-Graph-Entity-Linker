import json
from collections import defaultdict

def filter_jsonl_by_degree(file1, file2, file3, top_k=100):
    degree_dict = {}

    with open(file1, 'r', encoding='utf-8') as f1:
        for line in f1.readlines():
            qid, degree = line.strip().split('\t')
            degree_dict[qid] = int(degree)

    # 按度值对 QID 进行排序，并取前 k 个
    sorted_qids = sorted(degree_dict, key=lambda x: degree_dict[x], reverse=True)[:top_k]

    file2_data = defaultdict(list)

    with open(file2, 'r', encoding='utf-8') as f2:
        for line in f2.readlines():
            json_line = json.loads(line)
            target_qid = json_line.get("qid", "")
            if target_qid in sorted_qids:
                file2_data[target_qid].append(json_line)

    with open(file3, 'w', encoding='utf-8') as f3:
        for qid in sorted_qids:
            if qid in file2_data:
                for json_line in file2_data[qid]:
                    f3.write(json.dumps(json_line, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    # file1 = 'C:/code/dataprocess/datasets_4_12/02_tempel_2_blink/graph/01_relation/2014_train_qid_degree.txt'
    # file2 = 'C:/code/dataprocess/datasets_4_12/04_tempel_without_adj/blink_format/2014/train.jsonl'
    # file3 = 'C:/code/dataprocess/datasets_4_12/04_tempel_without_adj/blink_format/2014/2014_train_top_10000_qid.jsonl'
    for year in range(2013, 2023):
        file1 = f'D:/datasets_4_12/02_tempel_2_blink/graph/01_relation/{year}_train_qid_degree.txt'
        file2 = f'D:/datasets/06_dataset_7_4/02_blink_format_with_index/all/{year}/train.jsonl'
        file3 = f'D:/datasets/06_dataset_7_4/02_blink_format_with_index/top_100/{year}/train.jsonl'
        filter_jsonl_by_degree(file1, file2, file3)
