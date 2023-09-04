import json
import time

def read_qid_pairs(file1):
    with open(file1, 'r', encoding='utf-8') as f1:
        for line in f1:
            qid1, r, qid2 = line.strip().split('\t')
            yield qid1, r, qid2

def collect_target_qids(file2):
    target_qids_set = set()
    with open(file2, 'r', encoding='utf-8') as f2:
        for line in f2:
            json_line = json.loads(line)
            target_qid = json_line.get("qid", "")
            target_qids_set.add(target_qid)
    return target_qids_set

def write_matching_lines(qid_pairs, target_qids_set, file3, file4):
    with open(file3, 'w', encoding='utf-8') as f3, open(file4, 'w', encoding='utf-8') as f4:
        for qid1, r, qid2 in qid_pairs:
            if qid1 in target_qids_set and qid2 in target_qids_set:
                f3.write(f'{qid1}\t{r}\t{qid2}\n')
                f4.write(f'{qid1}\t{qid2}\n')

def extract_matching_lines(file1, file2, file3, file4):
    qid_pairs = read_qid_pairs(file1)
    target_qids_set = collect_target_qids(file2)
    write_matching_lines(qid_pairs, target_qids_set, file3, file4)

if __name__ == '__main__':
    file1 = 'D:/datasets/06_dataset_7_4/wikidata5m/wikidata5m_all_triplet/wikidata5m_all_triplet.txt'
    directories = ["train"]  # , "validation", "test"]
    for directory in directories:
        for year in range(2013, 2023):
            file2 = f"D:/datasets/06_dataset_7_4/02_blink_format_with_index/all/{year}/{year}_{directory}.jsonl"
            file3 = f'D:/datasets/06_dataset_7_4/03_relationship/mention_relation/{year}_{directory}_qid_triplet.txt'
            file4 = f'D:/datasets/06_dataset_7_4/03_relationship/mention_relation/{year}_{directory}_qid_relation.txt'
            extract_matching_lines(file1, file2, file3, file4)
