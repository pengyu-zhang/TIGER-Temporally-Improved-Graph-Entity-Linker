import json

def build_qid_index(input_file1, output_file1, input_file2, output_file2):
    # Step 1: Build the qid to index mapping.
    qid_dict = {}
    counter = 0
    with open(input_file1, 'r', encoding='utf-8') as f1:
        for line in f1:
            data = json.loads(line)
            qid = data.get("qid", "")
            if qid not in qid_dict:
                qid_dict[qid] = counter
                counter += 1

    with open(output_file1, 'w', encoding='utf-8') as f2:
        for qid, index in qid_dict.items():
            f2.write(f'{qid} {index}\n')

    # Step 2: Replace the qids in the input_file2 with their corresponding indices.
    with open(input_file2, 'r', encoding='utf-8') as f3, open(output_file2, 'w', encoding='utf-8') as f4:
        for line in f3:
            qid1, qid2 = line.strip().split('\t')
            index1 = qid_dict.get(qid1, 0)
            index2 = qid_dict.get(qid2, 0)
            f4.write(f'{index1} {index2}\n')


if __name__ == "__main__":
    directories = ["train"]  # , "validation", "test"]
    for directory in directories:
        for year in range(2013, 2023):
            input_file1 = f'D:/datasets/06_dataset_7_4/02_blink_format_with_index/all/{year}/{year}_{directory}.jsonl'
            output_file1 = f'D:/datasets/06_dataset_7_4/03_relationship/mention_relation/{year}_{directory}_qid_id_mapping.txt'
            input_file2 = f'D:/datasets/06_dataset_7_4/03_relationship/mention_relation/{year}_{directory}_qid_relation.txt'
            output_file2 = f'D:/datasets/06_dataset_7_4/03_relationship/mention_relation/{year}_{directory}_id_relation.txt'
            build_qid_index(input_file1, output_file1, input_file2, output_file2)
