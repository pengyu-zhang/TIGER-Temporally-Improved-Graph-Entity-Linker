import json

def filter_by_qid(input_file1: str, input_file2: str, output_file: str, min_degree: int):
    qid_set = set()

    with open(input_file2, 'r') as file:
        for line in file:
            qid, degree = line.strip().split(' ')
            if int(degree) >= min_degree:
                qid_set.add(qid)

    with open(input_file1, 'r', encoding='utf-8') as file1, open(output_file, 'w', encoding='utf-8') as file2:
        for line in file1:
            data = json.loads(line)
            if data['qid'] in qid_set:
                file2.write(json.dumps(data, ensure_ascii=False)+'\n')

if __name__ == "__main__":
    min_degree = 15
    for year in range(2013, 2023):
        input_file1 = f"D:/datasets/06_dataset_7_4/02_blink_format_with_index/all/{year}/{year}_train.jsonl"
        input_file2 = f"D:/datasets/06_dataset_7_4/06_degree/all/{year}/{year}_train_qid_degree.txt"
        output_file = f"D:/datasets/06_dataset_7_4/06_degree/all/{year}/{year}_train_filtered_{min_degree}.jsonl"
        filter_by_qid(input_file1, input_file2, output_file, min_degree)
