import json


def calculate_total_and_average_count(file1_path, file2_path):
    qid_count_dict = {}
    with open(file1_path, 'r', encoding='utf-8') as file1:
        for line in file1:
            qid, count = line.strip().split()
            qid_count_dict[qid] = int(count)

    total_count = 0
    sample_count = 0
    with open(file2_path, 'r', encoding='utf-8') as file2:
        for line in file2:
            data = json.loads(line)
            qid = data["qid"]
            total_count += qid_count_dict.get(qid, 0)
            sample_count += 1

    average_count = total_count / sample_count if sample_count else 0

    return total_count, average_count


def calculate_total_and_average_count_based_on_line(file1_path, file2_path):
    line_count_dict = {}
    with open(file1_path, 'r', encoding='utf-8') as file1:
        for line in file1:
            line_num, count = line.strip().split()
            line_count_dict[int(line_num)] = int(count)

    total_count = 0
    sample_count = 0
    with open(file2_path, 'r', encoding='utf-8') as file2:
        for index, line in enumerate(file2):
            total_count += line_count_dict.get(index, 0)
            sample_count += 1

    average_count = total_count / sample_count if sample_count else 0

    return total_count, average_count

for year in range(2013, 2023):
    total, average = calculate_total_and_average_count(
        f"D:/datasets/06_dataset_7_4/06_degree/new_entities/{year}/{year}_train_qid_degree.txt",
        f"D:/datasets/06_dataset_7_4/02_blink_format_with_index/all/{year}/train.jsonl"
    )
    # total, average = calculate_total_and_average_count_based_on_line(
    #     f"D:/datasets/06_dataset_7_4/06_degree/feature_graph_degree/{year}/{year}_train_qid_degree.txt",
    #     f"D:/datasets/06_dataset_7_4/02_blink_format_with_index/all/{year}/train.jsonl"
    # )
    print(f"Year: {year}, Total Count: {total}, Average Count: {average:.2f}")
