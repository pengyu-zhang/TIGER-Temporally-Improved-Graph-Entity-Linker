from collections import Counter

def count_token_frequency(input_file, output_file):
    # 读取TXT文件
    with open(input_file, 'r', encoding='utf-8') as file1:
        lines = file1.readlines()

    # 准备一个Counter来统计token ID的出现频次
    counter = Counter()

    # 循环处理每一行
    for line in lines:
        # 将行数据分割为token IDs
        token_ids = line.strip().split()
        # 更新Counter
        counter.update(token_ids)

    # 将Counter转换为一个列表，每个元素都是一个元组，元组的第一个元素是token ID，第二个元素是频次
    token_frequency = list(counter.items())

    # 按照频次从高到低排序
    token_frequency.sort(key=lambda x: x[1], reverse=True)

    # 将结果写入到TXT文件中
    with open(output_file, 'w', encoding='utf-8') as file2:
        for token_id, frequency in token_frequency:
            file2.write(f'{token_id} {frequency}\n')


for year in range(2013, 2023):
    # 调用函数
    count_token_frequency(f'D:/datasets/06_dataset_7_4/07_knn_relation/{year}/{year}_train_token_id.txt',
                          f'D:/datasets/06_dataset_7_4/06_feature/01_keyword/{year}/{year}_train_token_id_sorted.txt')
