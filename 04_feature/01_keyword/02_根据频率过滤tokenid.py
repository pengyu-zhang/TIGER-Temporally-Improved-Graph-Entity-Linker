def filter_tokens_by_frequency(input_file1, input_file2, output_file, frequency_ranges):
    # 读取包含token ID频次的文件
    with open(input_file1, 'r', encoding='utf-8') as file1:
        lines = file1.readlines()

    # 创建一个字典来存储token ID的频次
    token_frequency = {}

    # 处理每一行
    for line in lines:
        # 分割行数据为token ID和频次
        token_id, frequency = line.strip().split()
        frequency = int(frequency)
        # 如果频次在任何一个给定的范围内，将其添加到字典中
        if any(min_freq <= frequency <= max_freq for min_freq, max_freq in frequency_ranges):
            token_frequency[token_id] = frequency

    # 读取需要过滤的文件
    with open(input_file2, 'r', encoding='utf-8') as file2:
        lines = file2.readlines()

    # 创建一个列表来存储过滤后的token IDs
    filtered_tokens = []

    # 处理每一行
    for line in lines:
        # 分割行数据为token IDs
        token_ids = line.strip().split()
        # 过滤出满足频次范围要求的token IDs
        filtered = [token_id for token_id in token_ids if token_id in token_frequency]
        # 将过滤后的token IDs添加到列表中
        filtered_tokens.append(filtered)

    # 将过滤后的token IDs写入到TXT文件中
    with open(output_file, 'w', encoding='utf-8') as file3:
        for tokens in filtered_tokens:
            file3.write(' '.join(tokens) + '\n')


for year in range(2013, 2023):
    # 调用函数
    filter_tokens_by_frequency(f'D:/datasets/06_dataset_7_4/06_feature/01_keyword/{year}/{year}_train_token_id_sorted.txt',
                               f'D:/datasets/06_dataset_7_4/07_knn_relation/{year}/{year}_train_token_id.txt',
                               f'D:/datasets/06_dataset_7_4/06_feature/01_keyword/{year}/{year}_train_token_id_after_filter.txt',
                               [(46, 2000)])
