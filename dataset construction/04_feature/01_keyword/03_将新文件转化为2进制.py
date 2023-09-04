# def binary_representation(input_file, output_file):
#     # 读取需要转换的文件
#     with open(input_file, 'r', encoding='utf-8') as file3:
#         lines = file3.readlines()
#
#     # 创建一个集合来存储所有的token IDs
#     all_tokens = set()
#
#     # 创建一个列表来存储每一行的token IDs
#     lines_tokens = []
#
#     # 处理每一行
#     for line in lines:
#         # 分割行数据为token IDs
#         token_ids = line.strip().split()
#         # 将token IDs添加到列表中
#         lines_tokens.append(token_ids)
#         # 将token IDs添加到集合中
#         all_tokens.update(token_ids)
#
#     # 将集合转换为列表，并排序
#     all_tokens = sorted(list(all_tokens))
#
#     # 创建一个列表来存储每一行的二进制表示
#     binary_lines = []
#
#     # 处理每一行
#     for token_ids in lines_tokens:
#         # 创建一个列表来存储这一行的二进制表示
#         binary_line = []
#         # 检查每一个唯一的token ID是否在这一行中
#         for token in all_tokens:
#             if token in token_ids:
#                 binary_line.append('1')
#             else:
#                 binary_line.append('0')
#         # 将这一行的二进制表示添加到列表中
#         binary_lines.append(binary_line)
#
#     # 将二进制表示写入到TXT文件中
#     with open(output_file, 'w', encoding='utf-8') as file4:
#         for binary_line in binary_lines:
#             file4.write(' '.join(binary_line) + '\n')
#
# # 调用函数
# binary_representation('C:/code/dataprocess/datasets_4_12/10_text_2_token/10000_token_id_after_filter.txt',
#                       'C:/code/dataprocess/datasets_4_12/10_text_2_token/10000_token_id_after_filter_bin.txt')


def binary_representation_and_index(input_file, output_file, index_file):
    # 读取需要转换的文件
    with open(input_file, 'r', encoding='utf-8') as file3:
        lines = file3.readlines()

    # 创建一个集合来存储所有的token IDs
    all_tokens = set()

    # 创建一个列表来存储每一行的token IDs
    lines_tokens = []

    # 处理每一行
    for line in lines:
        # 分割行数据为token IDs
        token_ids = line.strip().split()
        # 将token IDs添加到列表中
        lines_tokens.append(token_ids)
        # 将token IDs添加到集合中
        all_tokens.update(token_ids)

    # 将集合转换为列表，并排序
    all_tokens = sorted(list(all_tokens))

    # 创建一个列表来存储每一行的二进制表示
    binary_lines = []

    # 创建一个列表来存储每一行的token ID及其在二进制表示中的索引
    token_indices = []

    # 处理每一行
    for token_ids in lines_tokens:
        # 创建一个列表来存储这一行的二进制表示
        binary_line = []
        # 创建一个列表来存储这一行的token ID及其在二进制表示中的索引
        token_index = []
        # 检查每一个唯一的token ID是否在这一行中
        for index, token in enumerate(all_tokens):
            if token in token_ids:
                binary_line.append('1')
                token_index.append((token, index))
            else:
                binary_line.append('0')
        # 将这一行的二进制表示添加到列表中
        binary_lines.append(binary_line)
        # 将这一行的token ID及其在二进制表示中的索引添加到列表中
        token_indices.append(token_index)

    # 将二进制表示写入到TXT文件中
    with open(output_file, 'w', encoding='utf-8') as file4:
        for binary_line in binary_lines:
            file4.write(' '.join(binary_line) + '\n')

    # 写入索引文件
    with open(index_file, 'w', encoding='utf-8') as file5:
        for indices in token_indices:
            for token, index in indices:
                file5.write(f'{token} {index}\n')


for year in range(2013, 2023):
    # 调用函数
    binary_representation_and_index(f'D:/datasets/06_dataset_7_4/06_feature/01_keyword/{year}/{year}_train_token_id_after_filter.txt',
                                    f'D:/datasets/06_dataset_7_4/06_feature/01_keyword/{year}/{year}_train_token_id_after_filter_bin.txt',
                                    f'D:/datasets/06_dataset_7_4/06_feature/01_keyword/{year}/{year}_index_file.txt')
