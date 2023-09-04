import os

# 定义基本路径
base_path = "D:/datasets/06_dataset_7_4/06_degree/feature_graph_degree/"

# 循环创建文件夹
for year in range(2013, 2023):
    dir_path = os.path.join(base_path, str(year))
    os.makedirs(dir_path, exist_ok=True)
