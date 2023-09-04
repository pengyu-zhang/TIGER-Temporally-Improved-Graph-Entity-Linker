import numpy as np
import faiss
import dgl
import torch

def load_features_from_npy(file_path):
    """从npy文件中加载特征。"""
    features = np.load(file_path).astype(np.float32)  # 转换类型到float32
    print("Features shape:", features.shape)
    features = features.reshape(features.shape[0], -1)
    print("Features shape:", features.shape)
    return features


def create_knn_graph(features, k):
    """使用Faiss计算KNN图并返回。"""
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    D, I = index.search(features, k + 1)  # 搜索K+1个最近邻居，包括节点本身
    src = np.repeat(np.arange(features.shape[0]), k)  # 重复节点索引k次
    dst = I[:, 1:].reshape(-1)  # 移除自环并展平邻居索引
    return dgl.graph((src, dst), num_nodes=features.shape[0])


def save_knn_graph_to_txt(knn_graph, file_path):
    """保存KNN图到TXT文件。"""
    edge_list = knn_graph.edges()
    edge_list = torch.stack(edge_list).t().numpy()  # 将元组转化为张量，然后转置并转换为numpy数组
    np.savetxt(file_path, edge_list, fmt='%d')



if __name__ == "__main__":
    for year in range(2013, 2023):
        # 加载特征
        features_file = f'./07_knn_relation_small/{year}/embeddings_train.npy'
        features = load_features_from_npy(features_file)

        # 生成KNN图
        for k in range(1, 10):
            # k = 3  # 您可以根据需求更改K的值
            knn_graph_result = create_knn_graph(features, k)

            # 保存KNN图到TXT文件
            output_file_path = f'./07_knn_relation_small/{year}/{year}_knn_graph_{k}.txt'
            save_knn_graph_to_txt(knn_graph_result, output_file_path)
