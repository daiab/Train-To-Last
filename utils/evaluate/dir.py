"""
txt format:
id_1 feature_1 feature_2 ...... feature_n
id_2 feature_1 feature_2 ...... feature_n
"""
import numpy as np


def read_id_feature(file_name):
    data = np.genfromtxt(file_name, dtype=float, delimiter=' ')
    id, feature = np.split(data, [1], axis=1)
    return id.astype(int), feature


def cal(M_file, N_file):
    """
    :param M_file: feature matrix file
    :param N_file: feature matrix file
    :return:
    """
    t_sep = np.arange(0, 1, 0.1)

    # 读取特征矩阵
    M_id, M_feature = read_id_feature(M_file)
    N_id, N_feature = read_id_feature(N_file)
    assert M_feature.shape[0] == M_feature.shape[1] == 2, "M file format error"
    assert N_feature.shape[0] == N_feature.shape[1] == 2, "N file format error"

    # 计算特征相似度矩阵
    similar_mat = np.dot(M_feature, N_feature.T)

    rows, cols = similar_mat.shape

    # 计算M和N集合中相同的id（人）
    intersect_id = np.intersect1d(M_id, N_id)

    # 计算相同id的人所在的index
    M_intersect_idx = np.where(np.in1d(M_id, intersect_id))[0]
    N_intersect_idx = np.where(np.in1d(N_id, intersect_id))[0]

    # 将相似矩阵reshape成向量后的index
    flatten_idx = list(range(rows * cols))
    # 相同id的人的index
    flatten_intersect_idx = []
    for i in M_intersect_idx:
        for j in N_intersect_idx:
            flatten_intersect_idx.append(i * cols + j)
    # 相同id之外的人的index
    flatten_diff_idx = [i for i in flatten_idx if i not in flatten_intersect_idx]

    flatten_similar = similar_mat.reshape([-1])

    # 导出positive和negtive向量
    positive = flatten_similar[flatten_intersect_idx]
    negtive = flatten_similar[flatten_diff_idx]

    # 从大到小排序
    negtive_sort = np.sort(negtive)[::-1]

    negtive_sort_len = negtive_sort.shape[0]
    negtive_t_idx = (t_sep * negtive_sort_len).astype(int)

    right_count = []
    t_val = []
    for idx in negtive_t_idx:
        neg_val = negtive_sort[idx]
        t_val.append(neg_val)
        # neg_val对应的正确个数
        right_count.append(np.count_nonzero(np.where(positive > neg_val)))

    positive_count = positive.shape[0]
    right_rate = [count / positive_count for count in right_count]
    return t_val, right_rate


def main():
    M_file = '/home/daiab/test/M.txt'
    N_file = '/home/daiab/test/N.txt'
    t_step, right_rate = cal(M_file, N_file)
    print(right_rate)


if __name__ == "__main__":
    main()




