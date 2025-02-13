import torch
import torch.nn as nn

def generate_matrix(N_samples,user_number,N_r,N_t,train_size=0.7):

    real_part = torch.randn(N_samples,user_number, N_r, N_t,dtype=torch.float64)
    imag_part = torch.randn(N_samples,user_number, N_r, N_t,dtype=torch.float64)
    complex_matrix = torch.complex(real_part, imag_part)

    # 划分数据集比例
    train_ratio = train_size
    val_ratio = (1-train_size)/2
    test_ratio = 1-train_ratio-val_ratio

    # 计算各个集合的样本数量
    N_train = int(train_ratio * N_samples)
    N_val = int(val_ratio * N_samples)
    N_test = N_samples - N_train - N_val  # 防止浮点数不精确

    # 随机打乱数据索引
    indices = torch.randperm(N_samples)

    # 划分数据集
    train_indices = indices[:N_train]
    val_indices = indices[N_train:N_train + N_val]
    test_indices = indices[N_train + N_val:]

    train_set = complex_matrix[train_indices]
    val_set = complex_matrix[val_indices]
    test_set = complex_matrix[test_indices]

    return train_set, val_set, test_set

def split_matrix(H,train_size=0.7):

    N_samples=H.size(0)

    # 划分数据集比例
    train_ratio = train_size
    val_ratio = (1-train_size)/2
    test_ratio = 1-train_ratio-val_ratio

    # 计算各个集合的样本数量
    N_train = int(train_ratio * N_samples)
    N_val = int(val_ratio * N_samples)
    N_test = N_samples - N_train - N_val  # 防止浮点数不精确

    # 随机打乱数据索引
    indices = torch.randperm(N_samples)
    # indices =torch.arange(N_samples)

    # 划分数据集
    train_indices = indices[:N_train]
    val_indices = indices[N_train:N_train + N_val]
    test_indices = indices[N_train + N_val:]

    train_set = H[train_indices]
    val_set = H[val_indices]
    test_set = H[test_indices]

    return train_set, val_set, test_set