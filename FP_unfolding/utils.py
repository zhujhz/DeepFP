import torch
import torch.nn as nn

class dataset(torch.utils.data.Dataset):

    def __init__(self, matrix_list):
        super().__init__()
        self.M = matrix_list
        self.data_size = self.M.size(0)

    def __len__(self):
        return self.data_size

    def __getitem__(self,idx):
        Matrix = self.M[idx,:,:,:]
        
        return Matrix
    

# 定义一个学习率更新函数


