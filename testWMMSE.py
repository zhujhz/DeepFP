import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from FP_unfolding.generate_data import split_matrix
from FP_unfolding.utils import dataset
from FP_unfolding.FastFP_net import FastFPnet
from FP_unfolding.WMMSE import WMMSE
from FP_unfolding.FastFP_alg import FastFP
from FP_unfolding.loss_fuction import LossUnfolding
from FP_unfolding.compute_WSR import WSR
from FP_unfolding.load_chn import load_H

# 定义系统参数
N_t=32
N_r=4
d=2
user_number=6
BS_number=3
w=[i+1 for i in range(user_number)] 
w=[w[i]/sum(w) for i in range(user_number)] #创建等差权重矩阵
signal_max_power=1e2
noise_power=1e-8
N_samples=1000
net_iterations = 10


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using: ",device)


Data_blocks=N_samples//1000
H=[]
for i in range(Data_blocks):
    H.append(load_H(f"Data/{i+1}_1000_Nr_4_Nt_32_N_user_6_BS_{BS_number}.mat"))
H=torch.cat(H,dim=0)
# 分割数据
train_set, val_set, test_set = split_matrix(H=H,train_size=0.7)


train_set=test_set[:10000].to(device)


wmmse=WMMSE()
Alg_wsr_wmmse=wmmse.performe(train_set[:1],w,10,d=2,signal_max_power=signal_max_power,noise_power=noise_power)