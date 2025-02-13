import torch
import torch.nn as nn

from FP_unfolding.compute_WSR import WSR
from FP_unfolding.FastFP_alg_n import FastFP


class ComplexMSELoss(nn.Module):
    def __init__(self):
        super(ComplexMSELoss, self).__init__()

    def forward(self, input, target):
        # 计算实部和虚部的均方误差
        real_input = input.real
        imag_input = input.imag
        real_target = target.real
        imag_target = target.imag
        
        mse_real = torch.mean((real_input - real_target) ** 2)
        mse_imag = torch.mean((imag_input - imag_target) ** 2)
        
        # 总均方误差
        return mse_real + mse_imag


class LossUnfolding(nn.Module):
    def __init__(self,signal_max_power=1e2,noise_power=1e-8):
        super(LossUnfolding, self).__init__()
        self.signal_max_power=signal_max_power
        self.noise_power = noise_power
        self.mse=ComplexMSELoss()
        self.fastFP=FastFP()
    

    def forward(self, H, V, w, V_init=None, mode='unsuper',target_iters=20):
        # 自定义损失的计算
        if mode=="unsuper":
            return -WSR(H,V,w,noise_power=self.noise_power,keep_tensor=True)
        
        ##mode=="super"
        V_opt=self.fastFP(H,w,target_iters,d=V.size(-1),signal_max_power=self.signal_max_power,noise_power=self.noise_power,init_V=V_init)

        return self.mse(V,V_opt)