import torch
import torch.nn as nn
from FP_unfolding.compute_WSR import WSR
import math
def print_tensor_memory():
    """打印所有变量的状态，包括其大小和位置。"""
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensors.append((obj.size(), obj.device))
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensors.append((obj.data.size(), obj.data.device))
        except Exception as e:
            pass
    for idx, (size, device) in enumerate(tensors):
        print(f"Tensor {idx}: Size {size}, Device {device}")

class IAIDNNnet(nn.Module):
    def __init__(self,BS_number,user_number,N_r,N_t,w,d,iterations,device,signal_max_power=1e2,noise_power=1e-7):

        # super(FastFPnet).__init__()
        super().__init__()

        self.BS_number=BS_number
        self.user_number=user_number#用户数量
        self.N_r=N_r#接收天线数量
        self.N_t=N_t#发送天线数量
        self.weight=w#权重向量
        self.device=device
        self.d=d

        self.iterations=iterations#网络unfolding的迭代次数

        ## defining learnable parameters
        self.X=nn.Parameter(torch.randn(iterations,BS_number,user_number,N_t,N_t,dtype=torch.cdouble).to(device),requires_grad=True)
        self.Y=nn.Parameter(torch.randn(iterations,BS_number,user_number,N_t,N_t,dtype=torch.cdouble).to(device),requires_grad=True)
        self.Z=nn.Parameter(torch.randn(iterations,BS_number,user_number,N_t,N_t,dtype=torch.cdouble).to(device),requires_grad=True)

        self.O=nn.Parameter(torch.randn(iterations,BS_number,user_number,N_t,d,dtype=torch.cdouble).to(device),requires_grad=True)


        ## defining unearnable parameters
        self.noise_power=torch.tensor([noise_power]).to(device)
        self.signal_max_power=signal_max_power
    

    def interation_forward(self,H,V_last,iteration_idx,w=None,signal_max_power=None,noise_power=None,action_noise_power=0):

        device=H.device

        Z=V_last.clone()

        d=V_last.size(-1)#数据流数

        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        HV_matrics = torch.zeros(N_samples, BS_number, BS_number, user_number, user_number, N_r, d ,dtype=torch.cdouble).to(device)
        HV_matrics2 = torch.zeros(N_samples, BS_number, user_number, N_r, d ,dtype=torch.cdouble).to(device)
        for k in range(BS_number):
            for l in range(BS_number):
                for i in range(user_number):
                    for j in range(user_number):
                        HV_matrics[:, k, l, i, j, :, :] = H[:, k, l, i, :, :] @ V_last[:, k, j, :, :] #H_{k,li}W{k,j}
                        if k==l and j==i:
                            HV_matrics2[:, k, i,:,:]=HV_matrics[:, k, l, i, j, :, :]


        _temp_HV = torch.zeros(N_samples,  BS_number, BS_number, user_number, user_number, N_r, N_r,dtype=torch.cdouble).to(device)
        # 计算 T
        for k in range(BS_number):
            for l in range(BS_number):
                for i in range(user_number):
                    for j in range(user_number):
                        _temp_HV[:,k,l,i,j,:,:]=HV_matrics[:,k,l,i, j,:,:] @ HV_matrics[:,k,l,i,j,:,:].mH


        # #更新Y  Y即WMMSE中的U矩阵
        # U_u=torch.zeros(N_samples,BS_number,user_number).to(device)
        # for l in range(BS_number):
        #     for k in range(user_number):
        #         U_u[:,l, k]=(noise_power/signal_max_power)*(V_last[:,l,k,:,:]@V_last[:,l,k,:,:].mH).diagonal(dim1=-2, dim2=-1).sum(-1).double()

        # U_u=U_u.view(N_samples,BS_number,user_number,1,1)
        # U=U_u*torch.eye(N_r,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
        # # 更新U
        # for l in range(BS_number):
        #     for k in range(user_number):
        #         U[:, l, k, :, :] = U[:, l, k, :, :] + _temp_HV[:,:,l, k, :, :, :].sum(dim=1).sum(dim=1)
        # Y=torch.linalg.inv(U)@HV_matrics2

        #更新Y  Y即WMMSE中的U矩阵
        U=torch.zeros(N_r,N_r,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
        # 更新U
        beta_U=torch.zeros(N_samples,dtype=torch.cdouble).to(device)
        for l in range(BS_number):
            for k in range(user_number):
                beta_U+=(noise_power/signal_max_power)*(Z[:,l,k,:,:]@Z[:,l,k,:,:].mH).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                U[:, l, k, :, :] = U[:, l, k, :, :] + _temp_HV[:,:,l, k, :, :, :].sum(dim=1).sum(dim=1)
        U+=beta_U.view(N_samples,1,1,1,1)*torch.eye(N_r,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
        Y=torch.linalg.inv(U)@HV_matrics2


        #更新W
        W=torch.eye(d, dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
        for l in range(BS_number):
            for k in range(user_number):
                W[:, l, k, :, :] = W[:, l, k, :, :] - Y[:,l,k,:,:].mH @ HV_matrics2[:,l,k,:,:]
        W=torch.linalg.inv(W)


        # # 计算u
        # u=torch.zeros(N_samples,BS_number).to(device)
        # for l in range(BS_number):
        #     for k in range(user_number):
        #         u[:,l]+=(noise_power/signal_max_power)*w[k]*(Y[:,l,k,:,:]@W[:, l, k, :, :]@Y[:,l,k,:,:].mH).diagonal(dim1=-2, dim2=-1).sum(-1).double()


        # #B矩阵
        # 单小区
        _temp_Matrix=torch.zeros(N_samples,1,1,N_t,N_t,dtype=torch.cdouble).to(device)
        beta_u=torch.zeros(N_samples,dtype=torch.cdouble).to(device)
        for idx_BS in range(BS_number):
            for idx_user in range(user_number):
                beta_u+=(noise_power/signal_max_power)*(w[idx_user]*Y[:,idx_BS,idx_user,:,:]@W[:,idx_BS,idx_user,:,:]@Y[:,idx_BS,idx_user,:,:].mH).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                _temp_Matrix[:,0,0,:,:]+=w[idx_user]*H[:,0,idx_BS,idx_user,:,:].mH@Y[:,idx_BS,idx_user,:,:]@W[:,idx_BS,idx_user,:,:]@Y[:,idx_BS,idx_user,:].mH@H[:,0,idx_BS,idx_user,:,:]

        _temp_Matrix+=beta_u.view(N_samples,1,1,1,1)*torch.eye(N_t, dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples,1,1,1,1).to(device)

        _temp_Matrix=_temp_Matrix.repeat(1, 1, user_number, 1, 1)

        _temp_B_add=torch.diagonal(_temp_Matrix, dim1=-2, dim2=-1)
        _temp_B_add=torch.diag_embed(1.0/_temp_B_add)

        _temp_B_inv=_temp_B_add@self.X[iteration_idx, :, :, :, :].unsqueeze(0).repeat(N_samples, 1, 1, 1, 1)+_temp_Matrix@self.Y[iteration_idx, :, :, :, :].unsqueeze(0).repeat(N_samples, 1, 1, 1, 1)+self.Z[iteration_idx, :, :, :, :].unsqueeze(0).repeat(N_samples, 1, 1, 1, 1)


        # _temp_B_inv=torch.linalg.inv(_temp_Matrix)
        # _temp_B_inv=_temp_Matrix.repeat(1, 1, user_number, 1, 1)


        V=self.O[iteration_idx, :, :, :, :].unsqueeze(0).repeat(N_samples, 1, 1, 1, 1)
        for l in range(BS_number):
            for k in range(user_number):
                V[:,l, k, :, :] += w[k]*_temp_B_inv[:,l,k,:,:]@H[:,l,l,k,:,:].mH@Y[:,l,k,:,:]@W[:,l,k,:,:]
                # V[:,l, k, :, :] = w[k]*_temp_B_inv[:,l,k,:,:]@H[:,l,l,k,:,:].mH@Y[:,l,k,:,:]@W[:,l,k,:,:]


        # if iteration_idx == self.iterations-1:
        # 功率归一化
        frobenius_norm_squared = torch.linalg.norm(V, ord='fro', dim=(-2, -1))**2
        frobenius_norm_squared=frobenius_norm_squared.sum(dim=2) #所有用户的总功率
        #         mask = frobenius_norm_squared > signal_max_power
        # if mask.any():
            # 计算当前矩阵的 Frobenius 范数的平方
        current_norm_squared = frobenius_norm_squared
        # 计算缩放系数
        scale_factors = (signal_max_power / current_norm_squared).sqrt()

        V=V.clone()
        # # 对符合条件的矩阵进行缩放
        V *= scale_factors.view(-1,1,1,1,1)

        # # 找出 Frobenius 范数的平方大于 P 的矩阵的掩码
        # mask = frobenius_norm_squared > signal_max_power
        # if mask.any():
        #     # 计算当前矩阵的 Frobenius 范数的平方
        #     current_norm_squared = frobenius_norm_squared[mask]
        #     # 计算缩放系数
        #     scale_factors = (signal_max_power / current_norm_squared).sqrt()

        #     V=V.clone()
        #     # # 对符合条件的矩阵进行缩放
        #     V[mask] *= scale_factors.view(-1, 1, 1 , 1)

        return V
        
    
    def init_input(self,N_samples,input_shape=None,signal_max_power=None,H=None):
        if input_shape is None:
            BS_number,user_number,N_t,stream_num=self.BS_number,self.user_number,self.N_t,self.d
        else:
            N_samples,BS_number,user_number,N_t,stream_num=input_shape
        
        if signal_max_power is None:
            signal_max_power=self.signal_max_power


        V=torch.randn(N_samples,BS_number,user_number,N_t,stream_num,dtype=torch.cdouble)
        frobenius_norm_squared = torch.linalg.norm(V, ord='fro', dim=(-2, -1))**2
        frobenius_norm_squared=frobenius_norm_squared.sum(dim=2) #所有用户的总功率
        scale_factors = (signal_max_power / frobenius_norm_squared).sqrt()
        V *= scale_factors.view(N_samples,BS_number, 1, 1 , 1)


        # # 初始化 ZF 预编码矩阵 W，大小为 (N, 1, 1, U, T, R)
        # W = torch.zeros(N_samples, BS_number,  user_number, N_t, stream_num, dtype=torch.cdouble)

        # # 计算每个用户的 ZF 预编码矩阵
        # for n in range(N_samples):
        #     for u in range(user_number):
        #         # 提取第 n 个样本和第 u 个用户的信道矩阵 H_nu，大小为 (R, T)
        #         H_nu = H[n, 0, 0, u, :, :]
                
        #         # 对 H_nu 进行 SVD 分解
        #         U, S, Vh = torch.linalg.svd(H_nu)
                
        #         # 取 Vh 的前 d 列（右奇异向量的转置矩阵），大小为 (T, d)
        #         W_nu = Vh[:, :stream_num]
                
        #         # 将计算结果保存到 W 中
        #         W[n, 0, u, :, :] = W_nu

        return V

    def performe(self,H,w=None,iterations=None,signal_max_power=None,noise_power=None):
        device=H.device
        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        if not iterations: iterations=self.iterations
        if signal_max_power is None: 
            signal_max_power=self.signal_max_power
        if w is None:
            w=self.weight
        if noise_power is None:
            noise_power=self.noise_power

        V=self.init_input(N_samples,(N_samples,BS_number,user_number,N_t,self.d),signal_max_power,H).to(device)# 目前并不实现对于d的通用

        wsr_record=[0 for i in range(iterations+1)]
        wsr_record[0]=WSR(H,V,w,noise_power=noise_power)
        print(0)
        print(wsr_record[0])
        for i in range(iterations):

            V=self.interation_forward(H,V,min(i,self.iterations-1),w=w,signal_max_power=signal_max_power,noise_power=noise_power)

            wsr_record[i+1]=WSR(H,V,w,noise_power=noise_power)

            print(i+1)
            print(wsr_record[i+1])

        return wsr_record

    
    def forward(self,H,w=None,iterations=None,signal_max_power=None,noise_power=None,action_noise_power=0,init_V=None):
        device=H.device
        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        if not iterations: iterations=self.iterations
        if signal_max_power is None: 
            signal_max_power=self.signal_max_power
        if w is None:
            w=self.weight
        if noise_power is None:
            noise_power=self.noise_power

        if not torch.is_tensor(init_V):
            V=self.init_input(N_samples,(N_samples,BS_number,user_number,N_t,self.d),signal_max_power).to(device)
        else:
            V=init_V.to(device)

        V_out_record=[]
        for i in range(iterations):
            V=self.interation_forward(H,V,i,w=w,signal_max_power=signal_max_power,noise_power=noise_power,action_noise_power=action_noise_power)
            V_out_record.append(V)

        return V,V_out_record