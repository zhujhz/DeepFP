import torch
import torch.nn as nn
import torch.utils.data
import math

from FP_unfolding.compute_WSR import WSR


class FastFP(nn.Module):
    def single_iteration_single_cell(self,H,V_last,w,signal_max_power=1e2,noise_power=1e-7):
        device=H.device

        Z=V_last.clone()

        N_samples,user_number,N_r,N_t=H.size()

        HV_matrics = torch.zeros(N_samples, user_number, user_number, N_r, 1,dtype=torch.cdouble).to(device)
        HV_matrics2=torch.zeros(N_samples, user_number, N_r, 1,dtype=torch.cdouble).to(device)
        for i in range(user_number):
            for j in range(user_number):
                HV_matrics[:, i, j, :, :] = H[:, i, :, :] @ V_last[:, j, :, :]
                if j==i:
                    HV_matrics2[:,i,:,:]=HV_matrics[:, i, j, :, :]

        _temp_HV = torch.zeros(N_samples, user_number, user_number, N_r, N_r,dtype=torch.cdouble).to(device)
        # 计算 T
        for i in range(user_number):
            for j in range(user_number):
                _temp_HV[:,i,j,:,:]=HV_matrics[:, i, j,:,:] @ HV_matrics[:, i, j,:,:].mH

        #更新Y
        U=noise_power*torch.eye(N_r,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).repeat(N_samples, user_number, 1, 1).to(device)
        # 更新U
        for k in range(user_number):
            U[:, k, :, :] = U[:, k, :, :]+ _temp_HV[:, k, :, :, :].sum(dim=1)
        Y=torch.linalg.inv(U)@HV_matrics2

        #更新Gamma
        F=torch.zeros(N_r,N_r,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).repeat(N_samples, user_number, 1, 1).to(device)
        # 更新F
        for k in range(user_number):
            F[:, k, :, :] = U[:, k, :, :] - _temp_HV[:, k, k, :, :]
        
        # 默认单流 Gamma 是一个数字 若多流 Gamma是矩阵
        Gamma=HV_matrics2.mH @torch.linalg.inv(F)@ HV_matrics2


        #临时变量
        H_HY = H.mH @ Y
        #更新V
        identity_matrices = 1 #单流 Gamma是数字
        Lambda= H_HY @ (identity_matrices+Gamma)
        Lambda =Lambda* torch.tensor(w).view(1, user_number, 1, 1).to(device)

        L = H_HY @ (identity_matrices+Gamma) @ H_HY.mH
        L = L* torch.tensor(w).view(1, user_number, 1, 1).to(device)
        L = L.sum(dim=1,keepdim=True)

        L_matrices = L[:,0,:,:]
        max_eigenvalues = torch.linalg.eigvalsh(L_matrices).max(dim=1)[0]

        max_eigenvalues= 1.0 / max_eigenvalues
        # 将最大特征值扩展为 [N, T, M, M] 以便与矩阵 B 进行广播数乘
        max_eigenvalues_expanded = max_eigenvalues.view(N_samples, 1, 1, 1).expand(N_samples, user_number, N_t, 1)#默认单流数据
        V = Z + max_eigenvalues_expanded*(Lambda - L @ Z)

        #功率归一化
        frobenius_norm_squared = torch.linalg.norm(V, ord='fro', dim=(-2, -1))**2
        frobenius_norm_squared=frobenius_norm_squared.sum(dim=1) #所有用户的总功率
        # 找出 Frobenius 范数的平方大于 P 的矩阵的掩码
        mask = frobenius_norm_squared > signal_max_power
        if mask.any():
            # 计算当前矩阵的 Frobenius 范数的平方
            current_norm_squared = frobenius_norm_squared[mask]
            # 计算缩放系数
            scale_factors = (signal_max_power / current_norm_squared).sqrt()
            # # 对符合条件的矩阵进行缩放
            V[mask] *= scale_factors.view(-1, 1, 1 , 1)

        return V

    def single_iteration(self,H,V_last,w,signal_max_power=1e2,noise_power=1e-7):
        device=H.device

        Z=V_last.clone()
        d=V_last.size(-1)#数据流数

        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        HV_matrics = torch.zeros(N_samples, BS_number, BS_number, user_number, user_number, N_r, d ,dtype=torch.cdouble).to(device)
        HV_matrics2=torch.zeros(N_samples, BS_number, user_number, N_r, d ,dtype=torch.cdouble).to(device)
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


        #更新Y
        U=noise_power*torch.eye(N_r,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
        # 更新U
        for l in range(BS_number):
            for k in range(user_number):
                U[:, l, k, :, :] = U[:, l, k, :, :] + _temp_HV[:,:,l, k, :, :, :].sum(dim=1).sum(dim=1)
        Y=torch.linalg.inv(U)@HV_matrics2

        #更新Gamma
        F=torch.zeros(N_r,N_r,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
        # 更新F
        for l in range(BS_number):
            for k in range(user_number):
                F[:, l, k, :, :] = U[:, l, k, :, :] - _temp_HV[:, l, l, k, k, :, :]
        

        # 默认单流 Gamma 是一个数字 若多流 Gamma是矩阵
        Gamma=HV_matrics2.mH @torch.linalg.inv(F)@ HV_matrics2


        L = torch.zeros(N_samples,BS_number,user_number,N_t,N_t,dtype=torch.cdouble).to(device)
        Lambda = torch.zeros(N_samples,BS_number,user_number,N_t,d,dtype=torch.cdouble).to(device)
        identity_matrices = torch.eye(d,dtype=torch.cdouble).to(device) #单流 Gamma是数字
        for l in range(BS_number):
            H_HY=H[:,l,:,:,:].mH @ Y    #NMd
            L[:,l,:,:,:] = (H_HY @ (identity_matrices+Gamma) @ H_HY.mH).sum(dim=1)

            Lambda[:,l,:,:,:] = H[:,l,l,:,:,:].mH @ Y[:,l,:,:,:] @ (identity_matrices+Gamma)[:,l,:,:,:]

        Lambda =Lambda* torch.tensor(w).view(1 ,1, user_number, 1, 1).to(device)

        L = L* torch.tensor(w).view(1,1, user_number, 1, 1).to(device)
        L = L.sum(dim=2,keepdim=True)#对用户维度求和

        L_matrices = L[:,:,0,:,:]
        max_eigenvalues = torch.linalg.eigvalsh(L_matrices).max(dim=2)[0]


        max_eigenvalues= 1.0 / max_eigenvalues
        # 将最大特征值扩展为 [N, T, M, M] 以便与矩阵 B 进行广播数乘
        max_eigenvalues_expanded = max_eigenvalues.view(N_samples, BS_number ,1, 1, 1).expand(N_samples, BS_number, user_number, N_t, d)#多流数据设置

        # print((Lambda - L @ Z).abs().mean())
        
        V = Z + max_eigenvalues_expanded*(Lambda - L @ Z)

        #功率归一化
        frobenius_norm_squared = torch.linalg.norm(V, ord='fro', dim=(-2, -1))**2
        frobenius_norm_squared=frobenius_norm_squared.sum(dim=2) #所有用户的总功率
        # 找出 Frobenius 范数的平方大于 P 的矩阵的掩码
        mask = frobenius_norm_squared > signal_max_power
        if mask.any():
            # 计算当前矩阵的 Frobenius 范数的平方
            current_norm_squared = frobenius_norm_squared[mask]
            # 计算缩放系数
            scale_factors = (signal_max_power / current_norm_squared).sqrt()
            # # 对符合条件的矩阵进行缩放
            V[mask] *= scale_factors.view(-1, 1, 1 , 1)

        return V
    
    def init_input(self, input_shape, signal_max_power=1e2):
        N_samples,BS_number,user_number,N_t,stream_num=input_shape

        # V=math.sqrt(signal_max_power/user_number)*torch.ones(N_t,dtype=torch.cdouble).reshape(1,1,1,N_t,1).repeat(N_samples,BS_number, user_number, 1, 1)

        # return V
        V=torch.randn(N_samples,BS_number,user_number,N_t,stream_num,dtype=torch.cdouble)
        frobenius_norm_squared = torch.linalg.norm(V, ord='fro', dim=(-2, -1))**2
        frobenius_norm_squared=frobenius_norm_squared.sum(dim=2) #所有用户的总功率
        scale_factors = (signal_max_power / frobenius_norm_squared).sqrt()
        V *= scale_factors.view(N_samples,BS_number, 1, 1 , 1)

        return V
    

    def performe(self,H,w,iterations,d=2,signal_max_power=1e2,noise_power=1e-7):
        device=H.device
        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        V=self.init_input((N_samples,BS_number,user_number,N_t,d),signal_max_power=signal_max_power).to(device)

        wsr_record=[0 for i in range(iterations+1)]
        for i in range(iterations+1):
            print(i)
            wsr_record[i]=WSR(H,V,w,noise_power=noise_power)
            print(wsr_record[i])
            V=self.single_iteration(H,V,w,signal_max_power=signal_max_power,noise_power=noise_power)

        return wsr_record
    

    def forward(self,H,w,iterations,d=2,signal_max_power=1e2,noise_power=1e-7,init_V=None):
        device=H.device
        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        if not torch.is_tensor(init_V):
            V=self.init_input((N_samples,BS_number,user_number,N_t,d),signal_max_power=signal_max_power).to(device)
        else:
            V=init_V.to(device)

        for _ in range(iterations):
            V=self.single_iteration(H,V,w,signal_max_power=signal_max_power,noise_power=noise_power)

        return V
    
    
    def forward_single_cell(self,H,w,iterations,d=2,signal_max_power=1e2,noise_power=1e-7,init_V=None):
        device=H.device
        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        if not torch.is_tensor(init_V):
            V=self.init_input((N_samples,BS_number,user_number,N_t,d),signal_max_power=signal_max_power).to(device)
        else:
            V=init_V.to(device)

        V=V[:,0,:,:,:]

        for _ in range(iterations):
            V=self.single_iteration_single_cell(H[:,0,0,:,:,:],V,w,signal_max_power=signal_max_power,noise_power=noise_power)

        return V