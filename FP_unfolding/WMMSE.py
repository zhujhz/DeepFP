import torch
import torch.nn as nn
import torch.utils.data

from FP_unfolding.compute_WSR import WSR


class WMMSE(nn.Module):
    def single_iteration_single_BS(self,H,V_last,w,signal_max_power=1e2,noise_power=1e-7):
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


        #更新V
        V=torch.zeros_like(V_last,dtype=torch.cdouble).to(device)

        # for n in range(N_samples):
        _temp_Matrix=torch.zeros(N_samples,N_t,N_t, dtype=torch.cdouble).unsqueeze(1).unsqueeze(1).to(device)
        beta_u=torch.zeros(N_samples,dtype=torch.cdouble).to(device)
        for idx_BS in range(BS_number):
            for idx_user in range(user_number):
                beta_u+=(noise_power/signal_max_power)*(w[idx_user]*Y[:,l,idx_user,:,:]@W[:,l,idx_user,:,:]@Y[:,l,idx_user,:,:].mH).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                _temp_Matrix[:,0,0,:,:]+=w[idx_user]*H[:,l,idx_BS,idx_user,:,:].mH@Y[:,idx_BS,idx_user,:,:]@W[:,idx_BS,idx_user,:,:]@Y[:,idx_BS,idx_user,:].mH@H[:,l,idx_BS,idx_user,:,:]
        _temp_Matrix+=beta_u.view(N_samples,1,1,1,1)*torch.eye(N_t, dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

        for k in range(user_number):
            V[:,l,k,:,:]=w[k]*torch.linalg.inv(_temp_Matrix[:,0,0,:,:])@H[:,l,l,k,:,:].mH@Y[:,l,k,:,:]@W[:,l,k,:,:]

        # def compute_V_power(u):
        #     # for k in range(user_number):
        #     _temp_Matrix=u*torch.eye(N_t, dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
        #     for idx_BS in range(BS_number):
        #         for idx_user in range(user_number):
        #             _temp_Matrix[n,0,0,:,:]+=w[idx_user]*H[n,l,idx_BS,idx_user,:,:].mH@Y[n,idx_BS,idx_user,:,:]@W[n,idx_BS,idx_user,:,:]@Y[n,idx_BS,idx_user,:].mH@H[n,l,idx_BS,idx_user,:,:]
            
        #     for k in range(user_number):
        #         V[n,l,k,:,:]=w[k]*torch.linalg.inv(_temp_Matrix)@H[n,l,l,k,:,:].mH@Y[n,l,k,:,:]@W[n,l,k,:,:]
        #     #检查功率
        #     frobenius_norm_squared=torch.linalg.norm(V, ord='fro', dim=(-2, -1))**2
        #     frobenius_norm_squared=frobenius_norm_squared.sum(dim=2) #所有用户的总功率

        #     return frobenius_norm_squared


        # for n in range(N_samples):
        #     #正是由于每个样本都要进行二分法 WMMSE不适合多样本并行
        #     for l in range(BS_number):
        #         frobenius_norm_squared=compute_V_power(0)
        #         if not frobenius_norm_squared[n,l]>signal_max_power:
        #             continue

        #         low,high=0,10000

        #         while True:
        #             u=(low+high)/2
        #             frobenius_norm_squared=compute_V_power(u)

        #             if abs(frobenius_norm_squared[n,l]-signal_max_power)<1e-4:
        #                break

        #             if frobenius_norm_squared[n,l]<signal_max_power:
        #                 high=u
        #             else:
        #                 low=u

        return V

    def single_iteration(self,H,V_last,w,signal_max_power=1e2,noise_power=1e-7):
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


        #更新Y  Y即WMMSE中的U矩阵
        U=noise_power*torch.eye(N_r,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
        # 更新U
        for l in range(BS_number):
            for k in range(user_number):
                U[:, l, k, :, :] = U[:, l, k, :, :] + _temp_HV[:,:,l, k, :, :, :].sum(dim=1).sum(dim=1)
        Y=torch.linalg.inv(U)@HV_matrics2


        #更新W
        W=torch.eye(d, dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
        for l in range(BS_number):
            for k in range(user_number):
                W[:, l, k, :, :] = W[:, l, k, :, :] - Y[:,l,k,:,:].mH @ HV_matrics2[:,l,k,:,:]
        W=torch.linalg.inv(W)


        #更新V
        V=torch.zeros_like(V_last,dtype=torch.cdouble).to(device)

        def compute_V_power(u):
            # for k in range(user_number):
            _temp_Matrix=u*torch.eye(N_t, dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            for idx_BS in range(BS_number):
                for idx_user in range(user_number):
                    _temp_Matrix[n,0,0,:,:]+=w[idx_user]*H[n,l,idx_BS,idx_user,:,:].mH@Y[n,idx_BS,idx_user,:,:]@W[n,idx_BS,idx_user,:,:]@Y[n,idx_BS,idx_user,:].mH@H[n,l,idx_BS,idx_user,:,:]
            
            for k in range(user_number):
                V[n,l,k,:,:]=w[k]*torch.linalg.inv(_temp_Matrix)@H[n,l,l,k,:,:].mH@Y[n,l,k,:,:]@W[n,l,k,:,:]
            #检查功率
            frobenius_norm_squared=torch.linalg.norm(V, ord='fro', dim=(-2, -1))**2
            frobenius_norm_squared=frobenius_norm_squared.sum(dim=2) #所有用户的总功率

            return frobenius_norm_squared


        for n in range(N_samples):
            #正是由于每个样本都要进行二分法 WMMSE不适合多样本并行
            for l in range(BS_number):
                frobenius_norm_squared=compute_V_power(0)
                if not frobenius_norm_squared[n,l]>signal_max_power:
                    continue

                low,high=0,10000

                while True:
                    u=(low+high)/2
                    frobenius_norm_squared=compute_V_power(u)

                    if abs(frobenius_norm_squared[n,l]-signal_max_power)<1e-4:
                       break

                    if frobenius_norm_squared[n,l]<signal_max_power:
                        high=u
                    else:
                        low=u

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
            wsr_record[i]=WSR(H,self.adjust_power(V,signal_max_power),w,noise_power=noise_power)
            print(wsr_record[i])
            # for n in range(N_samples):
            #     V[n:n+1]=self.single_iteration_single_BS(H[n:n+1],V[n:n+1],w,signal_max_power=signal_max_power,noise_power=noise_power)
            V=self.single_iteration_single_BS(H,V,w,signal_max_power=signal_max_power,noise_power=noise_power)

        return wsr_record
    
    def performe_bis(self,H,w,iterations,d=2,signal_max_power=1e2,noise_power=1e-7):
        device=H.device
        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        V=self.init_input((N_samples,BS_number,user_number,N_t,d),signal_max_power=signal_max_power).to(device)

        wsr_record=[0 for i in range(iterations+1)]
        for i in range(iterations+1):
            print(i)
            wsr_record[i]=WSR(H,self.adjust_power(V,signal_max_power),w,noise_power=noise_power)
            print(wsr_record[i])
            for n in range(N_samples):
                V[n:n+1]=self.single_iteration(H[n:n+1],V[n:n+1],w,signal_max_power=signal_max_power,noise_power=noise_power)
            # V=self.single_iteration(H,V,w,signal_max_power=signal_max_power,noise_power=noise_power)

        return wsr_record
    

    def adjust_power(self,V,signal_max_power):
        #在最后一步需要重整功率
        frobenius_norm_squared = torch.linalg.norm(V, ord='fro', dim=(-2, -1))**2
        frobenius_norm_squared=frobenius_norm_squared.sum(dim=2) #所有用户的总功率
        #         mask = frobenius_norm_squared > signal_max_power
        # if mask.any():
            # 计算当前矩阵的 Frobenius 范数的平方
        current_norm_squared = frobenius_norm_squared
        # 计算缩放系数
        scale_factors = (signal_max_power / current_norm_squared).sqrt()

        V2=V.clone()
        # # 对符合条件的矩阵进行缩放
        V2 *= scale_factors.view(-1,1,1,1,1)

        return V2

    

    def forward(self,H,w,iterations,d=2,signal_max_power=1e2,noise_power=1e-7,init_V=None):
        device=H.device
        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        if not torch.is_tensor(init_V):
            V=self.init_input((N_samples,BS_number,user_number,N_t,d),signal_max_power=signal_max_power).to(device)
        else:
            V=init_V.to(device)

        for _ in range(iterations):
            # for n in range(N_samples):
            #     V[n:n+1]=self.single_iteration_single_BS(H[n:n+1],V[n:n+1],w,signal_max_power=signal_max_power,noise_power=noise_power)
            V=self.single_iteration_single_BS(H,V,w,signal_max_power=signal_max_power,noise_power=noise_power)

        V2=self.adjust_power(V,signal_max_power)

        return V2
    

