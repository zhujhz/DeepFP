import torch
import torch.nn as nn
import math
from FP_unfolding.compute_WSR import WSR



def trace_list(H):
    # 1. 提取主对角线 (对每个 BxB 矩阵沿第 1 和第 2 维操作)
    diagonals = H.diagonal(dim1=-2, dim2=-1)  # 结果形状: (N, B, K, T, R)

    # 2. 对主对角线求和
    trace = diagonals.sum(dim=-1)  # 结果形状: (N, K, T, R)
    return trace


class ComplexReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # 分离实部和虚部
        real = input.real
        imag = input.imag
        
        # 对实部应用 ReLU 激活函数
        real = torch.relu(real)
        
        # 重新组合复数
        return real + 1j * imag
class ModReLU(nn.Module):
    def __init__(self, bias_init=0.1):
        super(ModReLU, self).__init__()
        self.bias = nn.Parameter(torch.tensor(bias_init, dtype=torch.float32))  # 可学习参数

    def forward(self, z):
        # z: 复数张量 (形状为 (batch_size, ...))
        modulus = torch.abs(z)  # 计算模值 |z|
        phase = z / (modulus + 1e-8)  # 保留相位信息
        # 激活函数逻辑
        activated_modulus = torch.relu(modulus - self.bias)
        return activated_modulus * phase  # 修正模值并组合相位

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        """
        p: dropout 的概率
        """
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            # 生成与输入大小一致的掩码，且对实部和虚部共享
            dropout_mask = torch.bernoulli(torch.ones(x.shape[:-1]) * (1 - self.p)).to(x.device)
            dropout_mask = dropout_mask.unsqueeze(-1)  # 保证在最后一维进行广播
            return dropout_mask * x / (1 - self.p)
        else:
            return x


class FastFPnet(nn.Module):
    def __init__(self,BS_number,user_number,N_r,N_t,w,d,iterations,device,hidden_size=200,signal_max_power=1e2,noise_power=1e-7):

        # super(FastFPnet).__init__()
        super().__init__()

        self.BS_number=BS_number
        self.user_number=user_number#用户数量
        self.N_r=N_r#接收天线数量
        self.N_t=N_t#发送天线数量
        self.weight=w#权重向量
        self.device=device
        self.d=d
        self.hidden_size=hidden_size

        self.iterations=iterations#网络unfolding的迭代次数

        ## defining learnable parameters
        self.blocks=nn.ModuleList([self.init_block(hidden_size).to(device) for _ in range(self.iterations)])

        # self.gamma=nn.Parameter(0.2*torch.ones(iterations,user_number).to(device),requires_grad=True)

        # #线型组合系数
        # self.alpha=nn.Parameter(torch.ones(iterations,user_number).to(device),requires_grad=True)
        # self.beta=nn.Parameter(torch.zeros(iterations,user_number).to(device),requires_grad=True)


        ## defining unearnable parameters
        self.noise_power=torch.tensor([noise_power]).to(device)
        self.signal_max_power=signal_max_power

    def init_block(self,hidden_size=64,dropout_rate=0.2):
        block=nn.Sequential(
            nn.Linear(2*self.N_t*self.d, hidden_size, dtype=torch.cdouble),  # 第一层
            ComplexReLU(),
            # ModReLU(),
            # ComplexDropout(),
            nn.Linear(hidden_size , hidden_size, dtype=torch.cdouble),  # 第一层
            ComplexReLU(),
            # ModReLU(),
            # ComplexDropout(),
            nn.Linear(hidden_size , hidden_size, dtype=torch.cdouble),  # 第一层                                        
            ComplexReLU(),
            # ModReLU(),
            # ComplexDropout(),
            nn.Linear(hidden_size , 1, dtype=torch.cdouble),  # 第一层
        )
        return block
    

    def block_forward(self, block, Z1, Z2):

        N_samples,BS_number,user_number,N_t,_=Z1.size()
        Z1=Z1.view(N_samples,BS_number,user_number,-1)
        Z2=Z2.view(N_samples,BS_number,user_number,-1)

        Input=torch.cat((Z1,Z2),dim=-1)
        Output=block(Input)

        Output=Output.real #先考虑不消除复数

        return Output
    

    def interation_forward(self,H,V_last,iteration_idx,w=None,signal_max_power=None,noise_power=None,action_noise_power=0,obs_step=False):

        # device=self.device
        # signal_max_power=self.signal_max_power
        # noise_power=self.noise_power
        # w=self.weight
        # d=self.d
        # N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        device=H.device
        d=V_last.size(-1)#数据流数
        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        if w is None:
            w=self.weight
        if signal_max_power is None:
            signal_max_power=self.signal_max_power
        if noise_power is None:
            noise_power=self.noise_power

        Z=V_last.clone()   

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
            H_HY=H[:,l,:,:,:].mH @ Y
            L[:,l,:,:,:] = (H_HY @ (identity_matrices+Gamma) @ H_HY.mH).sum(dim=1)

            Lambda[:,l,:,:,:] = H[:,l,l,:,:,:].mH @ Y[:,l,:,:,:] @ (identity_matrices+Gamma)[:,l,:,:,:]

        Lambda =Lambda* torch.tensor(w).view(1 ,1, user_number, 1, 1).to(device)

        L = L* torch.tensor(w).view(1,1, user_number, 1, 1).to(device)


        # L_o = L.sum(dim=2,keepdim=True)     


        Z2=Lambda-L @ Z
        step=self.block_forward(
            block=self.blocks[iteration_idx],
            Z1=Z,
            Z2=Z2
        )


        if obs_step:
            L_o = L.sum(dim=2,keepdim=True)#对用户维度求和

            L_matrices = L_o[:,:,0,:,:]
            max_eigenvalues = torch.linalg.eigvalsh(L_matrices).max(dim=2)[0]

            # max_eigenvalues= 1.0 / max_eigenvalues

            # max_eigenvalues=max_eigenvalues
            # 将最大特征值扩展为 [N, T, M, M] 以便与矩阵 B 进行广播数乘
            alg_step = max_eigenvalues.view(N_samples, BS_number ,1, 1, 1).expand(N_samples, BS_number, user_number, 1, 1).detach().cpu()
            net_step = (1.0/step).view(N_samples,BS_number,user_number, 1, 1).detach().cpu()


        step=step.view(N_samples,BS_number,user_number,1,1).expand(N_samples,BS_number,user_number, N_t, 1)

        # V=Z+step*(Lambda-L @ Z)
        V=Z+step*Z2
        

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

            V=V.clone()
            # # 对符合条件的矩阵进行缩放
            V[mask] *= scale_factors.view(-1, 1, 1 , 1)

        if not obs_step:
            return V
        
        low_matrices = Z.mH @ L_o @ Z
        I=torch.eye(N_t,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
        upper_matrices = net_step.to(device)*Z.mH@Z + 2*(V.mH @ (L_o-net_step.to(device)*I) @ Z).real-V.mH @ (L_o-net_step.to(device)*I) @ V
        upper_matrices_o = alg_step.to(device)*Z.mH@Z + 2*(V.mH @ (L_o-alg_step.to(device)*I) @ Z).real-V.mH @ (L_o-alg_step.to(device)*I) @ V


        low_matrices=low_matrices.detach().cpu()
        upper_matrices=upper_matrices.detach().cpu()
        upper_matrices_o=upper_matrices_o.detach().cpu()


        diff=trace_list(upper_matrices)-trace_list(low_matrices)
        diff_orignal=trace_list(upper_matrices_o)-trace_list(low_matrices)
        
        return V,alg_step,net_step,diff.real,diff_orignal.real


    def init_input(self,N_samples,input_shape=None,signal_max_power=None):
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

        return V

    def performe(self,H,w=None,iterations=None,signal_max_power=None,noise_power=None,obs_step=False):
        device=H.device
        N_samples,BS_number,_,user_number,N_r,N_t=H.size()

        if not iterations: iterations=self.iterations
        if signal_max_power is None: 
            signal_max_power=self.signal_max_power
        if w is None:
            w=self.weight
        if noise_power is None:
            noise_power=self.noise_power

        V=self.init_input(N_samples,(N_samples,BS_number,user_number,N_t,self.d),signal_max_power).to(device)# 目前并不实现对于d的通用

        wsr_record=[0 for i in range(iterations+1)]
        wsr_record[0]=WSR(H,V,w,noise_power=noise_power)
        print(0)
        print(wsr_record[0])

        alg_steps=[]
        net_steps=[]
        diff_steps=[]
        diffo_steps=[]

        for i in range(iterations):

            if not obs_step:
                V=self.interation_forward(H,V,min(i,self.iterations-1),w=w,signal_max_power=signal_max_power,noise_power=noise_power).detach()
            else:
                V,alg_step,net_step,diff,diff_o=self.interation_forward(H,V,min(i,self.iterations-1),w=w,signal_max_power=signal_max_power,noise_power=noise_power,obs_step=True)

                alg_steps.append(alg_step)
                net_steps.append(net_step)
                diff_steps.append(diff)
                diffo_steps.append(diff_o)


            wsr_record[i+1]=WSR(H,V,w,noise_power=noise_power)

            print(i+1)
            print(wsr_record[i+1])

        if not obs_step:
            return wsr_record
        return wsr_record,alg_steps,net_steps,diff_steps,diffo_steps


    
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