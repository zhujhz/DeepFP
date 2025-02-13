import torch
import torch.nn as nn
import numpy as np

def WSR(H,V,w,noise_power=1e-7,keep_tensor=False):
    device=H.device
    N_samples,BS_number,_,user_number,N_r,N_t=H.size()
    d=V.size(-1)#数据流数

    HV_matrics = torch.zeros(N_samples, BS_number, BS_number, user_number, user_number, N_r, d ,dtype=torch.cdouble).to(device)
    HV_matrics2 = torch.zeros(N_samples, BS_number, user_number, N_r, d ,dtype=torch.cdouble).to(device)
    for k in range(BS_number):
        for l in range(BS_number):
            for i in range(user_number):
                for j in range(user_number):
                    HV_matrics[:, k, l, i, j, :, :] = H[:, k, l, i, :, :] @ V[:, k, j, :, :] #H_{k,li}W{k,j}
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

    Gamma+=torch.eye(d).view(1,1,1,d,d).expand_as(Gamma).to(device)


    R=[0 for _ in range(len(w))]
    if keep_tensor: #
        for k in range(user_number):
            R[k]=torch.mean(torch.log2(torch.det(Gamma[:,:,k]).real))
    else:
        for k in range(user_number):
            R[k]=torch.mean(torch.log2(torch.det(Gamma[:,:,k]).real)).item()


    return sum([w[i]*R[i] for i in range(user_number)])


def WSR2(H,V,w,noise_power=1e-7,keep_tensor=False):
    device=H.device
    N_samples,BS_number,_,user_number,N_r,N_t=H.size()
    d=V.size(-1)#数据流数

    HV_matrics = torch.zeros(N_samples, BS_number, BS_number, user_number, user_number, N_r, d ,dtype=torch.cdouble).to(device)
    HV_matrics2 = torch.zeros(N_samples, BS_number, user_number, N_r, d ,dtype=torch.cdouble).to(device)
    for k in range(BS_number):
        for l in range(BS_number):
            for i in range(user_number):
                for j in range(user_number):
                    HV_matrics[:, k, l, i, j, :, :] = H[:, k, l, i, :, :] @ V[:, k, j, :, :] #H_{k,li}W{k,j}
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
    # Y=torch.linalg.inv(U)@HV_matrics2

    #更新Gamma
    F=torch.zeros(N_r,N_r,dtype=torch.cdouble).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(N_samples, BS_number, user_number, 1, 1).to(device)
    # 更新F
    for l in range(BS_number):
        for k in range(user_number):
            F[:, l, k, :, :] = U[:, l, k, :, :] - _temp_HV[:, l, l, k, k, :, :]
    
    # 默认单流 Gamma 是一个数字 若多流 Gamma是矩阵
    Gamma=HV_matrics2.mH @torch.linalg.inv(F)@ HV_matrics2

    Gamma+=torch.eye(d).view(1,1,1,d,d).expand_as(Gamma).to(device)

    # return Gamma

    R=np.zeros((N_samples,BS_number,user_number))

    for k in range(user_number):
        R[:,:,k]=w[k]*torch.log2(torch.det(Gamma[:,:,k]).real).cpu().detach().numpy()

    return np.sum(R, axis=(-2,-1))


    # R=[0 for _ in range(len(w))]
    # if keep_tensor: #
    #     for k in range(user_number):
    #         R[k]=torch.mean(torch.log2(torch.det(Gamma[:,:,k]).real))
    # else:
    #     for k in range(user_number):
    #         R[k]=torch.mean(torch.log2(torch.det(Gamma[:,:,k]).real)).item()


    # return sum([w[i]*R[i] for i in range(user_number)])