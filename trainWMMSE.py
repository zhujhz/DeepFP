import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

#导入自定义函数
from FP_unfolding.generate_data import split_matrix
from FP_unfolding.utils import dataset
from FP_unfolding.IAIDNN_net import IAIDNNnet
from FP_unfolding.WMMSE import WMMSE
from FP_unfolding.compute_WSR import WSR
from FP_unfolding.load_chn import load_H_BS1 as load_H

# 定义系统参数
N_t=64
N_r=4
d=2
user_number=6
BS_number=1
w=[i+1 for i in range(user_number)] 
w=[w[i]/sum(w) for i in range(user_number)] #创建等差权重矩阵
signal_max_power=1e2
noise_power=1e-8

# 定义网络参数
N_samples = 50000
epochs = 200
supervisor_epoch_num = 1
# supervisor_lr = 1.38*1e-4
supervisor_lr = 5*1e-2
unsupervisor_lr = 1.38*1e-5
batch_size = 200
net_iterations = 7


def lr_lambda(epoch):
    if epoch < 50:
        return 1.0  # 保持初始学习率
    else:
        return unsupervisor_lr / supervisor_lr  # 调整到目标学习率
    

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
        self.wmmse=WMMSE()
    

    def forward(self, H, V, w, V_init=None, mode='unsuper',target_iters=10):
        # 自定义损失的计算
        if mode=="unsuper":
            return -WSR(H,V,w,noise_power=self.noise_power,keep_tensor=True)
        ##mode=="super"
        V_opt=self.wmmse(H,w,target_iters,d=V.size(-1),signal_max_power=self.signal_max_power,noise_power=self.noise_power,init_V=V_init)

        return self.mse(V,V_opt)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using: ",device)


Data_blocks=N_samples//5000
H=[]
for i in range(Data_blocks):
    H.append(load_H(f"Data/{i+1}_5000_Nr_{N_r}_Nt_{N_t}_N_user_6_BS_{BS_number}.mat"))
H=torch.cat(H,dim=0)
# 分割数据
train_set, val_set, test_set = split_matrix(H=H,train_size=0.9)


# 创建DataLoder
trainset = dataset(train_set)
trainloader = torch.utils.data.DataLoader(trainset,batch_size = batch_size)

valset = dataset(val_set)
valloader = torch.utils.data.DataLoader(valset,batch_size = min(val_set.size(0),1000))
dataloaders = [trainloader,valloader]


Alg_test_data=train_set[:10].to(device)
wmmse=WMMSE()
Alg_WSR=wmmse.performe(Alg_test_data,w,10,d=d,signal_max_power=signal_max_power,noise_power=noise_power)


def train_net(net,epochs,loss_fc,optimizer,lr_scheduler,dataloaders,supervisor_epochs=-1):
    criterion = loss_fc

    trainloader,valloader = dataloaders
    train_epoch_losses,val_epoch_losses = [],[]
    for epoch in range(epochs):
        batch_losses = []
        # Print=True
        for batch_idx,H in enumerate(trainloader):
            
            H = H.to(device)
            optimizer.zero_grad()

            V_init=net.init_input(H.size(0))
            V,V_out_record = net(H,action_noise_power=0,init_V=V_init)

            if epoch == 0:
                print(batch_idx)

            if epoch < supervisor_epochs:
                loss = criterion(H,V,net.weight,V_init=V_init,mode='super',target_iters=30)
            else:
                loss = criterion(H,V,net.weight,mode='unsuper')
                # for V_middle in V_out_record[:-1]:
                #     loss += (1-discount_factor)*criterion(H,V_middle,net.weight,mode='unsuper')
            # print(batch_idx+1)

            loss.backward()
            # print(batch_idx+2)
            optimizer.step()
            batch_losses.append(loss.item())

        
        lr_scheduler.step()
        print("epoch: ",epoch,"epoch loss: ",np.mean(batch_losses))
        train_epoch_losses.append(np.mean(batch_losses))

        net.eval()

        with torch.no_grad():
            batch_losses = []
            for batch_idx,H in enumerate(valloader):          
                H = H.to(device)
                V,V_out_record = net(H)
                loss = criterion(H,V,net.weight,mode='unsuper')
                batch_losses.append(loss.item())

            print("val loss: ",np.mean(batch_losses))
            val_epoch_losses.append(np.mean(batch_losses))

        net.train()
        print("-------------------------------------------------------------------------------------")
    return train_epoch_losses,val_epoch_losses


net = IAIDNNnet(BS_number=BS_number,user_number=user_number,N_t=N_t,N_r=N_r,w=w,d=d,iterations=net_iterations,device=device,signal_max_power=signal_max_power,noise_power=noise_power)


# state = torch.load("models\Wmmse_model_weights_20241122-022442.pth")
# net.load_state_dict(state)
# 为每个 block 设置不同的学习率
# params = []

# for i, block in enumerate(net.blocks):
#     params.append({'params': block.parameters(), 'lr': supervisor_lr*(discount_factor**(net_iterations-1-i))})
# # 初始化优化器
# optimizer = optim.Adam(params)

optimizer = optim.Adam(net.parameters(), lr = supervisor_lr , weight_decay = 0)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
loss_fuc=LossUnfolding(signal_max_power=signal_max_power,noise_power=noise_power)


# 训练并记录时间
start_time=time.time()
train_loss,val_loss= train_net(net=net,epochs=epochs,loss_fc=loss_fuc,optimizer=optimizer,lr_scheduler=scheduler,dataloaders=dataloaders,supervisor_epochs=supervisor_epoch_num)
end_time = time.time()
# 计算运行时间
elapsed_time = end_time - start_time
print(f"The net took {elapsed_time} seconds to train.")

# 获取当前时间并格式化为字符串
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# 保存模型参数
torch.save(net.state_dict(), f'models\Wmmse_model_weights_{timestamp}.pth')

# 保存训练和验证误差
np.save(f'Loss\Wmmse_Tran_Val_loss_{timestamp}.npy', (np.array(train_loss,dtype=np.float64),np.array(val_loss,dtype=np.float64)))


# Net_WSR=net.performe(Alg_test_data)


# 创建一个包含两个子图的图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 绘制训练损失
ax1.plot(train_loss, label='Training Loss', color='blue')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid()

# 绘制验证损失
ax2.plot(val_loss, label='Validation Loss', color='orange')
ax2.set_title('Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid()

# 调整布局
plt.tight_layout()
plt.show()

