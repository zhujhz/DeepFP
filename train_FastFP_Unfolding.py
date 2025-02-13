import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Import custom functions
from FP_unfolding.generate_data import split_matrix
from FP_unfolding.utils import dataset
from FP_unfolding.FastFP_net import FastFPnet
from FP_unfolding.FastFP_alg import FastFP
from FP_unfolding.loss_fuction import LossUnfolding
from FP_unfolding.compute_WSR import WSR
from FP_unfolding.load_chn import load_H, load_H_BS1
from FP_unfolding.load_chn import load_H_3GPP

import argparse

def train_FPnet(net, epochs, loss_fc, optimizer, lr_scheduler, dataloaders, supervisor_batchs=-1, timestamp=None, record_interval=None, exp_name=None):
    
    criterion = loss_fc

    trainloader, valloader = dataloaders
    time_points, train_epoch_losses, val_epoch_losses = [], [], []

    def eval_net():
        """Evaluate the network on the validation set"""
        net.eval()

        with torch.no_grad():
            batch_losses = []
            for batch_idx, H in enumerate(valloader):          
                H = H.to(net.device)
                V, V_out_record = net(H)
                loss = criterion(H, V, net.weight, mode='unsuper')
                batch_losses.append(loss.item())

        net.train()
        return np.mean(batch_losses)
    
    # Initialize time and loss tracking
    time_points.append(0)
    val_epoch_losses.append(eval_net())
    train_epoch_losses.append(0)
    print("Elapsed_time: ", 0, "Val loss: ", val_epoch_losses[-1])
    start_time = time.time()
    last_record_time = start_time

    batch_losses = []
    for epoch in range(epochs):
        for batch_idx, H in enumerate(trainloader):
            H = H.to(net.device)
            optimizer.zero_grad()

            V_init = net.init_input(H.size(0))
            V, V_out_record = net(H, action_noise_power=0, init_V=V_init)

            # Use supervised loss for initial batches, then switch to unsupervised
            if supervisor_batchs >= 0:
                supervisor_batchs -= 1
                loss = criterion(H, V, net.weight, V_init=V_init, mode='super', target_iters=40)
            else:
                loss = criterion(H, V, net.weight, mode='unsuper')

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            # Record losses at specified intervals
            if time.time() - last_record_time > record_interval:
                last_record_time = time.time()
                elapsed_time = last_record_time - start_time
                time_points.append(elapsed_time)

                train_epoch_losses.append(np.mean(batch_losses))
                print("Elapsed_time: ", elapsed_time, "Train loss: ", train_epoch_losses[-1])
                batch_losses = []

                val_epoch_losses.append(eval_net())
                print("Elapsed_time: ", elapsed_time, "Val loss: ", val_epoch_losses[-1])

        # lr_scheduler.step()  # Uncomment if you want to use learning rate scheduler
        print("epoch: ", epoch, " is finished------------------------------------")

        # Save model after each epoch
        torch.save(net.state_dict(), f'models\{exp_name}_model_weights_{timestamp}.pth')

        # Save the training and validation errors
        np.save(f'Loss\{exp_name}_Time_Tran_Val_loss_{timestamp}.npy', (np.array(time_points, dtype=np.float64), np.array(train_epoch_losses, dtype=np.float64), np.array(val_epoch_losses, dtype=np.float64)))

    return time_points, train_epoch_losses, val_epoch_losses

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse hyperparameters for the script.")

    # Add arguments
    parser.add_argument("--N_t", type=int, default=64, help="Number of transmit antennas")
    parser.add_argument("--N_r", type=int, default=4, help="Number of receive antennas")
    parser.add_argument("--d", type=int, default=2, help="Number of data streams")
    parser.add_argument("--user_number", type=int, default=6, help="Number of users")
    parser.add_argument("--BS_number", type=int, default=1, help="Number of base stations")
    parser.add_argument("--signal_max_power", type=float, default=1e2, help="Maximum signal power")
    parser.add_argument("--noise_power", type=float, default=1e-8, help="Noise power")
    parser.add_argument("--N_samples", type=int, default=50000, help="Number of samples for training")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training")
    parser.add_argument("--supervisor_epoch_num", type=float, default=0.05, help="Number of epochs for supervised learning")
    parser.add_argument("--supervisor_lr", type=float, default=1.38e-3, help="Learning rate for supervised learning")
    parser.add_argument("--unsupervisor_lr", type=float, default=1.38e-5, help="Learning rate for unsupervised learning")
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for training")
    parser.add_argument("--net_iterations", type=int, default=8, help="Number of iterations for the network")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden layer size of the network")
    parser.add_argument("--single_number", type=int, default=5000, help="Single number parameter")
    parser.add_argument("--record_interval", type=int, default=120)
    parser.add_argument("--exp_name", type=str, default='')
    
    parser.add_argument("--w", nargs='+', type=float, default=None, help="Weight array")

    args = parser.parse_args()

    # Set up weights
    if args.w is None:
        w = [1 for i in range(args.user_number)]
        w = [w[i] / sum(w) for i in range(args.user_number)]
    else:
        w = args.w

    # Extract arguments
    N_t = args.N_t
    N_r = args.N_r
    d = args.d
    user_number = args.user_number
    BS_number = args.BS_number
    signal_max_power = args.signal_max_power
    noise_power = args.noise_power
    N_samples = args.N_samples
    epochs = args.epochs
    supervisor_epoch_num = args.supervisor_epoch_num
    supervisor_lr = args.supervisor_lr
    unsupervisor_lr = args.unsupervisor_lr
    batch_size = args.batch_size
    net_iterations = args.net_iterations
    hidden_size = args.hidden_size
    single_number = args.single_number
    record_interval = args.record_interval
    exp_name = args.exp_name

    # Define learning rate schedule
    def lr_lambda(epoch):
        if epoch < 100:
            return 1.0  # Maintain the initial learning rate
        else:
            return unsupervisor_lr / supervisor_lr  # Adjust to the target learning rate.

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using: ", device)

    # Load and prepare data
    Data_blocks = N_samples // single_number
    H = []
    for i in range(Data_blocks):
        if BS_number > 1:
            H.append(load_H(f"Data/data/{i+1}_{single_number}_Nr_{N_r}_Nt_{N_t}_N_user_{user_number}_BS_{BS_number}.mat"))
        else:
            H.append(load_H_BS1(f"Data/data/{i+1}_{single_number}_Nr_{N_r}_Nt_{N_t}_N_user_{user_number}_BS_{BS_number}.mat"))
    H = torch.cat(H, dim=0)
    
    # Split the dataset
    train_set, val_set, test_set = split_matrix(H=H, train_size=0.8)

    # Prepare DataLoaders
    trainset = dataset(train_set)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)

    valset = dataset(val_set[:500])
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size)
    dataloaders = [trainloader, valloader]

    # Initialize network, optimizer, scheduler, and loss function
    net = FastFPnet(BS_number=BS_number, user_number=user_number, N_t=N_t, N_r=N_r, w=w, d=d, iterations=net_iterations, device=device, hidden_size=hidden_size, signal_max_power=signal_max_power, noise_power=noise_power)
    optimizer = optim.Adam(net.parameters(), lr=supervisor_lr, weight_decay=0)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_fuc = LossUnfolding(signal_max_power=signal_max_power, noise_power=noise_power)

    # Training
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    supervisor_batchs = int(supervisor_epoch_num * train_set.size(0) / batch_size)
    record_time, train_loss, val_loss = train_FPnet(net=net, epochs=epochs, loss_fc=loss_fuc, optimizer=optimizer, lr_scheduler=scheduler, dataloaders=dataloaders, supervisor_batchs=supervisor_batchs, timestamp=timestamp, record_interval=record_interval, exp_name=exp_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The net took {elapsed_time} seconds to train.")

    # Save model and losses
    torch.save(net.state_dict(), f'models\{exp_name}_model_weights_{timestamp}.pth')
    np.save(f'Loss\{exp_name}_Time_Tran_Val_loss_{timestamp}.npy', (np.array(record_time, dtype=np.float64), np.array(train_loss, dtype=np.float64), np.array(val_loss, dtype=np.float64)))

if __name__ == "__main__":
    main()