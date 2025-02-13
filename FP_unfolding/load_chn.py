import torch
import h5py


def load_H(filename,chnname='chn'):
    matfile = h5py.File(filename, 'r')
    H=matfile[chnname][:]
    real,imag=torch.tensor(H['real'], dtype=torch.float64),torch.tensor(H['imag'], dtype=torch.float64)
    H=torch.complex(real, imag)
    H=H.permute(5,0,1,2,4,3)

    return H


def load_H_BS1(filename):
    matfile = h5py.File(filename, 'r')
    H=matfile['chn'][:]
    real,imag=torch.tensor(H['real'], dtype=torch.float64),torch.tensor(H['imag'], dtype=torch.float64)
    H=torch.complex(real, imag)
    H=H.permute(3,0,2,1).unsqueeze(1).unsqueeze(1)

    return H



def load_H_3GPP(filename):
    matfile = h5py.File(filename, 'r')
    H=matfile['H'][:]
    real,imag=torch.tensor(H['real'], dtype=torch.float64),torch.tensor(H['imag'], dtype=torch.float64)
    H=torch.complex(real, imag)
    H=H.permute(0,1,3,2).unsqueeze(1).unsqueeze(1)

    return H



