import torch
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from scipy import ndimage

class data_set(Dataset):
    def __init__(self,data_dir):
        self.musicdir=[]
        self.lb=[]
        self.classes=-1
        for i in os.listdir(data_dir):
            tmpdir=os.path.join(data_dir,i)
            self.classes+=1
            for j in os.listdir(tmpdir):
                if ".wav" not in j:
                    continue
                self.lb.append(self.classes)
                self.musicdir.append(os.path.join(tmpdir,j))

    def __len__(self):
        return len(self.musicdir)
    def __getitem__(self,idx):
        cm=self.get_con_map(self.musicdir[idx])
        """img=np.array(img,dtype=np.float32)
        img/=255.
        img.resize((1,128,128))"""
        lb=self.lb[idx]
        lb=np.eye(self.classes+1)[lb]
        cm=np.array([cm])
        img=torch.tensor(cm,dtype=torch.float32)
        lb=torch.tensor(lb,dtype=torch.float32)
        return img,lb
    def compute_spectrogram(self,audio_path, Fs=22050, N=2048, H=1024, bin_max=130, frame_max=None):
        x, Fs = librosa.load(audio_path, sr=Fs)
        x_duration = len(x) / Fs
        X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann')
        if bin_max is None:
            bin_max = X.shape[0]
        if frame_max is None:
            frame_max = X.shape[0]
        Y = np.abs(X[:bin_max, :frame_max])
        return Y

    def compute_constellation_map(self,Y, dist_freq=7, dist_time=7, thresh=0.01):
        result = ndimage.maximum_filter(Y, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
        Cmap = np.logical_and(Y == result, result > thresh)
        return Cmap
    def get_con_map(self,path):
        Y = self.compute_spectrogram(path)
        #CM = self.compute_constellation_map(Y=Y)
        return Y
train_data=data_set("./train")
#test_data=data_set("./train",train=0)