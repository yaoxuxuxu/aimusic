import ctypes
import librosa
import numpy as np
import numpy as np
import librosa
import librosa.display
import os
from scipy import ndimage

class Shazam:
    def __init__(self):
        self.shazam=self.dll_import("./shazam.dll")
        self.music_base_dir="./songs"

        self.music_list=[]
        self.dist_freq = 11
        self.dist_time = 7
    def dll_import(self,dir):
        return ctypes.CDLL(dir,winmode=0)
    
    def insert(self,path):
        self.music_list.append(path)
        print(path)
        cm=self.get_con_map(path)
        h,t=cm.shape
        cm=cm.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        self.shazam.insert(cm,h,t,len(self.music_list)-1)
    def query(self,path):
        cm=self.get_con_map(path)
        h,t=cm.shape
        cm=cm.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))
        print(path)
        top=self.shazam.query(cm,h,t)
        print(self.music_list[top])
        
    def create_database(self,dir_list):
        for i in dir_list:
            self.insert(i)
        return
    def compute_spectrogram(self,audio_path, Fs=22050, N=2048, H=1024, bin_max=128, frame_max=None):
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
        CM = self.compute_constellation_map(Y, self.dist_freq, self.dist_time)
        return CM

if __name__ == "__main__":
    shazam=Shazam()
    musiclist=[]
    maindir="../mp3s/"
    for i in os.listdir(maindir):
        tmp=os.path.join(maindir,i)
        for j in os.listdir(tmp):
            if ".mp3" in j:
                musiclist.append(os.path.join(tmp,j))
    shazam.create_database(musiclist)
    testdir=[]
    dir="./test_final"
    for fp in os.listdir(dir):
        if ".mp3" not in fp:
            continue
        qdir=os.path.join(dir,fp)
        testdir.append(qdir)
    testdir=sorted(testdir)
    print(testdir)
    for i in testdir:
        shazam.query(i)
    while 1:
        s=input()
        if s=="query":
            for i in testdir:
                shazam.query(i)
    #shazam.insert("./songs/NationalAnthemIndia.wav")
    