import faiss
import numpy as np
from model import mynet
from dataset import data_set
from torch.utils.data import DataLoader
import torch
import os
class Database:
    def __init__(self) -> None:
        self.feature_len=2048

        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.model=mynet().to(self.device)
        self.model.load_state_dict(torch.load("./15resnet.pt"))
        self.index=np.array([])
        self.musiclist=[]
        self.musiclb=[]

        self.database = faiss.IndexFlatL2(self.feature_len) 
    def create_database(self,music_list):
        test_data=data_set(music_list)
        self.musiclb=test_data.lb
        self.musiclist=test_data.musicdir
        feature_list=np.zeros((1,self.feature_len))
        test_dataloader=DataLoader(test_data,32,False)
        with torch.no_grad():
            for x,y in test_dataloader:
                x=x.to(self.device)
                pred = self.model(x)
                pred=pred.to("cpu")
                pred.numpy()
                #print(pred.shape)
                feature_list=np.append(feature_list,pred,axis=0)
                self.index=np.append(self.index,y.argmax(axis=1).numpy())
        self.database.add(feature_list[1:])
        return
    def query(self,music_list,top_k=3):
        test_data=data_set(music_list)
        music=test_data.musicdir
        test_dataloader=DataLoader(test_data,32,False)
        feature_list=np.zeros((1,self.feature_len))
        with torch.no_grad():
            for x,y in test_dataloader:
                x=x.to(self.device)
                pred = self.model(x)
                pred=pred.to("cpu")
                pred.numpy()
                feature_list=np.append(feature_list,pred,axis=0)
        feature_list=feature_list[1:]
        cnt=0
        for i in feature_list:
            tmp=np.array([i])
            dis ,indices = self.database.search(tmp,top_k)
            for j in indices[0]:
                
                print(music[cnt])
                print(self.__indice_to_music_name__(indices[0][0]))
            cnt+=1
        return
    def __indice_to_music_name__(self,indice):
        index=int(self.index[indice])
        return self.musiclist[self.musiclb[index]]
if __name__ == "__main__":
    database=Database()
    database.create_database("./train")
    maindir="./"
    dir="./test"
    database.query(dir)





