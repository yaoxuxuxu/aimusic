import librosa
import os
import soundfile as sf
import random

class seperate:
    def __init__(self) -> None:
        self.inputdir="../mp3s/"
        self.testdir="./test/"
        self.outdir="./test/"
        self.target_sr = 22050 
        self.duration=6
    def main(self):
        musiclist=[]
        maindir="../mp3s/"
        for i in os.listdir(maindir):
            tmp=os.path.join(maindir,i)
            for j in os.listdir(tmp):
                if ".mp3" in j:
                    musiclist.append(os.path.join(tmp,j))
        for fp in musiclist:
            y, sr = librosa.load(fp, sr=self.target_sr)
            os.makedirs(self.outdir,exist_ok=False)
            self.cut_save(y,sr,self.outdir)
            break
        
    def cut_save(self,y,sr,fp):
        iter=self.duration*sr
        cnt=0
        for i in range(0,len(y),iter):
            if i+iter>len(y):
                """end=len(y)-1
                begin=end-(iter)
                tmp=y[begin:end]"""
                break
            else:
                tmp=y[i:i+iter]
            sf.write(os.path.join(fp,str(cnt)+".wav"), tmp, self.target_sr)
            cnt+=1
    def random_cut(self,y,sr,fp):
        iter=self.duration*sr
        cnt=0
        for i in range(10):
            start=random.randint(0,len(y)-iter-1)
            end=start+iter
            tmp=y[start:end]
            sf.write(os.path.join(fp,str(cnt)+".wav"), tmp, self.target_sr)
            cnt+=1
        return
    def data_improvement(self,y,sr,fp):
        pass
    def generate_test(self):
        for fp in os.listdir(self.inputdir):
            y, sr = librosa.load(os.path.join(self.inputdir,fp), sr=self.target_sr)
            newdir=os.path.join(self.testdir,fp.replace(".wav",""))
            print(newdir)
            os.makedirs(newdir,exist_ok=False)
            self.random_cut(y,sr,newdir)

if __name__ == "__main__":
    sp=seperate()
    sp.main()

        