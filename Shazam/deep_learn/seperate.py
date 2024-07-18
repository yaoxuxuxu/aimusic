import librosa
import os
import soundfile as sf
import random

class seperate:
    def __init__(self) -> None:
        self.inputdir="./original_data/"
        self.testdir="./test/"
        self.outdir="./train/"
        self.target_sr = 22050 
        self.duration=6
    def main(self):
        
        for fp in os.listdir(self.inputdir):
            y, sr = librosa.load(os.path.join(self.inputdir,fp), sr=self.target_sr)
            newdir=os.path.join(self.outdir,fp.replace(".wav",""))
            print(newdir)
            os.makedirs(newdir,exist_ok=False)
            self.cut_save(y,sr,newdir)

        
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
    sp.generate_test()

        