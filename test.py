import os
import librosa
import soundfile as sf
inputdir="./mp3s/"
testdir="./test/"
outdir="./sdofijw/"
musiclist=[]
maindir="./mp3s/"
for i in os.listdir(maindir):
    tmp=os.path.join(maindir,i)
    for j in os.listdir(tmp):
        if ".mp3" in j:
            musiclist.append(os.path.join(tmp,j))
print(musiclist)
cnt=0
for fp in musiclist:
    y, sr = librosa.load(fp, sr=22050)
    os.makedirs(outdir,exist_ok=True)
    sf.write(os.path.join(outdir,str(cnt)+".wav"), y, sr)
    cnt+=1
