import numpy as np
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
import torch
import csv
from torch import nn
import torch.nn.functional as F
loss_tim=[]
got_class=[0,1,2,3,4,5,6,7,8,9]
torch.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=20)
#metadata路径
csv_path = "UrbanSound8K\metadata\\UrbanSound8K.csv"

data = []
label = []
test_fold = "1"
for t_f in range(1, 11):
    with open(csv_path) as csvfile:
        check=0
        test_fold = str(t_f)
        csv_reader = csv.reader(csvfile)
        tmp_in_data=[]
        label = []
        for row in csv_reader:
            if row[5] == test_fold and int(row[6]) in got_class:
                check+=1
                #数据集路径
                data_path="UrbanSound8K\\audio"
                data_path+='\\fold'+str(test_fold)+"\\"
                now_data=data_path+str(row[0])
                print(now_data)
                wav, sr = librosa.load(now_data, sr=16000)
                print("采样率："+str(sr))
                # 分帧，做傅里叶变换
                tmp_www =librosa.feature.melspectrogram(y = wav, sr = sr,n_mels = 128)
                torch_data = torch.from_numpy(tmp_www)
                torch_data = torch_data.t()
                print("qwe")
                print(torch_data.size())
                for i in range(len(torch_data)//128):
                    tmp_in_data.append(torch_data[i:i + 128].reshape(1,128,128))
                    label.append(row[6])
        print("over!")
        for tmp in tmp_in_data:
            print(tmp.size())
        with open("feature"+str(test_fold)+".txt","w")as fd:
            fd.write(str(tmp_in_data))
        fd.close()
        with open("label"+str(test_fold)+".txt","w")as fd:
            fd.write(str(label))
        fd.close()
#0 文件名 5 fold 6 class id
print("特征提取完成")
