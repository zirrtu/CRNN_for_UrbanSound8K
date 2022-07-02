import torch
from torchvision import models
import numpy as np
import random
from CRNNmodel import CRNN
import torchvision.transforms as transforms
from helpers import get_device, one_hot_embedding
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
input=[]
t_input=[]
label=[]
tx_x=[]
tx_y=[]
va_y=[]
loss_y=[]
eval_input=[]
eval_label=[]
test_fold=1
train_num=2
BS = 64#batch_size
for t_f in range(1,train_num):
    test_fold = str(t_f)
    with open("feature"+str(test_fold)+".txt","r")as fd:
        tmp_input=eval(str(fd.read()).replace("tensor","torch.tensor"))
        for tmp_tensor in tmp_input:
            input.append(tmp_tensor)
        print(tmp_input[0].size())
    fd.close()
print(len(input))
for t_f in range(1,train_num):
    test_fold = str(t_f)
    with open("label"+str(test_fold)+".txt","r")as fd:
        tmp_label=eval(str(fd.read()).replace("'",""))
        print("qqq:"+str(tmp_label))
        for tmp_int in tmp_label:
            label.append(tmp_int)
        #print(tmp_input[0].size())
    fd.close()

cc = list(zip(input,label))
random.shuffle(cc)
input[:],label[:]=zip(*cc)
train_fg=len(input)*19//20
eval_input=input[train_fg+1:len(input)]
eval_label=label[train_fg+1:len(input)]
input=input[0:train_fg]
label=label[0:train_fg]

for tmp in eval_input:
    t_input.append(tmp.tolist())
eval_input=torch.tensor(t_input)
eval_label=torch.tensor(eval_label)
print(eval_input.size())
print(eval_label.size())
eval_input = (eval_input-torch.mean(eval_input))/torch.std(eval_input)
eval = torch.utils.data.TensorDataset(eval_input,eval_label)
dataloader_eval = torch.utils.data.DataLoader(eval, batch_size=BS,shuffle=True)
t_input.clear()
for tmp in input:
    t_input.append(tmp.tolist())
input=torch.tensor(t_input)
label=torch.tensor(label)
print(input.size())
print(label.size())
input=(input-torch.mean(input))/torch.std(input)
train = torch.utils.data.TensorDataset(input,label)
dataloader_val = torch.utils.data.DataLoader(train, batch_size=BS,shuffle=True)
model = CRNN(num_classes=10)
model = model.cuda()
#
ansnet = model
mark_suc=0
#loss_func = edl_digamma_loss
loss_func =nn.CrossEntropyLoss()
learning_rate=0.001
opt = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=0.0005)
scheduler = MultiStepLR(opt, milestones=[50, 80], gamma=0.1)
al_epoch=100
al_sum = len(dataloader_val) * al_epoch
cnt = 0
device = get_device()
num_classes = 10
for epoch in range(al_epoch):
    print("traing...")
    #model.train()
    sum_loss=0
    running_corrects = 0.0000001
    len_dataset = 0.000001
    running_loss = 0.000001
    for step, (t_x, t_y) in enumerate(dataloader_val):
        cnt += 1
        # print("ty:")
        # print(t_y.size())
        # #try:
        print("tx:")
        print(t_x.size())
        # print(num_classes)

        t_x = t_x.cuda()
        t_y = t_y.cuda()
        opt.zero_grad()
        outputs = model(t_x)
        loss = loss_func(outputs,t_y)

        _, preds = torch.max(outputs, 1)
        # y = one_hot_embedding(t_y, num_classes)
        # outputs = model(t_x)
        # print(outputs.size())
        # _, preds = torch.max(outputs, 1)
        # loss = loss_func(
        #     outputs, y.float(), epoch, num_classes, 10, device
        # )
        # match = torch.reshape(torch.eq(preds, t_y).float(), (-1, 1))
        # acc = torch.mean(match)
        # evidence = relu_evidence(outputs)
        # alpha = evidence + 1
        # u = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        #
        # total_evidence = torch.sum(evidence, 1, keepdim=True)
        # mean_evidence = torch.mean(total_evidence)
        # mean_evidence_succ = torch.sum(
        #     torch.sum(evidence, 1, keepdim=True) * match
        # ) / torch.sum(match + 1e-20)
        # mean_evidence_fail = torch.sum(
        #     torch.sum(evidence, 1, keepdim=True) * (1 - match)
        # ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

        loss.backward()
        opt.step()
        scheduler.step()
        print(str(cnt) + "/" + str(al_sum))
        running_corrects += torch.sum(preds == t_y.data)
        running_loss += loss.item()
        len_dataset += len(t_y.data)
        #except:
        #    pass
    tx_y.append(running_corrects/len_dataset)
    tx_x.append(epoch)
    loss_y.append(running_loss/len_dataset)
    print("suc:"+str(running_corrects/len_dataset))
    print("loss:"+str(running_loss/len_dataset))
    if running_corrects/len_dataset > mark_suc:
        mark_suc = running_corrects/len_dataset
        ansnet = model
    #model.eval()
    running_corrects = 0.0000001
    len_dataset = 0.000001
    running_loss = 0.000001
    for step, (t_x, t_y) in enumerate(dataloader_eval):
        try:
            t_x = t_x.cuda()
            t_y = t_y.cuda()
            out=model(t_x)
            _, preds = torch.max(out, 1)
            running_corrects += torch.sum(preds == t_y.data)
            len_dataset += len(t_y.data)
        except:
            pass
    print(running_corrects)
    va_y.append(running_corrects/len_dataset)
    print("val_suc:"+str(running_corrects/len_dataset))
    print("val_loss:"+str(running_loss/len_dataset))
ty_y=[]
ty_yy=[]
print(tx_x)
print(tx_y)
for tmp in tx_y:
    ty_y.append(tmp.cpu())
for tmp in va_y:
    ty_yy.append(tmp.cpu())
print(ty_y)
torch.save(ansnet,"mm.pt")
print("best epoch suc:"+str(mark_suc))
plt.plot(tx_x,ty_y,label='success rate')
plt.plot(tx_x,ty_yy)
plt.xlabel('epoch')
plt.show()