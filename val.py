import torch
model = torch.load("mm.pt")
input=[]
t_input=[]
label=[]
tx_x=[]
tx_y=[]
loss_y=[]
model.eval()
with open("feature.txt","r")as fd:
    input=eval(str(fd.read()).replace("tensor","torch.tensor"))
fd.close()
with open("label.txt","r")as fd:
    label=eval(str(fd.read()).replace("'",""))
fd.close()
for tmp in input:
    t_input.append(tmp.tolist())
input=torch.tensor(t_input)
label=torch.tensor(label)
print(input.size())
print(label.size())
train = torch.utils.data.TensorDataset(input,label)
dataloader_val = torch.utils.data.DataLoader(train, batch_size=16,shuffle=True)
running_corrects = 0
len_dataset = 0
for step, (t_x, t_y) in enumerate(dataloader_val):
    try:
        #print(t_x.size())
        #print(t_y.size())
        out=model(t_x)
        _, preds = torch.max(out, 1)
        #print(t_y)
        #print(preds)
        print(preds)
        print(t_y)
        running_corrects += torch.sum(preds == t_y.data)
        len_dataset += len(t_y.data)
    except:
        pass
print("suc:"+str(running_corrects/len_dataset))