import torch
from torch import nn
from torch import optim
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from model import mynet
from dataset import data_set

train_dir = './train/'
test_dir = './test/'
train_data=data_set(train_dir)
test_data=data_set(test_dir)
#print(train_data[5][0]!=1)


model=mynet()
batch_size = 128
epochs = 256
learning_rate = 5e-4
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# set device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
min_acc=0
def test(model):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    loss_fn.to(device)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x,y in test_dataloader:
            x=x.to(device)
            y=y.to(device)
            pred = model(x)
            test_loss+= np.mean(loss_fn(pred, y).item())
            correct+= (pred.argmax(1) == y.argmax(1)).type(torch.float).mean().item()
    test_loss/=len(test_dataloader)
    correct/=len(test_dataloader)
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Test Avg loss: {test_loss:>8f} ")
    model.train()
    return correct
def auto_save(model,epoch):
    global min_acc
    acc=test(model)
    if acc>min_acc:
        min_acc=acc
        torch.save(model.state_dict(), str(epoch)+"resnet.pt")
        print("saved")
    
model = model.to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loss_history = []
test_loss_history = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct =0.0
    for x,y in train_dataloader:
        x=x.to(device)
        y=y.to(device)
        # Compute prediction and loss
        pred = model(x)
        #print(pred.shape,y.shape)
        loss = loss_fn(pred,y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).mean().item()
        running_loss+= loss.item()
        
    running_loss /= len(train_dataloader)
    correct /= len(train_dataloader)
    print(f" Epoch:{epoch+1}, loss: {running_loss:>8f},  Accuracy: {(100*correct):>0.1f}%")
    auto_save(model,epoch)

