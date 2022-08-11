import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import models


import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
import time

import numpy as np
from PIL import Image
import cv2


from models import MyDataset
from models import ppnet
from utils import *

python_path = '/home/sunny/local/anaconda3/envs/torch37/bin/python'







print('%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



training_set = './data/train.txt'
trainset =MyDataset(txt=training_set, transforms=None, train=True, imside=128, outchannels=1)


test_set = './data/test.txt'
testset =MyDataset(txt=test_set, transforms=None, train=False, imside=128, outchannels=1)


batch_size = 8

assert batch_size % 2 == 0

data_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

data_loader_show = DataLoader(dataset=trainset, batch_size=8, shuffle=True)








net = ppnet(num_classes=600) # IITD: 460    KTU: 145    Tongji: 600     REST: 358   DCPD:271    XJTU: 200
# net.load_state_dict(torch.load('net_params.pkl'))

print(net)




criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.0001)#0.0003

scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)



def contrastive_loss(target, dis, margin=5.0):

    n = len(target)//2

    y1 = target[:n]
    y2 = target[n:]

    y = np.zeros((1, n), dtype=np.float)
    y = y.squeeze()
    y = torch.Tensor(y)    
    

    margin = torch.Tensor(np.array([margin]*n))

    y = y.to(device)
    margin = margin.to(device)

    y[y1==y2] = 1
    
    contra_loss = torch.mean((y) * torch.pow(dis, 2) +  (1-y) * torch.pow(torch.clamp(margin - dis, min=0.0), 2))     
    return contra_loss
 


 
def fit(epoch, model, data_loader, phase='training', volatile=False):
    
    if phase != 'training' and phase != 'validation':
        raise TypeError('input error!')

    if phase == 'training':
        model.train()

    if phase == 'validation':
        model.eval()
       
    

    running_loss = 0
    running_correct = 0

    for batch_id, (data, target) in enumerate(data_loader):

        if len(target) % 2 !=0:            
            target = torch.cat((target, torch.LongTensor((target[0],))),dim=0) 
            data = torch.cat((data, data[0,:,:,:].unsqueeze(0)), dim=0)
        
        data = data.to(device)
        target = target.to(device)


        if phase == 'training':
             optimizer.zero_grad()

        output, dis = model(data)
 

        cross = criterion(output, target)
 
        l2 = torch.norm(model.fc2.weight, 2) + torch.norm(model.fc3.weight, 2)

        contra = contrastive_loss(target, dis, margin=5.0)     


        loss = cross + 1e-4*l2 + 2*1e-4*contra + 1e-4*torch.mean(torch.pow(dis, 2))


        ##### log
        running_loss += loss.data.cpu().numpy() # item()

        preds = output.data.max(dim=1, keepdim=True)[1] # (max_value, max_index)
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()
        
        
        ##### update
        if phase == 'training':
            loss.backward()
            optimizer.step()
           

    ### log
    num_imgs = len(data_loader.dataset)

    if num_imgs % 2 != 0:
        num_imgs += 1

    loss = running_loss / num_imgs
    accuracy = (100.0 * running_correct) / num_imgs

    if epoch % 10 == 0:
        print('epoch %d: \t%s loss is \t%7.5f    and %s \taccuracy is \t%d/%d \t%7.3f%%'%(epoch, phase, loss, phase, running_correct, num_imgs, accuracy))
        
    return loss, accuracy
    




net.to(device)

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []

bestacc = 0

for epoch in range(3000):

    epoch_loss, epoch_accuracy = fit(epoch, net, data_loader, phase='training')

    val_epoch_loss, val_epoch_accuracy = fit(epoch, net, data_loader_test, phase='validation')

    scheduler.step()

    #------------------------logs----------------------
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    # save the best model
    if val_epoch_accuracy > bestacc:
        bestacc= val_epoch_accuracy
        torch.save(net.state_dict(), 'net_params_best.pth') 

    # save the current model and log info:
    if epoch % 10 == 0 or epoch == 2999 and epoch != 0:
        torch.save(net.state_dict(), 'net_params.pth') 

        plotLossACC(train_losses, val_losses, train_accuracy, val_accuracy)
        saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc)   
            

    # visualization
    if epoch % 200 == 0 or epoch == 2999: 
        saveFeatureMaps(net, data_loader_show, epoch)
        saveConvFilters(net, epoch)




# finished training
# torch.save(net.state_dict(), 'net_params.pth')
# torch.save(net, 'net.pkl') 


print('Finished Trainning')
print('the best testing acc is: ', bestacc, '%')
print('%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))




print('\n\n=======')
print('testing ...')
os.system(python_path+' test.py')


print('%s'%(time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
