import numpy as np
import torch
import os
import pickle
from PIL import Image
import cv2

import matplotlib.pyplot as plt
plt.switch_backend('agg')




def getFileNames(txt): 
    '''
    description: parse the xxx.txt (image_path+' '+label) file::
    input: path of the .txt file\n
    return: a list of DB image pathes
    '''
    fileDB = []
    with open(txt, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip().split(' ')
            fileDB.append(item[0])
    return fileDB






def saveimgs(act, dir='featImgs', epoch=0):
    '''
    description: save batched feature maps (GPU->CPU) into one figure using PIL Image\n
    act: activation values obtained from the network layer\n
    dir: output folder (will be automatically created)\n
    epoch: the epoch ID\n
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)    
    
    # when input more than one layer
    # for id, act in enumerate(acts):

    act = act.detach().cpu().numpy()

    b, c, h, w = act.shape
    stp = max(w//20, 1)

    imgs = np.ones((b*h+(b+1)*stp, c*w+(c+1)*stp), dtype=np.uint8)*255       

    # batch and channels
    for bid in range(b):
        for cid in range(c):
            img = act[bid, cid, :, :]
            img = (img - img.min())/(img.max()-img.min()+1e-10)*255
                       
            srow = stp*(bid+1)
            scol = stp*(cid+1)
            imgs[h*bid + srow : h*(bid+1) + srow, w*cid + scol : w*(cid+1)+ scol] = img.astype(np.uint8)


    im = Image.fromarray(imgs.astype("uint8")).convert('L')
    
    im.save(os.path.join(dir, 'layer'+('_ep%04d'%epoch)+'.png'))
    # im.save(os.path.join(dir, 'layer'+str(id)+('_ep%04d'%epoch)+'.png'))
    # cv2.imwrite(os.path.join(dir, 'layer'+str(id)+('_ep%04d'%epoch)+'.png'), imgs)
    # img = np.array(img, dtype=np.uint8) 




def saveimgs2(act, dir='featImgs', epoch=0):
    '''
    description: save batched feature maps (GPU->CPU) into one figure using plt.figure() subplots\n
    act: activation values obtained from the network layer\n
    dir: output folder (will be automatically created)\n
    epoch: the epoch ID\n
    '''
    if not os.path.exists(dir):
        os.makedirs(dir)   
     
    act = act.detach().cpu().numpy()

    b, c, h, w = act.shape

    fig = plt.figure(figsize=(100, int(100.*b/c)))#
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)

    for bid in range(b): # batch id
        for cid in range(c): # channel id
            ax = fig.add_subplot(b, c, bid*c+cid+1, xticks=[], yticks=[])
            ax.imshow(act[bid, cid, :, :])
    plt.savefig(os.path.join(dir, 'layer'+('_ep%04d'%epoch)+'.png'))





def saveimgs3(netOut, name='feat', featflag=False):
    '''
    save feature maps via global normalization
    '''
    imgs = netOut.data
    
    imgs = torch.cat(torch.split(imgs, 1, dim=1), dim=3)
    imgs = torch.cat(torch.split(imgs, 1, dim=0), dim=2)
    imgs = torch.squeeze(imgs)
    imgs = imgs.numpy()
    
    imgs = (imgs-imgs.min())/(imgs.max()-imgs.min()+1e-8)*255    
    cv2.imwrite(name+'.bmp', imgs.astype(np.uint8))
    # imgs = Image.fromarray(imgs.numpy())
    # imgs.save(name+'.jpg')



class RegLayers():
    features = []
    def __init__(self, net):
        self.hooks = []
        
        self.hooks.append(net.layer1.register_forward_hook(self.hook_fn))
        self.hooks.append(net.layer2.register_forward_hook(self.hook_fn))
        self.hooks.append(net.layer3.register_forward_hook(self.hook_fn))     
        self.hooks.append(net.layer4.register_forward_hook(self.hook_fn)) 
        self.hooks.append(net.layer5.register_forward_hook(self.hook_fn))    
       

    def hook_fn(self, model, input, output):
        self.features.append(output)

    def remove(self):
        for hook in self.hooks:
            hook.remove()


def extract_layers(model, input):

    la = RegLayers(model)

    la.features = []
    
    model.eval()

    o = model(input)

    la.remove()

    acts = la.features
    return acts

   


class RegLayers_pvf():
    features = []
    def __init__(self, net):
        self.hooks = []
        
        self.hooks.append(net.layer1.register_forward_hook(self.hook_fn))
        self.hooks.append(net.layer2.register_forward_hook(self.hook_fn))
        self.hooks.append(net.layer3.register_forward_hook(self.hook_fn))     
        self.hooks.append(net.layer4.register_forward_hook(self.hook_fn)) 
        self.hooks.append(net.layer5.register_forward_hook(self.hook_fn))    
       

    def hook_fn(self, model, input, output):
        self.features.append(output)

    def remove(self):
        for hook in self.hooks:
            hook.remove()


def extract_layers_pvf(model, input):

    la = RegLayers_pvf(model)

    la.features = []
    
    model.eval()

    o = model(input)

    la.remove()

    acts = la.features
    return acts









def saveFeatureMaps(net, data_loader, epoch):
    data, _ = iter(data_loader).next()
    device = net.fc1.weight.device

    saveimgs(data, dir='./rst/images/00_input_Img',epoch=epoch)

    acts = extract_layers(net, data.to(device))
    saveimgs(acts[0], dir='./rst/images/01_layer1',epoch=epoch)
    saveimgs(acts[1], dir='./rst/images/02_layer2',epoch=epoch)
    saveimgs(acts[2], dir='./rst/images/03_layer3',epoch=epoch)
    saveimgs(acts[3], dir='./rst/images/04_layer4',epoch=epoch)
    saveimgs(acts[4], dir='./rst/images/05_layer5',epoch=epoch)
  
   

def saveConvFilters(net, epoch):
    '''
    save the learned Gabor filters of LGC
    '''
    if not os.path.exists('./rst/images/convfilters'): 
        os.makedirs('./rst/images/convfilters')

    kernel1 = net.layer1.conv.weight  

    kernel = kernel1.detach().cpu().numpy()

    channel_in = kernel.shape[1]
    channel_out = kernel.shape[0]

    for o in range(channel_out):
        for i in range(channel_in):
            ws = kernel[o, i, :, :]
            # plt.matshow(ws, cmap='gist_gray')
            # plt.savefig('convfilters/%03d.png'%o )
            # plt.show()
            img = ws
            img = (img - img.min())/(img.max()-img.min()+1e-10)*255 
            im = Image.fromarray(img.astype("uint8")).convert('L')
            im.save('./rst/images/convfilters/%d_layer1_%03d.png'%(epoch, o))

    







def plotLossACC(train_losses, val_losses, train_accuracy, val_accuracy):
    path_rst = './rst'
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b', label='training loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='test loss')
    plt.legend()
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.savefig(os.path.join(path_rst, 'losses.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'b', label='training accuracy')
    plt.plot(range(1, len(val_accuracy)+1), val_accuracy, 'r', label='test accuracy')
    plt.legend()
    plt.grid()
    plt.xlabel('epoch number')
    plt.ylabel('accuracy (%)')
    plt.savefig(os.path.join(path_rst, 'accuracy.png'))
    plt.close()


def saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc):
    path_rst = './rst'
    if not os.path.exists(path_rst):
        os.makedirs(path_rst)

    # save as pickle
    with open(os.path.join(path_rst,'train_losses.pickle'), 'wb') as f:
        pickle.dump(train_losses, f)
    with open(os.path.join(path_rst,'val_losses.pickle'), 'wb') as f:
        pickle.dump(val_losses, f)
    with open(os.path.join(path_rst,'train_accuracy.pickle'), 'wb') as f:
        pickle.dump(train_accuracy, f)
    with open(os.path.join(path_rst,'val_accuracy.pickle'), 'wb') as f:
        pickle.dump(val_accuracy, f)     


    # save as txt
    with open(os.path.join(path_rst,'train_losses.txt'), 'w') as f:
        for v in train_losses:
            f.write(str(v)+'\n')
    with open(os.path.join(path_rst,'train_accuracy.txt'), 'w') as f:
        for v in train_accuracy:
            f.write(str(v)+'\n')
    with open(os.path.join(path_rst,'val_losses.txt'), 'w') as f:
        for v in val_losses:
            f.write(str(v)+'\n')
    with open(os.path.join(path_rst,'val_accuracy.txt'), 'w') as f:
        for v in val_accuracy:
            f.write(str(v)+'\n')

    with open(os.path.join(path_rst,'best_val_accuracy.txt'), 'w') as f:
        f.write(str(bestacc))

