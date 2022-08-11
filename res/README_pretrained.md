[![pvfnet](https://img.shields.io/badge/Backto-PPNet-green)](https://github.com/xuliangcs/ppnet/blob/main/README.md#4-pytorch-implementation)

## Using pretrained models

### Different Number of Classes

Using pretrained models (on Tongji) for PPNet/PVFNet:

```python
net = ppnet(num_classes=xxx)
print(net)
model_dict = net.state_dict()

pretrained_dict = torch.load('net_params_tongji.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and k != 'fc3.bias' and k !='fc3.weight'} 

model_dict.update(pretrained_dict)

net.load_state_dict(model_dict)
# net.load_state_dict(torch.load('net_params.pkl'))
```

Using pretrained models for resnet *etc.*:

```python
from torchvision import models

net = models.resnet18(pretrained=False, num_classes=xxx)
print(net)
model_dict = net.state_dict()

tmpnet = models.resnet18(pretrained=True)
pretrained_dict = tmpnet.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and k != 'fc.bias' and k !='fc.weight'} 

model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
```

### Same Number of Classes

```python
net = ppnet(num_classes=600)
net.load_state_dict(torch.load('net_params.pkl'))
```



Print Parameters:

```python
for param in net.named_parameters():
    print(param)

for param in net.fc1.named_parameters():
	print(param) 
```

