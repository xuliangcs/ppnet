# PPNet

## 1. Framework

```shell
ppnet(
  (layer1): Sequential(
    (conv): Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1))
    (bn): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (layer2): Sequential(
    (conv): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1))
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True)
    (sigmoid): Sigmoid()
    (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (layer3): Sequential(
    (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True)
    (sigmoid): Sigmoid()
    (avgpool): AvgPool2d(kernel_size=2, stride=2, padding=0)
  )
  (layer4): Sequential(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True)
    (relu): ReLU()
  )
  (layer5): Sequential(
    (conv): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1))
    (bn): BatchNorm2d(256, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True)
    (relu): ReLU()
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc1): Linear(in_features=43264, out_features=512, bias=True)
  (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True)
  (relu1): ReLU()
  (fc2): Linear(in_features=512, out_features=512, bias=True)
  (bn2): BatchNorm1d(512, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True)
  (relu2): ReLU()
  (drop2): Dropout(p=0.25, inplace=False)
  (dis): PairwiseDistance()
  (fc3): Linear(in_features=512, out_features=600, bias=True)
)
```


## 2. Supplementary Materials

- Supplementary Material: [ [pdf](https://ieeexplore.ieee.org/abstract/document/9707646/) | [supp](https://ieeexplore.ieee.org/abstract/document/9707646/media#media)  ]
- Pretrained Models: [ [@](https://pan.baidu.com/s/1Y990hI1diS0bwmCetHTfPA) ] :key: iavt
- Raspberry Pi 4B Development Environment Establishment: [:scroll:](https://github.com/xuliangcs/env/blob/main/doc/RaspberryPi4B.md) 
- Publicly Available Datasets: [Tongji](https://sse.tongji.edu.cn/linzhang/contactlesspalm/index.htm), [IITD](https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm), [REST](https://ieee-dataport.org/open-access/rest-database),[NTU](https://github.com/BFLTeam/NTU_Dataset), [XJTU-UP](https://gr.xjtu.edu.cn/en/web/bell)
- Profiler: `pip install ptflops` [Flops-Counter](https://github.com/sovrasov/flops-counter.pytorch) 


## 3. Citation

```tex
@article{liang2022innovative,
title={Innovative Contactless Palmprint Recognition System Based on Dual-Camera Alignment},
author={Liang, Xu and Li, Zhaoqun and Fan, Dandan and Zhang, Bob and Lu, Guangming and Zhang, David},
journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
year={2022},
publisher={IEEE}
}
```

## 4. PyTorch Implementation

**Requirements**

- pytorch-1.2.0 

- torchvision-0.4.0

- python-3.7.4

- anaconda-4.9.0

- opencv-3.2.7 

 :point_right:[establishment](https://github.com/xuliangcs/env/blob/main/doc/PyTorch.md#PyTorch1.2)

**Configurations**

1. modify `path1` and `path2` in `genText.py`

    - `path1`: path of the training set (e.g., Tongji session1)
    - `path2`: path of the testing set (e.g., Tongji session2)
    
2. modify `num_classes` in `train.py` and `test.py`
    - Tongji: 600, IITD: 460, REST: 358, XJTU-UP: 200, KTU: 145, DCPD: 271
    
3. modify `python_path` in `train.py` and `test.py` according to which python you are using. ('python')

**Commands**


```shell
cd path/to/PPNet/

#in the PPNet folder:

#generate the training and testing data sets
python ./data/genText.py
mv ./train.txt ./data/
mv ./test.txt ./data/

#train the network
python train.py

#test the model
python test.py

#inference
python inference.py

#Metrics
#obtain the genuine-impostor matching score distribution curve
python    getGI.py   ./rst/veriEER/scores_xxx.txt    scores_xxx

#obtain the EER and the ROC curve
python    getEER.py   ./rst/veriEER/scores_xxx.txt    scores_xxx
```
The `.pth` file will be generated at the current folder, and all the other results will be generated in the `./rst` folder.

[![How to use pretrained models](https://img.shields.io/badge/Goto-UsePretrained-green)](https://github.com/xuliangcs/ppnet/blob/main/res/README_pretrained.md)
