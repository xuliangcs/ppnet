# PPNet

[ [paper](https://ieeexplore.ieee.org/document/9707646) | [cite](./res/cite.txt) | [license](./LICENSE) ]


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

- Paper: [online](https://ieeexplore.ieee.org/abstract/document/9707646)

- Supplementary Material: [ [supp](https://ieeexplore.ieee.org/abstract/document/9707646/media#media) ]
- Pretrained Models: [google](https://drive.google.com/drive/folders/1xcBzSxIDDWeIKK4mb6XHIdgwGsvQURHc?usp=drive_link) or [baidu :key: iavt](https://pan.baidu.com/s/1Y990hI1diS0bwmCetHTfPA?pwd=iavt)

- Raspberry Pi 4B Development Environment Establishment: [:scroll:](https://github.com/xuliangcs/env/blob/main/doc/RaspberryPi4B.md)

- Publicly Available Datasets: [DCPD](http://xliang.me/res/supp/dual-camera/dcpd.txt), [Tongji](https://cslinzhang.github.io/ContactlessPalm), [IITD](https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Palm.htm), [REST](https://ieee-dataport.org/open-access/rest-database), [NTU](https://github.com/BFLTeam/NTU_Dataset), [XJTU-UP](https://gr.xjtu.edu.cn/en/web/bell)

- Profiler: `pip install ptflops` [Flops-Counter](https://github.com/sovrasov/flops-counter.pytorch) 


## 3. PyTorch Implementation

![](https://img.shields.io/badge/Ubuntu-tested-green) ![](https://img.shields.io/badge/Windows11-tested-green) 

**Requirements**
Recommanded hardware requirement **for training**:
- `GPU Mem` $\ge$ 3G
- `CPU Mem` $\ge$ 16G

Development environment establishment:
- [cuda & cudnn & gpu-driver](https://github.com/xuliangcs/env/blob/main/doc/PyTorch.md)
- `Anaconda`: [download & install](https://www.anaconda.com/download/success)
- `PyTorch`: installation command lines are as follows
  ```
  conda create -n ppnet python=3.8 
  conda activate ppnet

  conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

  pip install -r requirements.txt
  ```
tips:
- `requirements.txt` could be found at the root folder of this project
- use different [CUDA versions](https://pytorch.org/get-started/previous-versions/#v170)

Tested versions:

- pytorch：1.2 to 1.7

- torchvision：0.4 to 0.8

- python: 3.7 to 3.8

- opencv: 4.8 (Generally, the default version is sufficient)

 :point_right:[more details](https://github.com/xuliangcs/env/blob/main/doc/PyTorch.md)

**Configurations**

1. modify `path1` and `path2` in `genText.py`

    - `path1`: path of the training set (e.g., Tongji session1)
    - `path2`: path of the testing set (e.g., Tongji session2)
    
2. modify `num_classes` in `train.py`, `test.py`, and `inference.py`
    - DCPD: 271, Tongji: 600, IITD: 460, REST: 358, XJTU-UP: 200, KTU: 145

**Commands**


```shell
cd path/to/PPNet/
#in the PPNet folder:

#prepare data
cp ./data/for_reference/genText_xxx.py ./data/genText.py
#where xxx is the dataset name, e.g., tongji =>genText_tongji.py
Modify the DB path variable in ./data/genText.py
#the sample naming format should be consistent with the script's requirements

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

**Dataset preparation**
1. The `genText.py` script is responsible for traversing images in the dataset folder and parsing class labels (starting from 0) based on each filename's format.
    - For each sample, the full path (including the filename) and its corresponding class label (separated by a space) are saved as a single line in either the `train.txt` or `test.txt` file.
    - In our experiments, each individual palm represents a unique class.
2. The method used to extract `userID` and `sampleID` from image filenames is implemented within the script, handling two main scenarios:
    - For the original Tongji dataset, image filenames range sequentially from `00001.bmp` to 06000.bmp. Every consecutive group of 10 samples originates from the same palm. Therefore, in genText.py, the userID (class label) is derived by integer division of the numeric filename by 10 (i.e., filename // 10).
    - or other datasets with complex directory structures, preprocessing can be applied to simplify organization, such as renaming files and placing them into a single folder. In such cases, the userID parsing logic in genText.py must align with the new filename and directory conventions.
    - The recommended renaming format is: xxxx_yyyy.zzz
        - xxxx denotes the userID, representing a unique palm.
        - yyyy denotes the sampleID, representing an individual capture of that palm.
        - IDs with fewer than four digits are zero-padded on the left.
        - zzz is the image file extension (e.g., bmp, jpg, tiff,etc.).
        - Example: 0010_0003.bmp represents the 3rd sample of palm #10.

Sample output of `genText.py`:
test.txt (Tongji):
```shell
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00001.bmp 0
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00002.bmp 0
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00003.bmp 0
...
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00008.bmp 0
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00009.bmp 0
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00010.bmp 0
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00011.bmp 1
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00012.bmp 1
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00013.bmp 1
...
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00018.bmp 1
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00019.bmp 1
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00020.bmp 1
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00021.bmp 2
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00022.bmp 2
/home/sunny/datasets/Tongji/palmprint/ROI/session2/00023.bmp 2
...
/home/sunny/datasets/Tongji/palmprint/ROI/session2/05991.bmp 599
/home/sunny/datasets/Tongji/palmprint/ROI/session2/05992.bmp 599
/home/sunny/datasets/Tongji/palmprint/ROI/session2/05993.bmp 599
...
/home/sunny/datasets/Tongji/palmprint/ROI/session2/05998.bmp 599
/home/sunny/datasets/Tongji/palmprint/ROI/session2/05999.bmp 599
/home/sunny/datasets/Tongji/palmprint/ROI/session2/06000.bmp 599
```
test.txt (IITD):
```shell
/home/sunny/datasets/IITD/roi/0001_0001.bmp 0
/home/sunny/datasets/IITD/roi/0001_0002.bmp 0
/home/sunny/datasets/IITD/roi/0001_0003.bmp 0
/home/sunny/datasets/IITD/roi/0002_0001.bmp 1
/home/sunny/datasets/IITD/roi/0002_0002.bmp 1
/home/sunny/datasets/IITD/roi/0002_0003.bmp 1
/home/sunny/datasets/IITD/roi/0003_0001.bmp 2
/home/sunny/datasets/IITD/roi/0003_0002.bmp 2
/home/sunny/datasets/IITD/roi/0003_0003.bmp 2
/home/sunny/datasets/IITD/roi/0004_0001.bmp 3
/home/sunny/datasets/IITD/roi/0004_0002.bmp 3
/home/sunny/datasets/IITD/roi/0004_0003.bmp 3
...
```

## 6. Citation
🌻If it helps you, please cite the following paper:🌱

```tex
@article{liang2022innovative,
  author={Liang, Xu and Li, Zhaoqun and Fan, Dandan and Zhang, Bob and Lu, Guangming and Zhang, David},
  journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems}, 
  title={Innovative Contactless Palmprint Recognition System Based on Dual-Camera Alignment}, 
  year={2022},
  volume={52},
  number={10},
  pages={6464-6476},
  doi={10.1109/TSMC.2022.3146777}}
```

Xu Liang, Zhaoqun Li, Dandan Fan, Bob Zhang, Guangming Lu and David Zhang, "Innovative Contactless Palmprint Recognition System Based on Dual-Camera Alignment," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 52, no. 10, pp. 6464-6476, Oct. 2022, doi: 10.1109/TSMC.2022.3146777.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## References

[1] X. Liang, D. Fan, J. Yang, W. Jia, G. Lu and D. Zhang, "PKLNet: Keypoint Localization Neural Network for Touchless Palmprint Recognition Based on Edge-Aware Regression," in IEEE Journal of Selected Topics in Signal Processing, 17(3), pp. 662-676, May 2023, [doi](https://ieeexplore.ieee.org/document/10049596): 10.1109/JSTSP.2023.3241540. (`Palmprint ROI extraction`) [pklnet](https://github.com/xuliangcs/pklnet)🖖

[2] X. Liang, J. Yang, G. Lu and D. Zhang, "CompNet: Competitive Neural Network for Palmprint Recognition Using Learnable Gabor Kernels," in IEEE Signal Processing Letters, vol. 28, pp. 1739-1743, 2021, [doi](https://ieeexplore.ieee.org/document/9512475): 10.1109/LSP.2021.3103475. (`Orientation coding`) [compnet](https://github.com/xuliangcs/compnet)🖐️

[3] X. Liang, Z. Li, D. Fan, B. Zhang, G. Lu and D. Zhang, "Innovative Contactless Palmprint Recognition System Based on Dual-Camera Alignment," in IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 52, no. 10, pp. 6464-6476, Oct. 2022, [doi](https://ieeexplore.ieee.org/document/9707646): 10.1109/TSMC.2022.3146777. (`Bimodal alignment`)

[4] PyTorch API Documents: https://pytorch.org/docs/stable/index.html