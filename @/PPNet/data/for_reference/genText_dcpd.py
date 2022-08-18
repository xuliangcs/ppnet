import os
import numpy as np


path_p = '/home/sunny/datasets/dcpd/roi_print'


root = './'
ftrain = open(os.path.join(root, 'train.txt'), 'w')
ftest = open(os.path.join(root, 'test.txt'), 'w')


pimgs = sorted(os.listdir(path_p))
i = 0

for i in range(len(pimgs)):
    filename = pimgs[i]
    userID = int(filename[:4])
    sampleID = int(filename[5:9])

    print(userID, sampleID)

    printImgPath = os.path.join(path_p, filename)  
    
    if sampleID <= 5: 
        ftrain.write('%s %d\n'%(printImgPath, userID-1))
    else:
        ftest.write('%s %d\n'%(printImgPath, userID-1))


ftrain.close()
ftest.close()

