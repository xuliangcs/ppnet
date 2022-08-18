import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models import MyDataset
from models import ppnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('\ndevice-> ', device, '\n\n')


test_set = './data/test.txt'
testset =MyDataset(txt=test_set, transforms=None, train=False)
batch_size = 1
data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)


net = ppnet(num_classes=271) # IITD: 460    KTU: 145    Tongji: 600     REST: 358   DCPD:271    XJTU: 200

net.load_state_dict(torch.load('./net_params.pth'))

net.to(device)
net.eval()



# feature extraction:
featDB_test = []
iddb_test = []
with torch.no_grad():
    for batch_id, (data, target) in enumerate(data_loader_test):
        
        data = data.to(device)
        target = target.to(device)
        
        # feature extraction
        codes = net.getFeatureCode(data) 

        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test =  y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

print('completed feature extraction for test set.')
print('(number of samples, feature vector dimensionality): ', featDB_test.shape)
print('\n')


feat1 = featDB_test[0]
feat2 = featDB_test[1]
feat3 = featDB_test[-1]

# feature matching: feat1 vs feat2
dis = np.linalg.norm((feat1-feat2), 2)

print('matching distance, label1 vs label2: \t%.2f, %d vs %d'%(dis, iddb_test[0], iddb_test[1]))

# feature matching: feat1 vs feat3 
dis = np.linalg.norm((feat1-feat3), 2)

print('matching distance, label1 vs label3: \t%.2f, %d vs %d'%(dis, iddb_test[0], iddb_test[-1]))