'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms
import os
import pickle
import argparse
from structures import *


def saveData(dir, filename, obj):

    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(dir + filename):
        # 注意字符串中含有空格，所以有r' '；touch指令创建新的空文件
        os.system(r'touch {}'.format(dir + filename))

    with open(dir + filename, 'wb') as file:
        # pickle.dump(obj,file[,protocol])，将obj对象序列化存入已经打开的file中，file必须以二进制可写模式打开（“wb”），可选参数protocol表示高职pickler使用的协议，支持的协议有0，1，2，3，默认的协议是添加在python3中的协议3。
        pickle.dump(obj, file)
        file.close()
    print('Save data', dir + filename, '......ok')

    return


def loadData(filename):

    with open(filename, 'rb') as file:
        print('Open file', filename, '......ok')
        obj = pickle.load(file, encoding='latin1')
        file.close()
        return obj

batch_size = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_dir = '/data/gehan/pytorchProjects/MIagainstGAN/structures/pytorch-cifar-master/1_pureAugmentation/Shadow_lenet_aug_92.95_53.98.pth'
train_dir = '/data/gehan/pytorchProjects/MIagainstGAN/GANs/data/cifar10_pre/Shadow_train'
test_dir = '/data/gehan/pytorchProjects/MIagainstGAN/GANs/data/cifar10_pre/test_shadow'
save_dir = '/data/gehan/pytorchProjects/MIagainstGAN/GANs/data/attack_1pureAug/'
model_name = 'shadow_aug'

test_size_half = 100


Mem = loadData(train_dir)
NonMem = loadData(test_dir)

length = len(Mem)
indices = list(range(length))
Mem_train = torch.utils.data.Subset(Mem, indices[:length - test_size_half])
Mem_test = torch.utils.data.Subset(Mem, indices[length - test_size_half:])
#saveData(data_dir+'cifar10_pre/', 'Target1_train', Target1_train)
#saveData(data_dir+'cifar10_pre/', 'Shadow_train', Shadow_train)

length = len(NonMem)
indices = list(range(length))
NonMem_train = torch.utils.data.Subset(NonMem, indices[:length - test_size_half])
NonMem_test = torch.utils.data.Subset(NonMem, indices[length - test_size_half:])
#saveData(data_dir+'cifar10_pre/', 'Target1_train', Target1_train)
#saveData(data_dir+'cifar10_pre/', 'Shadow_train', Shadow_train)


#TrainSet = {'image': [], 'PostLabel': [], 'membership': []}
#TestSet = {'image': [], 'PostLabel': [], 'membership': []}


model = LeNet()
model = model.to(device)
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint['net'])

images = torch.tensor(0)
PostLabel = torch.tensor(0)
members = torch.tensor(0)
for dataset in [Mem_train, NonMem_train]:

    if dataset is Mem_train:
        #TrainSet['membership'].extend(np.ones(len(dataset)))
        members = torch.from_numpy(np.ones(len(dataset)))
    if dataset is NonMem_train:
        #TrainSet['membership'].extend(np.zeros(len(dataset)))
        members = torch.cat((members, torch.from_numpy(np.zeros(len(dataset)))), 0)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.eval()
    with torch.no_grad():
        for batch_idx, (img, l) in enumerate(dataloader):
            img = img.to(device)
            outputs = model(img)
            labels = torch.zeros(outputs.shape).to(device)
            for i in range(len(l)):
                labels[i][l[i]] = 1
            info = torch.cat([outputs, labels], 1)
            #info = [x for x in info.detach().cpu().numpy()]
            #TrainSet['PostLabel'].extend(info)
            if len(PostLabel.shape) == 0:
                PostLabel = info
            else:
                PostLabel = torch.cat((PostLabel, info), 0)

            if len(images.shape) == 0:
                #TrainSet['image'].extend(img)
                images = img
            else:
                images = torch.cat((images, img), 0)

#i = torch.stack(TrainSet['image'], dim=0)
#i = torch.cat(TrainSet['image'], dim=0)
#x = torch.stack(TrainSet['PostLabel'], dim=0)
#y = torch.LongTensor(TrainSet['membership'])
TrainSet = torch.utils.data.TensorDataset(images, PostLabel, members)
saveData(save_dir, model_name+'_%d_%d_Train' % (members.sum(), len(members)-members.sum()), TrainSet)


images = torch.tensor(0)
PostLabel = torch.tensor(0)
members = torch.tensor(0)
for dataset in [Mem_test, NonMem_test]:

    if dataset is Mem_test:
        members = torch.from_numpy(np.ones(len(dataset)))
    if dataset is NonMem_test:
        members = torch.cat((members, torch.from_numpy(np.zeros(len(dataset)))), 0)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.eval()
    with torch.no_grad():
        for batch_idx, (img, l) in enumerate(dataloader):
            img = img.to(device)
            outputs = model(img)
            labels = torch.zeros(outputs.shape).to(device)
            for i in range(len(l)):
                labels[i][l[i]] = 1
            info = torch.cat([outputs, labels], 1)
            if len(PostLabel.shape) == 0:
                PostLabel = info
            else:
                PostLabel = torch.cat((PostLabel, info), 0)

            if len(images.shape) == 0:
                images = img
            else:
                images = torch.cat((images, img), 0)


TestSet = torch.utils.data.TensorDataset(images, PostLabel, members)
saveData(save_dir, model_name+'_%d_%d_Test' % (members.sum(), len(members)-members.sum()), TestSet)








'''
for dataset in [Mem_test, NonMem_test]:
    if dataset is Mem_test:
        TestSet['membership'].extend(np.ones(len(dataset)))
    if dataset is NonMem_test:
        TestSet['membership'].extend(np.zeros(len(dataset)))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.eval()
    with torch.no_grad():
        for batch_idx, (img, l) in enumerate(dataloader):
            img = img.to(device)
            outputs = model(img)
            labels = torch.zeros(outputs.shape).to(device)
            for i in range(len(l)):
                labels[i][l[i]] = 1
            info = torch.cat([outputs, labels], 1)
            #info = [x for x in info.detach().cpu().numpy()]
            TestSet['PostLabel'].extend(info)
            TestSet['image'].extend(img)

x = torch.stack(TestSet['PostLabel'], dim=0)
y = torch.LongTensor(TestSet['membership'])
TestSet = torch.utils.data.TensorDataset(x, y)
saveData(save_dir, model_name+'_Test', TestSet)
'''