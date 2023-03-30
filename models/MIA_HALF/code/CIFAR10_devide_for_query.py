# coding:utf-8
"""
DataSetSegmentation.py
==========================
Segment a torchvision dataset.py for the training and testing of the target model and shadow structures.
"""

import torch
import torchvision
import pickle  # pickle提供了一个简单的持久化功能，可以序列化对象并以文件的形式存放在磁盘上
import os
import PIL.Image as Image


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


transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(64, 64), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)

Trainset1 = loadData('/data/gehan/Datasets/CIFAR10/Trainset1')
size_whole = len(Trainset1)
size_Din_t = 1000
divide = [size_Din_t, size_whole - size_Din_t]
Din_t, _ = torch.utils.data.random_split(Trainset1, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Din_t', Din_t)

Trainset2 = loadData('/data/gehan/Datasets/CIFAR10/Trainset2')
size_whole = len(Trainset2)
size_Din = 2500
divide = [size_Din, size_whole - size_Din]
Din_real1, _ = torch.utils.data.random_split(Trainset2, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Din_real1', Din_real1)

Trainset3 = loadData('/data/gehan/Datasets/CIFAR10/Trainset3')
size_whole = len(Trainset3)
size_Din = 2500
divide = [size_Din, size_whole - size_Din]
Din_real2, _ = torch.utils.data.random_split(Trainset3, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Din_real2', Din_real2)

dataset = loadData('/data/gehan/Datasets/CIFAR10/GAN-generated/StyleGAN2_ada/Fake2')
size_whole = len(dataset)
size_Din = 2500
divide = [size_Din, size_whole - size_Din]
Din_same, _ = torch.utils.data.random_split(dataset, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Din_same', Din_same)

dataset = loadData('/data/gehan/Datasets/CIFAR10/GAN-generated/StyleGAN2_ada/Fake3')
size_whole = len(dataset)
size_Din = 5000
divide = [size_Din, size_whole - size_Din]
Din_ALLsame, _ = torch.utils.data.random_split(dataset, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Din_ALLsame', Din_ALLsame)

dataset = loadData('/data/gehan/Datasets/CIFAR10/GAN-generated/BigGAN_CR/Fake1')
size_whole = len(dataset)
size_Din = 2500
divide = [size_Din, size_whole - size_Din]
Din_other, _ = torch.utils.data.random_split(dataset, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Din_other', Din_other)

dataset = loadData('/data/gehan/Datasets/CIFAR10/GAN-generated/BigGAN_CR/Fake3')
size_whole = len(dataset)
size_Din = 5000
divide = [size_Din, size_whole - size_Din]
Din_ALLother, _ = torch.utils.data.random_split(dataset, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Din_ALLother', Din_ALLother)

# ======= Dout ======= #

CIFAR10_test = torchvision.datasets.CIFAR10(root='/data/gehan/Datasets/CIFAR10/', train=False, download=True, transform=transform)
size_whole = len(CIFAR10_test)
size_Dout_t = 1000
size_Dout_real = 2500
size_Dout_Allreal = 5000
divide = [size_Dout_t, size_Dout_real, size_Dout_Allreal, size_whole - size_Dout_t - size_Dout_real - size_Dout_Allreal]
Dout_t, Dout_real, Dout_Allreal, _ = torch.utils.data.random_split(CIFAR10_test, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Dout_t', Dout_t)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Dout_real', Dout_real)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Dout_Allreal', Dout_Allreal)


dataset = loadData('/data/gehan/Datasets/CIFAR10/GAN-generated/StyleGAN2_ada/Other')
size_whole = len(dataset)
size_Dout_same = 2500
size_Dout_Allsame = 5000
divide = [size_Dout_same, size_Dout_Allsame, size_whole - size_Dout_same - size_Dout_Allsame]
Dout_same, Dout_Allsame, _ = torch.utils.data.random_split(dataset, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Dout_same', Dout_same)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Dout_Allsame', Dout_Allsame)


dataset = loadData('/data/gehan/Datasets/CIFAR10/GAN-generated/BigGAN_CR/Other')
size_whole = len(dataset)
size_Dout_other = 2500
size_Dout_Allother = 5000
divide = [size_Dout_other, size_Dout_Allother, size_whole - size_Dout_other - size_Dout_Allother]
Dout_other, Dout_Allother, _ = torch.utils.data.random_split(dataset, divide)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Dout_other', Dout_other)
saveData('/data/gehan/PytorchProjects/GANInference/models/MIA_HALF/data_all/', 'Dout_Allother', Dout_Allother)
