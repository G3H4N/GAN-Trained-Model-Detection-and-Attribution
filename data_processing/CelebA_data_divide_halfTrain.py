# coding:utf-8


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

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset_dir = '/data/gehan/Datasets/CelebA/png/align/CelebA_png_align_Attr_22_21_20_sets/TrainSet'
CelebA_attr_train = loadData(trainset_dir)
CelebA_attr_train.dataset.transform = transform
CelebA_attr_train.dataset.transforms.transform = transform
lenth_sub_train = int(len(CelebA_attr_train)/2)
split = [lenth_sub_train, len(CelebA_attr_train) - lenth_sub_train]
sub_train1, sub_train2 = torch.utils.data.random_split(CelebA_attr_train, split)

saveData('/data/gehan/Datasets/CelebA/png/align/CelebA_png_align_Attr_22_21_20_sets/', 'CelebA_train_half1', sub_train1)
saveData('/data/gehan/Datasets/CelebA/png/align/CelebA_png_align_Attr_22_21_20_sets/', 'CelebA_train_half2', sub_train2)
