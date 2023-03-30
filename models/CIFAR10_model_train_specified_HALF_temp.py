import numpy as np
import matplotlib.image as plimg
from PIL import Image
import pickle as p


def load_CIFAR_batch(filename):
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y


def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        lines = [x for x in f.readlines()]
        print(lines)


if __name__ == "__main__":
    # 文件的路径
    load_CIFAR_Labels("/data/gehan/Datasets/CIFAR10/cifar-10-batches-py/batches.meta")
    imgX, imgY = load_CIFAR_batch("/data/gehan/Datasets/CIFAR10/cifar-10-batches-py/test_batch")
    print(imgX.shape)

    for i in range(150):

        imgs = imgX[i]
        img0 = imgs[0]
        img1 = imgs[1]
        img2 = imgs[2]
        i0 = Image.fromarray(img0)  # 从数据，生成image对象
        i1 = Image.fromarray(img1)
        i2 = Image.fromarray(img2)
        img = Image.merge("RGB", (i0, i1, i2))
        img = img.resize((64, 64))
        name = "img" + str(i) + ".png"
        if imgY[i]==0 or imgY[i]==1 or imgY[i]==2 or imgY[i]==7:
            img.save("/data/gehan/PytorchProjects/GANInference/GANs/Conditional-GANs-Pytorch-master/CIFAR10_outputs/CIFAR10/" + str(imgY[i]) + "/" + name, "png")
        '''
        for j in range(imgs.shape[0]):
            img = imgs[j]
            name = "img" + str(i) + str(j) + ".png"
            plimg.imsave("/data/gehan/PytorchProjects/GANInference/GANs/Conditional-GANs-Pytorch-master/CIFAR10_outputs/CIFAR10/" + name, img)
        '''