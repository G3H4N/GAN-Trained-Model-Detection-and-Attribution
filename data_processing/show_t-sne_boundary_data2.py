import os
import pickle
import torch
import torchvision
import numpy
import PIL.Image as Image
import matplotlib.pyplot as plt # matplotlib 1.4.3
from sklearn.manifold import TSNE # scikit-learn 0.17
from sklearn.neighbors import KNeighborsClassifier

import structures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 导入iris数据
from sklearn.datasets import load_iris
from sklearn import cluster

def loadData(filename):
    with open(filename, 'rb') as file:
        #print('Open file', filename, '......ok')
        obj = pickle.load(file, encoding='latin1')
        file.close()
        return obj
def saveData(dir, filename, obj):
    path = os.path.join(dir, filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if not os.path.exists(path):
        # 注意字符串中含有空格，所以有r' '；touch指令创建新的空文件
        os.system(r'touch {}'.format(path))

    with open(path, 'wb') as file:
        # pickle.dump(obj,file[,protocol])，将obj对象序列化存入已经打开的file中，file必须以二进制可写模式打开（“wb”），可选参数protocol表示高职pickler使用的协议，支持的协议有0，1，2，3，默认的协议是添加在python3中的协议3。
        pickle.dump(obj, file)
        file.close()
    print('Save data', path, '......ok')

    return
def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_path)
    return ckpt

def ConcatDataset2List(Concated):
    datasets = []
    for i in range(len(Concated.datasets)):
        datasets.append(list(Concated.datasets[i].tensors))

    data = datasets[0]
    for i in range(len(data)):
        for j in range(len(datasets)):
            if j == 0:
                continue
            data[i] = torch.cat((data[i], datasets[j][i]), 0)

    return data

# ==============================================================================
# =                                real data                                   =
# ==============================================================================

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(64, 64), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)
real = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform, split='test')
'''
real_loader = torch.utils.data.DataLoader(real, batch_size=5, shuffle=False)
X_real = [numpy.array(real[0][0]).ravel()]
Y_real = [real[0][1]]
for i in range(1, len(real)):
    X_real.append(numpy.array(real[i][0]).ravel())#torch.cat((X, real_sub[i][0].unsqueeze(0)), 0)
    Y_real.append(real[i][1])

X_real = numpy.array(X_real)
Y_real = numpy.array(Y_real)
'''
#lenth_real_sub = 1000
#split = [lenth_real_sub, len(real) - lenth_real_sub]
#real_sub, _ = torch.utils.data.random_split(real, split)
X_real = []
Y_real = []
classes = [0, 1, 2]
len_classes = [0, 0, 0]
X_real_selected = []
X_real_selected_tensor = []
Y_real_selected = []
for i in range(len(real)):
    if real[i][1] in classes:
        if len_classes[classes.index(real[i][1])] >= 15:
            continue
        X_real_selected_tensor.append((real[i][0]))
        X_real_selected.append(numpy.array(real[i][0]).ravel())
        Y_real_selected.append(real[i][1])
        len_classes[classes.index(real[i][1])] += 1
    #X_real.append(numpy.array(real[i][0]).ravel())#torch.cat((X, real[i][0].unsqueeze(0)), 0)
    #Y_real.append(real[i][1])
    if len(X_real_selected) >= 45:
        break
#X_real = numpy.array(X_real)
#Y_real = numpy.array(Y_real)
X_real_selected = numpy.array(X_real_selected)
Y_real_selected = numpy.array(Y_real_selected)

# ==============================================================================
# =                                gan data                                    =
# ==============================================================================
from structures import GAN_structures

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def Denormalize(data):
    return (data + 1) / 2
def GenerateDataSet(gan_path, z_dim, c_dim, dataset_size_perclass, dataset_path):

    # Prepare GAN model
    _, gan_name = os.path.split(gan_path)
    if gan_name.startswith('DCGAN'):
        G = GAN_structures.GeneratorDCGAN(nch_in=z_dim, n_class=c_dim).to(device)
    else:
        G = GAN_structures.GeneratorACGAN(z_dim=z_dim, c_dim=c_dim).to(device)

    # Load GAN
    assert os.path.exists(gan_path), 'Wrong directory for loading GAN!'
    ckpt_dir = os.path.join(gan_path, 'checkpoints')
    ckpt = load_checkpoint(ckpt_dir)
    G.load_state_dict(ckpt['G'])

    # Make directories
    for i in range(c_dim):
        path = '%s/%d/' % (dataset_path, i)
        if not os.path.exists(path):
            os.makedirs(path)

    with torch.no_grad():
        G.eval()
        # One-hot labels from 0 to c_dim
        input_labels = torch.zeros(c_dim, c_dim).to(device)
        for i in range(c_dim):
            input_labels[i][i % c_dim] = 1

        if gan_name.startswith('DCGAN'):
            input_labels = input_labels.unsqueeze(-1).unsqueeze(-1)

        for n in range(dataset_size_perclass):
            input_noise = torch.randn(c_dim, z_dim).to(device)
            if gan_name.startswith('DCGAN'):
                input_noise = input_noise.unsqueeze(-1).unsqueeze(-1)
            output = G(input_noise, input_labels)

            output = Denormalize(output)
            for i in range(output.shape[0]):
                im = output[i].cpu().clone()
                im = im.squeeze(0)
                temp = '%s/%d/testimage_gen_%04d.png' % (dataset_path, i%c_dim, n)
                torchvision.utils.save_image(im, temp)

            if (dataset_size_perclass >= 200) and (n + 1) % 100 == 0:
                print('%d / %d' % ((n + 1), dataset_size_perclass))

    print('DataSet generation finished!')

gan_path = '/data/gehan/GANInference/GANs/Conditional-GANs-Pytorch-master/SVHN_outputs/DCGAN_wgan-gp_4'
gan_data_dir = './GAN_data'
_, gan_name = os.path.split(gan_path)
gan_data_path = os.path.join(gan_data_dir, gan_name)
if not os.path.exists(gan_data_path):
    GenerateDataSet(gan_path, 100, 10, 100, gan_data_path)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
gan = torchvision.datasets.ImageFolder(gan_data_path, transform=transform)


classes = [0, 1, 2]
len_classes = [0, 0, 0]
X_gan_selected = []
X_gan_selected_tensor = []
Y_gan_selected = []
for i in range(len(gan)):
    if gan[i][1] in classes:
        if len_classes[classes.index(gan[i][1])] >= 15:
            continue
        X_gan_selected_tensor.append((gan[i][0]))
        X_gan_selected.append(numpy.array(gan[i][0]).ravel())
        Y_gan_selected.append(gan[i][1])
        len_classes[classes.index(gan[i][1])] += 1

    if len(X_gan_selected) >= 45:
        break

X_gan_selected = numpy.array(X_gan_selected)
Y_gan_selected = numpy.array(Y_gan_selected)


# ==============================================================================
# =                          classifier prediction                             =
# ==============================================================================
X_real_selected_tensor.extend(X_gan_selected_tensor)
P_real = []
c_dim = 10
#/data/gehan/GANInference/models/SVHN_outputs/Real/VGG11_4
#/data/gehan/GANInference/models/SVHN_outputs/DCGAN_wgan-gp_4/VGG11_3
classifier_path = '/data/gehan/GANInference/models/SVHN_outputs/DCGAN_wgan-gp_4/VGG11_3'
_, classifier_name = os.path.split(classifier_path)
if classifier_name.startswith('LeN'):
    classifier = structures.lenet.LeNet(c_dim=c_dim).to(device)
elif classifier_name.startswith('Smp'):
    classifier = structures.lenet.SmplCNN(c_dim=c_dim).to(device)
elif classifier_name.startswith('VGG'):
    classifier = structures.vgg.VGG('VGG11', c_dim=c_dim).to(device)
else:# classifier_name.startswith('Res'):
    classifier = structures.resnet.ResNet18(c_dim=c_dim).to(device)
# Load checkpoint
model_ckpt_dir = os.path.join(classifier_path, 'checkpoints')
classifier.eval()
ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
if ckpt_best:
    classifier.load_state_dict(ckpt_best['net'])
else:
    ckpt = load_checkpoint(model_ckpt_dir)
    if ckpt:
        classifier.load_state_dict(ckpt['net'])
    else:
        print(' [Wrong] No checkpoint!')
with torch.no_grad():
    for input in X_real_selected_tensor:

        input = input.unsqueeze(0).to(device)
        output = classifier(input)
        # correctness of classification
        _, predicted = output.max(1)
        P_real.extend(predicted.cpu())
P_real = numpy.array(P_real)

# ==============================================================================
# =                           dimension reduction                              =
# ==============================================================================
# 导入模型
from sklearn.neighbors import KNeighborsClassifier

X = np.concatenate([X_real_selected, X_gan_selected],axis=0)
model_pca = PCA(n_components=2)
model_pca.fit(X)
pac_X = model_pca.transform(X)
pca_X_real_selected = model_pca.transform(X_real_selected)
pca_X_gan_selected = model_pca.transform(X_gan_selected)


# ==============================================================================
# =                    decision boundary visualization                         =
# ==============================================================================
# 训练模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(pac_X, P_real)

def plot_decision_boundary(clf, axes):
    xp = np.linspace(axes[0], axes[1], 50)  # 均匀300个横坐标
    yp = np.linspace(axes[2], axes[3], 50)  # 均匀300个纵坐标
    x1, y1 = np.meshgrid(xp, yp)  # 生成300x300个点
    xy = np.c_[x1.ravel(), y1.ravel()]  # 按行拼接，规范成坐标点的格式
    y_pred = clf.predict(xy)  # 训练之后平铺
    xx = x1
    yy = y1
    z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.6, cmap=plt.get_cmap('Spectral'))#cm.Paired)#get_cmap('Spectral'))

    x2 = []
    x1 = []
    for i in xy:
        x1.append(i[0])
        x2.append(i[1])
    color = {0: '#fafab0', 1: '#9898ff', 2: '#a0faa0'}
    #plt.scatter(x1, x2, color=[color[i % 3] for i in y_pred], alpha=0.6)
    #plt.scatter(x1, x2, c=y_pred, alpha=0.3, cmap=plt.get_cmap('Spectral'))


plot_decision_boundary(knn, axes=[np.min(pac_X[:, 0]), np.max(pac_X[:, 0]), np.min(pac_X[:, 1]),
                                  np.max(pac_X[:, 1])])

# ==============================================================================
# =                        data points visualization                           =
# ==============================================================================

x2 = []
x1 = []
for i in pca_X_real_selected:
    x1.append(i[0])
    x2.append(i[1])

predicts_real = knn.predict(pca_X_real_selected)
plt.scatter(x1[:15], x2[:15], c=Y_real_selected[:15], alpha=1, marker='^', edgecolors=['black'], cmap=plt.get_cmap('Spectral'))

x2 = []
x1 = []
for i in pca_X_gan_selected:
    x1.append(i[0])
    x2.append(i[1])

predicts_gan = knn.predict(pca_X_gan_selected)
plt.scatter(x1[:15], x2[:15], c=Y_gan_selected[:15], alpha=1, marker='o', edgecolors=['black'], cmap=plt.get_cmap('Spectral'))

#predicts_real
plt.show()