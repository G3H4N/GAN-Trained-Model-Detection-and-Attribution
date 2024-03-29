
import pickle
import torch
import numpy
import os
import sys
sys.path.append("..")
from models import structures
import matplotlib.pyplot as plt # matplotlib 1.4.3
import umap

def loadData(filename):
    with open(filename, 'rb') as file:
        #print('Open file', filename, '......ok')
        obj = pickle.load(file, encoding='latin1')
        file.close()
        return obj


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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Read data
Concated = loadData('/data/gehan/PytorchProjects/GANInference/models/CIFAR10_outputs/real_GANship_Trainset_AllStrctr')
data = ConcatDataset2List(Concated)

model_ckpt_dir = os.path.join('/data/gehan/PytorchProjects/GANInference/models/attackers/CIFAR10_outputs/real_GANship_withReal/posterior_label/', 'checkpoints')
net = structures.lenet.attacker_real(nch_info=2000, nch_output=9).to(device)
net.eval()

ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
if ckpt_best:
    best_acc = ckpt_best['acc']
    net.load_state_dict(ckpt_best['net'])
else:
    ckpt = load_checkpoint(model_ckpt_dir)
    if ckpt:
        best_acc = ckpt['acc']
        net.load_state_dict(ckpt['net'])
    else:
        print(' [Wrong] No checkpoint!')
X = []
for i, info in enumerate(data[1]):
    embedding = net.embedding_(info.to(device))
    X.append(embedding)

X = torch.tensor([item.cpu().detach().numpy() for item in X])#torch.Tensor(X)
Y = numpy.array(data[3].cpu().ravel())

# Fit model
model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric='correlation')
x_umap = model.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], x_umap.shape[-1]))

'''嵌入空间可视化'''
x_min, x_max = x_umap.min(0), x_umap.max(0)
X_norm = (x_umap - x_min) / (x_max - x_min)  # 归一化
y_min, y_max = Y.min(0), Y.max(0)
Y_norm = (Y - y_min) / (y_max - y_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(int(Y[i])), color=plt.cm.Set1(Y_norm[i]), fontdict={'weight': 'bold', 'size': 15})
plt.xticks([])
plt.yticks([])
plt.savefig('/data/gehan/PytorchProjects/GANInference/data_processing/plot/CIFAR10_real_trainset_DNA_umap.jpg')
plt.show()