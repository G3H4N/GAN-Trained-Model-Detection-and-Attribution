
import pickle
import torch
import numpy
import os
import sys
sys.path.append("..")
from models import structures
import matplotlib.pyplot as plt # matplotlib 1.4.3
from sklearn.manifold import TSNE # scikit-learn 0.17

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

GAN_classes_to_index_tensor = {'CGAN_gan_3':torch.tensor(0), 'ACGAN_hinge2_2':torch.tensor(1), 'DCGAN_gan_0':torch.tensor(2), 'DCGAN_wgan-gp_1':torch.tensor(3),
                               'SAGAN':torch.tensor(4), 'BigGAN_CR':torch.tensor(5), 'ContraGAN':torch.tensor(6), 'StyleGAN2_ada':torch.tensor(7),
                               'Real':torch.tensor(8)}
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
model = TSNE(n_components=3, perplexity=10, verbose=2, method='barnes_hut', init='pca', n_iter=1000)

X_tsne = model.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
y_min, y_max = Y.min(0), Y.max(0)
Y_norm = (Y - y_min) / (y_max - y_min)  # 归一化
#plt.figure(figsize=(8, 8))

hAx = plt.figure().add_subplot(projection='3d')
#hAx.scatter(xs=model.embedding_[:, 0], ys=model.embedding_[:, 1], zs=model.embedding_[:, 2], zdir='z', s=20)
for i in range(X_norm.shape[0]):
    hAx.text(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], str(int(Y[i])), color=plt.cm.Set1(Y_norm[i]), fontdict={'weight': 'bold', 'size': 15})
    #plt.text(X_norm[i, 0], X_norm[i, 1], str(int(Y[i])), plt.cm.Set1(Y_norm[i]), fontdict={'weight': 'bold', 'size': 15})
#plt.xticks([])
#plt.yticks([])
hAx.text2D(0.05, 0.95, "2D T-SNE", transform=hAx.transAxes)
hAx.set_xlim(0, 1)
hAx.set_ylim(0, 1)
hAx.set_zlim(0, 1)
plt.savefig('/data/gehan/PytorchProjects/GANInference/data_processing/plot/CIFAR10_real_trainset_DNA_tsne_3d.jpg')
plt.show()