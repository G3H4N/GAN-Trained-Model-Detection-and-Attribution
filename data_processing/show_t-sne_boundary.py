
import pickle
import torch
import numpy
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


GAN_classes_to_index_tensor = {'CGAN_gan_3':torch.tensor(0), 'ACGAN_hinge2_2':torch.tensor(1), 'DCGAN_gan_0':torch.tensor(2), 'DCGAN_wgan-gp_1':torch.tensor(3),
                               'SAGAN':torch.tensor(4), 'BigGAN_CR':torch.tensor(5), 'ContraGAN':torch.tensor(6), 'StyleGAN2_ada':torch.tensor(7),
                               'Real':torch.tensor(8)}
# Read data
Concated = loadData('/data/gehan/GANInference/models/CIFAR10_outputs/real_GANship_Trainset_AllStrctr')
data = ConcatDataset2List(Concated)
X = data[1].cpu()
Y = numpy.array(data[3].cpu().ravel())

# Fit model
model = TSNE(n_components=2, perplexity=10, verbose=2, method='barnes_hut', init='pca', n_iter=1000)
'''
model.fit(X)
# Plot results
hFig, hAx = plt.subplots()
hAx.scatter(model.embedding_[:, 0], model.embedding_[:, 1], 20, color="rgb")
for i, txt in enumerate(GAN_classes_to_index_tensor):
    hAx.annotate(txt, (model.embedding_[i, 0], model.embedding_[i, 1]))
plt.show()
'''
X_tsne = model.fit_transform(X)

print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
y_min, y_max = Y.min(0), Y.max(0)
Y_norm = (Y - y_min) / (y_max - y_min)  # 归一化
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(int(Y[i])), color=plt.cm.Set1(Y_norm[i]), fontdict={'weight': 'bold', 'size': 15})
plt.xticks([])
plt.yticks([])
plt.savefig('./plot/CIFAR10_real_trainset_tsne.jpg')
plt.show()