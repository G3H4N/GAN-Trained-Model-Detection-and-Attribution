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

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(64, 64), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)
real = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform, split='test')
X_real = [numpy.array(real[0][0]).ravel()]
Y_real = [real[0][1]]
for i in range(1, len(real)):
    X_real.append(numpy.array(real[i][0]).ravel())#torch.cat((X, real_sub[i][0].unsqueeze(0)), 0)
    Y_real.append(real[i][1])

X_real = numpy.array(X_real)
Y_real = numpy.array(Y_real)

#lenth_real_sub = 1000
#split = [lenth_real_sub, len(real) - lenth_real_sub]
#real_sub, _ = torch.utils.data.random_split(real, split)
X_real = []
Y_real = []
classes = [0, 1, 2]
X_real_selected = []
Y_real_selected = []
for i in range(len(real)):
    if real[i][1] in classes:
        X_real_selected.append(numpy.array(real[i][0]).ravel())
        Y_real_selected.append(real[i][1])
    #X_real.append(numpy.array(real[i][0]).ravel())#torch.cat((X, real[i][0].unsqueeze(0)), 0)
    #Y_real.append(real[i][1])
    #if len(X_real_selected) >= 50:
    #    break

#X_real = numpy.array(X_real)
#Y_real = numpy.array(Y_real)

X_real_selected = numpy.array(X_real_selected)
Y_real_selected = numpy.array(Y_real_selected)

# Fit model
model = TSNE(n_components=2, perplexity=10, verbose=2, method='barnes_hut', init='pca', n_iter=1000)

model.fit(X_real_selected)
'''
# Plot results
hFig, hAx = plt.subplots()
hAx.scatter(model.embedding_[:, 0], model.embedding_[:, 1], 20, color="rgb")
for i, txt in enumerate(GAN_classes_to_index_tensor):
    hAx.annotate(txt, (model.embedding_[i, 0], model.embedding_[i, 1]))
plt.show()
'''
#X_tsne_real = model.fit_transform(X_real)
X_tsne_real_selected = model.fit_transform(X_real_selected)

'''
P_real = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'
c_dim = 10
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
    for batch_idx, (inputs, _) in enumerate(real_loader):

        inputs = inputs.to(device)
        outputs = classifier(inputs)
        # correctness of classification
        _, predicted = outputs.max(1)
        P_real.extend(predicted.cpu())
P_real = numpy.array(P_real)
'''
#嵌入空间可视化
x_tsne_real_selected_min, x_tsne_real_selected_max = X_tsne_real_selected.min(0), X_tsne_real_selected.max(0)
X_tsne_real_selected_norm = (X_tsne_real_selected - x_tsne_real_selected_min) / (x_tsne_real_selected_max - x_tsne_real_selected_min)  # 归一化

'''
y_real_min, y_real_max = Y_real.min(0), Y_real.max(0)
Y_real_norm = (Y_real - y_real_min) / (y_real_max - y_real_min)  # 归一化

plt.figure(figsize=(8, 8))
for i in range(X_tsne_real_norm.shape[0]):
    plt.text(X_tsne_real_norm[i, 0], X_tsne_real_norm[i, 1], str(int(Y_real_norm[i])), color=plt.cm.Set1(Y_real_norm[i]), fontdict={'weight': 'bold', 'size': 15})
plt.xticks([])
plt.yticks([])
#plt.savefig('./plot/CIFAR10_real_trainset_tsne.jpg')
#plt.show()
'''
num = 20
X_tsne_real_correct = []
Y_real_correct = []
while len(Y_real_correct) < 15:
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_tsne_real_selected, Y_real_selected)
    for i in range(num):
        x = X_tsne_real_selected[i, :].reshape(1, -1)
        p = knn_clf.predict(x)
        if p == Y_real_selected[i]:
            X_tsne_real_correct.append(X_tsne_real_selected[i])
            Y_real_correct.append(Y_real_selected[i])
        if len(Y_real_correct) >= 10:
            break

X_tsne_real_correct = numpy.array(X_tsne_real_correct)
Y_real_correct = numpy.array(Y_real_correct)

h = 1
#xticks_min, xticks_max = X_tsne_real_selected[:, 0].min() - h, X_tsne_real_selected[:, 0].max() + h
#yticks_min, yticks_max = X_tsne_real_selected[:, 1].min() - h, X_tsne_real_selected[:, 1].max() + h

#xx, yy = numpy.meshgrid(numpy.arange(xticks_min, xticks_max, h),
#                        numpy.arange(yticks_min, yticks_max, h))
xx, yy = numpy.meshgrid(numpy.arange(0, 1, 0.01),
                        numpy.arange(0, 1, 0.01))
Z = knn_clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)

#xx = numpy.diag(X_tsne_real_selected[:, 0])
#yy = numpy.diag(X_tsne_real_selected[:, 1])
#Z = numpy.diag(Y_real_selected)
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.get_cmap('Spectral'))#cm.Paired)#get_cmap('Spectral'))
plt.axis('off')
# Plot also the training points
plt.scatter(X_tsne_real_selected[:, 0], X_tsne_real_selected[:, 1], edgecolors=['black'], c=Y_real_selected, cmap=plt.get_cmap('Spectral'))
plt.show()
print('111')