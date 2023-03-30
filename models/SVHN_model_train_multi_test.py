'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import os
import argparse
import pickle
import sys
import PIL
from numpy import *
import shutil
import structures
import GAN_structures

# ==============================================================================
# =                                  Settings                                  =
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
# Train setting
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.0001)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=64)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=100)
parser.add_argument('--trainset_size_perclass', dest='trainset_size_perclass', type=int, help='size of train set',
                    default=10000)
parser.add_argument('--testset_gen_size_perclass', dest='testset_gen_size_perclass', type=int, help='size of generated test set',
                    default=200)
parser.add_argument('--c_dim', dest='c_dim', type=int, help='number of classes',
                    default=10)
parser.add_argument('--num_models', dest='num_models', type=int, help='number of models to be trained for each GAN',
                    default=5)
# GANs
parser.add_argument('--gan_addr', dest='gan_addr', type=str, help='gan address',
                    default='/data/gehan/PytorchProjects/GANInference/GANs/Conditional-GANs-Pytorch-master/SVHN_outputs')
parser.add_argument('--gan_name', dest='gan_name', type=str, help='gan name',
                    default='CGAN_gan_')# DCGAN_wgan-gp_  DCGAN_gan_  ACGAN_hinge2_lr_  CGAN_gan_
parser.add_argument('--z_dim', dest='z_dim', type=int, help='dimension of input noise for GAN',
                    default=100)
parser.add_argument('--d_learning_rate', dest='d_learning_rate', type=float, default=0.0002)
parser.add_argument('--g_learning_rate', dest='g_learning_rate', type=float, default=0.001)
parser.add_argument('--n_d', dest='n_d', type=int, help='# of d updates per g update', default=1)
parser.add_argument('--loss_mode', dest='loss_mode', choices=['gan', 'lsgan', 'wgan', 'hinge_v1', 'hinge_v2'], default='gan')
parser.add_argument('--gp_mode', dest='gp_mode', choices=['none', 'dragan', 'wgan-gp'], default='none')
parser.add_argument('--gp_coef', dest='gp_coef', type=float, default=1.0)
parser.add_argument('--norm', dest='norm', choices=['none', 'batch_norm', 'instance_norm'], default='none')
parser.add_argument('--weight_norm', dest='weight_norm', choices=['none', 'spectral_norm', 'weight_norm'], default='spectral_norm')

# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/SVHN_outputs/')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='VGG11_')
parser.add_argument('--noise_m', dest='noise_m', type=float, help='magnitude of addictive noise',
                    default=5)
args = parser.parse_args()

# Train settings
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
#resume = args.resume
trainset_size_perclass = args.trainset_size_perclass
testset_gen_size_perclass = args.testset_gen_size_perclass
c_dim = args.c_dim
num_models = args.num_models
# GANs
gan_addr = args.gan_addr
gan_name = args.gan_name
z_dim = args.z_dim
d_learning_rate = args.d_learning_rate
g_learning_rate = args.g_learning_rate
n_d = args.n_d
loss_mode = args.loss_mode
gp_mode = args.gp_mode
gp_coef = args.gp_coef
norm = args.norm
weight_norm = args.weight_norm
# Save settings
save_addr = args.save_addr
save_name = args.save_name
noise_m = args.noise_m
suffix = '' if noise_m==0 else '_noise%d-%02d'%(int(noise_m), int((noise_m-int(noise_m))*100))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# =                                    Utils                                   =
# ==============================================================================

class Logger(object):
    def __init__(self, fileN="Default.txt"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def loadData(filename):
    with open(filename, 'rb') as file:
        print('Open file', filename, '......ok')
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

def save_checkpoint(obj, save_path, is_best=False, max_keep=None):
    # save checkpoint
    torch.save(obj, save_path)

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    save_name = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_name + '\n'] + ckpt_list
    else:
        ckpt_list = [save_name + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))

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

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.CenterCrop(64),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
    dataset_dir, dataset_name = os.path.split(dataset_path)
    saveData(dataset_dir, dataset_name+'_dat', dataset)

    return dataset

def test_original(net, test_loader):
    net.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx >= 300: break
    acc = 100. * correct / total

    print('>> Test\nOriginal accuracy on REAL %.3f%%' % (acc))

    return acc

def test_noised(net, test_loader, noise_m):
    net.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            noise = torch.randn(outputs.shape).to(device)
            outputs = outputs + noise_m * noise

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx >= 300: break
    acc = 100. * correct / total

    print('>> Test\nNoised accuracy on REAL %.3f%%' % (acc))

    return acc


# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================


# Check save address
logger_addr = '/data/gehan/PytorchProjects/GANInference/models/attackers/SVHN_outputs'
if not os.path.exists(logger_addr):
    os.makedirs(logger_addr)
sys.stdout = Logger("%s/%sTestLog"% (logger_addr, save_name) + suffix + ".txt")

# List related GANs
assert os.path.exists(gan_addr), 'Wrong GAN address!'
filelist = os.listdir(gan_addr)
gan_list = []
for file in filelist:
    if file.startswith(gan_name):# if gan_name in file:
        gan_list.append(file)
assert gan_list, 'No GAN saved in current address!'

# data
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(64, 64), interpolation=PIL.Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)

testset_real = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform, split='test')
test_real_loader = torch.utils.data.DataLoader(
    dataset=testset_real,
    batch_size=10,
    shuffle=True,
    num_workers=2,
    drop_last=True
)

acc_orig_list = []
acc_nois_list = []
for gan in gan_list:
    model_gan_dir = os.path.join(save_addr, gan)

    if not os.path.exists(model_gan_dir): continue
    model_gan = os.listdir(model_gan_dir)
    model_list = []
    for model in model_gan:
        contents = os.listdir(os.path.join(model_gan_dir, model))
        if 'TrainSet_dat' in contents:
            continue
        if model.startswith(save_name):
            model_list.append(model)
    assert model_list, 'No usable model saved for current GAN: %s!' % gan

    for model in model_list:

        print("\nGAN: %s, Model: %s" % (gan, model))

        # Directory to save model
        model_dir = os.path.join(os.path.join(save_addr, gan), model)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        # Building model
        if model.startswith('LeN'):
            net = structures.lenet.LeNet(c_dim=c_dim).to(device)
        elif model.startswith('Smp'):
            net = structures.lenet.SmplCNN(c_dim=c_dim).to(device)
        elif model.startswith('VGG'):
            net = structures.vgg.VGG('VGG11', c_dim=c_dim).to(device)
        elif model.startswith('Res'):
            net = structures.resnet.ResNet18(c_dim=c_dim).to(device)
        else:
            continue

        # Load checkpoint
        model_ckpt_dir = os.path.join(model_dir, 'checkpoints')
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
                continue

        with torch.no_grad():
            acc_orig = test_original(net, test_real_loader)
            acc_orig_list.append(acc_orig)
            acc_nois = test_noised(net, test_real_loader, noise_m)
            acc_nois_list.append(acc_nois)

acc_orig_list.append(acc_orig)
acc_nois_list.append(acc_nois)
print('Original averaged acc is: %.3f%%' % mean(acc_orig_list))
print('Noised averaged acc is: %.3f%%' % mean(acc_nois_list))

