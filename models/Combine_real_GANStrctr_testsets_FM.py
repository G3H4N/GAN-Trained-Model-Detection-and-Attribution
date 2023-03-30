
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from numpy import *
import os
import argparse
import pickle
import sys
import PIL.Image as Image
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
                    default=10)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=100)
parser.add_argument('--trainset_size_perclass', dest='trainset_size_perclass', type=int, help='size of train set',
                    default=10000)
parser.add_argument('--testset_gen_size_perclass', dest='testset_gen_size_perclass', type=int, help='size of generated test set',
                    default=100)
parser.add_argument('--c_dim', dest='c_dim', type=int, help='number of classes',
                    default=10)
parser.add_argument('--num_models', dest='num_models', type=int, help='number of models to be trained for each GAN',
                    default=5)
# GANs
parser.add_argument('--gan_addr', dest='gan_addr', type=str, help='gan address',
                    default='/data/gehan/PytorchProjects/GANInference/GANs/Conditional-GANs-Pytorch-master/FM_outputs/')
parser.add_argument('--gan_name', dest='gan_name', type=str, help='gan name',
                    default='')
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
# Real data for testing
parser.add_argument('--test_real_path', dest='test_real_path', type=str, help='real test data',
                    default='/data/gehan/Datasets/CelebA/png/align/CelebA_png_align_Attr_22_21_20_sets/TestSet')
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/FM_outputs/')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='')
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
# Real data for testing
test_real_path = args.test_real_path
# Save settings
save_addr = args.save_addr
save_name = args.save_name

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

        for n in range(dataset_size_perclass):
            input_noise = torch.randn(c_dim, z_dim).to(device)
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

def train(net, net_optimizer, train_loader, epoch):
    print('\n>> TRAIN\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs, targets = inputs.to(device), targets.to(device)
        net_optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        net_optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('batch_idx %d, len(trainloader) %d, Loss: %.6f | Acc: %.3f%% (%d/%d)'
                     % (batch_idx, len(train_loader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return net

def test(net, test_gen_loader, test_real_loader):
    #global best_acc
    net.eval()

    test_loss_gen = 0
    correct_gen = 0
    total_gen = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_gen_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss_gen += loss.item()
            _, predicted = outputs.max(1)
            total_gen += targets.size(0)
            correct_gen += predicted.eq(targets).sum().item()
    acc_gen = 100. * correct_gen / total_gen

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_real_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total

    print('>> Test\nAccuracy on FAKE: %.3f%%, accuracy on REAL %.3f%%'
          % (acc_gen, acc))
    
    return acc

def test_gen(net, test_gen_loader):
    net.eval()

    correct_gen = 0
    total_gen = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_gen_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total_gen += targets.size(0)
            correct_gen += predicted.eq(targets).sum().item()
    acc_gen = 100. * correct_gen / total_gen

    return acc_gen


# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

# Check save address
logger_addr = save_addr#os.path.join(save_addr, gan_name)
if not os.path.exists(logger_addr):
    os.makedirs(logger_addr)
sys.stdout = Logger("%s/%sDatasetGenerationLog_real_multiclass.txt" % (logger_addr, save_name))

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(64, 64), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)
FMNIST = torchvision.datasets.FashionMNIST('/data/gehan/Datasets/FashionMNIST/', train=False, download=True, transform=transform)
FMNIST_indices = loadData('/data/gehan/Datasets/FashionMNIST/ProbeSet_indices')
FMNIST_probset = torch.utils.data.Subset(FMNIST, FMNIST_indices)
'''
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(64, 64), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)

SVHN_test = torchvision.datasets.SVHN('/data/gehan/Datasets/SVHN', download=True, transform=transform, split='test')
SVHN_indices = loadData('/data/gehan/Datasets/SVHN/ProbeSet_indices')
SVHN_probset = torch.utils.data.Subset(SVHN_test, SVHN_indices)
'''

CelebA_prob_path = '/data/gehan/Datasets/CelebA/png/align/CelebA_png_align_Attr_22_21_20_sets/ProbeSet'
CelebA_probset = loadData(CelebA_prob_path)

probset = torch.utils.data.ConcatDataset([CelebA_probset, FMNIST_probset])
prob_loader = torch.utils.data.DataLoader(probset, batch_size=args.batch_size, shuffle=False)

GAN_classes_to_index_tensor = {'CGAN_gan_0':torch.tensor(0), 'CGAN_gan_1':torch.tensor(1), 'CGAN_gan_2':torch.tensor(2), 'CGAN_gan_3':torch.tensor(3), 'CGAN_gan_4':torch.tensor(4), 'CGAN_gan_5':torch.tensor(5), 'CGAN_gan_6':torch.tensor(6), 'CGAN_gan_7':torch.tensor(7), 'CGAN_gan_8':torch.tensor(8), 'CGAN_gan_9':torch.tensor(9),
                    'ACGAN_hinge2_0':torch.tensor(10), 'ACGAN_hinge2_1':torch.tensor(11), 'ACGAN_hinge2_2':torch.tensor(12), 'ACGAN_hinge2_3':torch.tensor(13), 'ACGAN_hinge2_4':torch.tensor(14), 'ACGAN_hinge2_5':torch.tensor(15), 'ACGAN_hinge2_6':torch.tensor(16), 'ACGAN_hinge2_7':torch.tensor(17), 'ACGAN_hinge2_8':torch.tensor(18), 'ACGAN_hinge2_9':torch.tensor(19),
                    'DCGAN_gan_lr__0':torch.tensor(20), 'DCGAN_gan_lr__1':torch.tensor(21), 'DCGAN_gan_lr__2':torch.tensor(22), 'DCGAN_gan_lr__3':torch.tensor(23), 'DCGAN_gan_lr_4':torch.tensor(24), 'DCGAN_gan_lr_5':torch.tensor(25), 'DCGAN_gan_lr_6':torch.tensor(26), 'DCGAN_gan_lr_7':torch.tensor(27), 'DCGAN_gan_lr_8':torch.tensor(28), 'DCGAN_gan_lr_9':torch.tensor(29),
                    'DCGAN_wgan-gp_0':torch.tensor(30), 'DCGAN_wgan-gp_1':torch.tensor(31), 'DCGAN_wgan-gp_2':torch.tensor(32), 'DCGAN_wgan-gp_3':torch.tensor(33), 'DCGAN_wgan-gp_4':torch.tensor(34), 'DCGAN_wgan-gp_5':torch.tensor(35), 'DCGAN_wgan-gp_6':torch.tensor(36), 'DCGAN_wgan-gp_7':torch.tensor(37), 'DCGAN_wgan-gp_8':torch.tensor(38), 'DCGAN_wgan-gp_9':torch.tensor(39)}
Strctr_classes_to_index_tensor = {'CGAN_gan':torch.tensor(0), 'ACGAN_hinge2':torch.tensor(1), 'DCGAN_gan':torch.tensor(2), 'DCGAN_wgan-gp':torch.tensor(3)}

# Generate train data
dataset_strctr_path_list = []
for strctr in Strctr_classes_to_index_tensor:

    # Initiate the Testset of current GAN-strctr
    posterior_sorted = torch.tensor([]).to(device)
    #posterior_label = torch.tensor([]).to(device)  # un-sorted
    #posterior_correct = torch.tensor([]).to(device)  # sorted
    #GAN_name = torch.tensor([]).to(device)
    Strctr_name = torch.tensor([]).to(device)

    # List related GANs
    assert os.path.exists(save_addr), 'Wrong address!'
    filelist = os.listdir(save_addr)
    GANs_list = []
    for file in filelist:
        if os.path.isdir(os.path.join(save_addr, file)):
            if file.startswith(strctr):
                if file.endswith('_'):
                    strctr_dataset_save_dir = os.path.join(save_addr, file)
                    dataset_strctr_path_list.append(strctr_dataset_save_dir)
                elif (not file.endswith('_')) and (not file.endswith('T')):
                    GANs_list.append(file)
    assert GANs_list, 'No GAN saved in current address!'

    # ====== Testset generation of current GAN structure ====== #

    # Non-targeted GANs
    for gan in GANs_list:

        # List models trained on current GAN
        model_gan_dir = os.path.join(save_addr, gan)
        assert os.path.exists(model_gan_dir), 'Wrong address!'
        model_gan = os.listdir(model_gan_dir)
        model_list = []
        for model in model_gan:
            contents = os.listdir(os.path.join(model_gan_dir, model))
            if 'TrainSet_dat' in contents:
                continue
            if model.startswith('Ale') or model.startswith('Goo'):
                continue
            if model.endswith('4'):
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

            # Query the model by probing set
            information0 = torch.tensor([]).to(device)
            #information1 = torch.tensor([]).to(device)
            #information2 = torch.tensor([]).to(device)
            with torch.no_grad():
                for batch_idx, (inputs, _) in enumerate(prob_loader):

                    inputs = inputs.to(device)
                    outputs = net(inputs)
                    outputs_sorted = outputs.sort(dim=1)[0]

                    information0 = torch.cat((information0, outputs_sorted[:, :8]), 0)
                information0 = information0.view(1, -1)
                # update dataset
                posterior_sorted = torch.cat((posterior_sorted, information0), 0)

                strctr_name = Strctr_classes_to_index_tensor[strctr]
                strctr_name = strctr_name.repeat(len(information0),1).to(device)
                Strctr_name = torch.cat((Strctr_name, strctr_name), 0)


            #break# since current tested model
        #break# since gan in a strctr
    TestSet_strctr = torch.utils.data.TensorDataset(posterior_sorted, Strctr_name)
    saveData(strctr_dataset_save_dir, 'Combine_real_GANStrctr_Testset', TestSet_strctr)
    print(len(TestSet_strctr))
    del TestSet_strctr


# Combine datasets among all structures
trainsets_strctr_list = []
for data_path_strctr in dataset_strctr_path_list:
    trainset = loadData(os.path.join(data_path_strctr, 'Combine_real_GANStrctr_Testset'))
    trainsets_strctr_list.append(trainset)

TestSet_AllStrctr = torch.utils.data.ConcatDataset(trainsets_strctr_list)
saveData(save_addr, 'Combine_real_GANStrctr_Testset_AllStrctr', TestSet_AllStrctr)
print(len(TestSet_AllStrctr))
