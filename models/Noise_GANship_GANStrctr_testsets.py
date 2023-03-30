
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
#parser.add_argument('--c_dim', dest='c_dim', type=int, help='number of classes', default=8)
parser.add_argument('--num_models', dest='num_models', type=int, help='number of models to be trained for each GAN',
                    default=5)
# GANs
parser.add_argument('--gan_addr', dest='gan_addr', type=str, help='gan address',
                    default='/data/gehan/PytorchProjects/GANInference/GANs/Conditional-GANs-Pytorch-master/')
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
                    default='/data/gehan/PytorchProjects/GANInference/models/')
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
#c_dim = args.c_dim
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

prob_path = '/data/gehan/PytorchProjects/GANInference/models/attackers/Datasets/ProbeSet_noise'
probset = loadData(prob_path)
prob_loader = torch.utils.data.DataLoader(probset, batch_size=args.batch_size, shuffle=False)
'''
real_prob_path = '/data/gehan/Datasets/CelebA/png/align/CelebA_png_align_Attr_22_21_20_sets/ProbeSet'
real_probset = loadData(real_prob_path)
real_prob_loader = torch.utils.data.DataLoader(real_probset, batch_size=args.batch_size, shuffle=False)
for i, (data, label) in enumerate(real_prob_loader):
    break
'''
GAN_classes_to_index_tensor = {'CelebA_CGAN':torch.tensor(1), 'CelebA_ACGAN':torch.tensor(2), 'CelebA_DCGAN':torch.tensor(3), 'CelebA_WGAN':torch.tensor(4), 'CelebA_Real':torch.tensor(0),
                               'FM_CGAN':torch.tensor(5), 'FM_ACGAN':torch.tensor(6), 'FM_DCGAN':torch.tensor(7), 'FM_WGAN':torch.tensor(8), 'FM_Real':torch.tensor(0),
                               'SVHN_CGAN':torch.tensor(9), 'SVHN_ACGAN':torch.tensor(10), 'SVHN_DCGAN':torch.tensor(11), 'SVHN_WGAN':torch.tensor(12), 'SVHN_Real':torch.tensor(0)}
GAN_classes_to_path = {'CelebA_CGAN':'outputs/CGAN_gan_lr_1', 'CelebA_ACGAN':'outputs/ACGAN_hinge2_0', 'CelebA_DCGAN':'outputs/DCGAN_gan_1', 'CelebA_WGAN':'outputs/DCGAN_wgan-gp_4', 'CelebA_Real':'outputs/Real',
                       'FM_CGAN':'FM_outputs/CGAN_gan_0', 'FM_ACGAN':'FM_outputs/ACGAN_hinge2_0', 'FM_DCGAN':'FM_outputs/DCGAN_gan_lr_4', 'FM_WGAN':'FM_outputs/DCGAN_wgan-gp_0', 'FM_Real':'FM_outputs/Real',
                       'SVHN_CGAN':'SVHN_outputs/CGAN_gan_0', 'SVHN_ACGAN':'SVHN_outputs/ACGAN_hinge2_lr_1', 'SVHN_DCGAN':'SVHN_outputs/DCGAN_gan_4', 'SVHN_WGAN':'SVHN_outputs/DCGAN_wgan-gp_0', 'SVHN_Real':'SVHN_outputs/Real'}
Strctr_classes_to_index_tensor = {'CGAN_gan':torch.tensor(0), 'ACGAN_hinge2':torch.tensor(1), 'DCGAN_gan':torch.tensor(2), 'DCGAN_wgan-gp':torch.tensor(3), 'Real':torch.tensor(4)}

data_byGANstr = {'CGAN_gan':[torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)], 'ACGAN_hinge2':[torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)], 'DCGAN_gan':[torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)], 'DCGAN_wgan-gp':[torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)], 'Real':[torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)]}
data_byMstr = {'Smp':[torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)], 'LeN':[torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)], 'VGG':[torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)], 'Res':[torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)]}

# Generate train data
dataset_save_addr = os.path.join(save_addr, '_datasets_noise_selectedGAN')
if not os.path.exists(dataset_save_addr):
    os.makedirs(dataset_save_addr)

# Generate train data
for gan in GAN_classes_to_path:

    # Initiate the Testset of current GAN
    posterior_sorted = torch.tensor([]).to(device)
    GAN_name = torch.tensor([]).to(device)
    Strctr_name = torch.tensor([]).to(device)

    gan_directory, gan_name = os.path.split(GAN_classes_to_path[gan])

    # c_dim related to dataset
    if gan_directory.startswith('outputs'):
        c_dim = 8
    else:
        c_dim = 10

    # GAN structure
    for GANstr in Strctr_classes_to_index_tensor:
        if gan_name.startswith(GANstr):
            break

    # ====== Testset generation of selected GAN ====== #

    # List models trained on current GAN
    model_gan_dir = os.path.join(save_addr, GAN_classes_to_path[gan])
    assert os.path.exists(model_gan_dir), 'Wrong gan address!'
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
        model_dir = os.path.join(model_gan_dir, model)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        # Building model
        if model.startswith('LeN'):
            Mstr = 'LeN'
            net = structures.lenet.LeNet(c_dim=c_dim).to(device)
        elif model.startswith('Smp'):
            Mstr = 'Smp'
            net = structures.lenet.SmplCNN(c_dim=c_dim).to(device)
        elif model.startswith('VGG'):
            Mstr = 'VGG'
            net = structures.vgg.VGG('VGG11', c_dim=c_dim).to(device)
        elif model.startswith('Res'):
            Mstr = 'Res'
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
        with torch.no_grad():
            for batch_idx, (inputs) in enumerate(prob_loader):

                inputs = inputs.to(device)
                outputs = net(inputs)
                outputs_sorted = outputs.sort(dim=1)[0]

                information0 = torch.cat((information0, outputs_sorted[:, :8]), 0)

            information0 = information0.view(1, -1)

            # update dataset
            posterior_sorted = torch.cat((posterior_sorted, information0), 0)
            data_byMstr[Mstr][0] = torch.cat((data_byMstr[Mstr][0], information0), 0)

            gan_name = GAN_classes_to_index_tensor[gan]
            gan_name = gan_name.repeat(len(information0), 1).to(device)
            GAN_name = torch.cat((GAN_name, gan_name), 0)
            data_byMstr[Mstr][1] = torch.cat((data_byMstr[Mstr][1], gan_name), 0)

            strctr_name = Strctr_classes_to_index_tensor[GANstr]
            strctr_name = strctr_name.repeat(len(information0), 1).to(device)
            Strctr_name = torch.cat((Strctr_name, strctr_name), 0)
            data_byMstr[Mstr][2] = torch.cat((data_byMstr[Mstr][2], strctr_name), 0)
    # After all models, update current gan data to the All data
    data_byGANstr[GANstr][0] = torch.cat((data_byGANstr[GANstr][0], posterior_sorted), 0)
    data_byGANstr[GANstr][1] = torch.cat((data_byGANstr[GANstr][1], GAN_name), 0)
    data_byGANstr[GANstr][2] = torch.cat((data_byGANstr[GANstr][2], Strctr_name), 0)

# Save by Mstr
for i in data_byMstr:
    dataset = torch.utils.data.TensorDataset(data_byMstr[i][0], data_byMstr[i][1], data_byMstr[i][2])
    saveData(dataset_save_addr, 'noise_Testset_'+i, dataset)
del data_byMstr

# Save by GANstr
for i in data_byGANstr:
    dataset = torch.utils.data.TensorDataset(data_byGANstr[i][0], data_byGANstr[i][1], data_byGANstr[i][2])
    saveData(dataset_save_addr, 'noise_Testset_'+i, dataset)

# GAN_name
information = torch.tensor([]).to(device)
target = torch.tensor([]).to(device)
for i in data_byGANstr:
    if i == 'Real':
        information = torch.cat((information, data_byGANstr[i][0]), 0)
        target = torch.cat((target, data_byGANstr[i][1]), 0)
    else:
        for a in range(3):
            information = torch.cat((information, data_byGANstr[i][0]), 0)
            target = torch.cat((target, data_byGANstr[i][1]), 0)

Testset_All = torch.utils.data.TensorDataset(information, target)
saveData(dataset_save_addr, 'noise_GANship_Testset_All', Testset_All)
print(len(Testset_All))
del information, target, Testset_All

# GAN_Strctr
information = torch.tensor([]).to(device)
target = torch.tensor([]).to(device)
for i in data_byGANstr:
    information = torch.cat((information, data_byGANstr[i][0]), 0)
    target = torch.cat((target, data_byGANstr[i][2]), 0)

Testset_All = torch.utils.data.TensorDataset(information, target)
saveData(dataset_save_addr, 'noise_GANStrctr_Testset_All', Testset_All)
print(len(Testset_All))
del information, target, Testset_All,

del data_byGANstr

