
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
parser.add_argument('--c_dim', dest='c_dim', type=int, help='number of classes',
                    default=10)
parser.add_argument('--num_models', dest='num_models', type=int, help='number of models to be trained for each GAN',
                    default=5)
# GANs
parser.add_argument('--gan_addr', dest='gan_addr', type=str, help='gan address',
                    default='/data/gehan/PytorchProjects/GANInference/GANs/Conditional-GANs-Pytorch-master/CIFAR10_outputs/')
parser.add_argument('--z_dim', dest='z_dim', type=int, help='dimension of input noise for GAN',
                    default=100)
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/CIFAR10_outputs/')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='')
args = parser.parse_args()

# Train settings
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
c_dim = args.c_dim
num_models = args.num_models
# GANs
gan_addr = args.gan_addr
z_dim = args.z_dim
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


# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

# Check save address
logger_addr = save_addr
if not os.path.exists(logger_addr):
    os.makedirs(logger_addr)
sys.stdout = Logger("%s/%sDatasetGenerationLog_real_multiclass.txt" % (logger_addr, save_name))

transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(size=(64, 64), interpolation=Image.BICUBIC),
     torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)

CIFAR10_test = torchvision.datasets.CIFAR10(root='/data/gehan/Datasets/CIFAR10/', train=False, download=True, transform=transform)
CIFAR10_indices = loadData('/data/gehan/Datasets/CIFAR10/ProbeSet_indices')
probset = torch.utils.data.Subset(CIFAR10_test, CIFAR10_indices)

prob_loader = torch.utils.data.DataLoader(probset, batch_size=args.batch_size, shuffle=False)

GAN_classes_to_index_tensor = {'CGAN_gan_3':torch.tensor(0), 'ACGAN_hinge2_2':torch.tensor(1), 'DCGAN_gan_0':torch.tensor(2), 'DCGAN_wgan-gp_1':torch.tensor(3),
                               'SAGAN':torch.tensor(4), 'BigGAN_CR':torch.tensor(5), 'ContraGAN':torch.tensor(6), 'StyleGAN2_ada':torch.tensor(7),
                               'BDGAN_00_deer_25':torch.tensor(8), 'Real':torch.tensor(9)}

# Generate Test data
dataset_strctr_path_list = []
for strctr in GAN_classes_to_index_tensor:

    # Initiate the Testset of current GAN-strctr
    posterior_sorted = torch.tensor([]).to(device)
    posterior_label = torch.tensor([]).to(device)  # un-sorted
    posterior_correct = torch.tensor([]).to(device)  # sorted
    GAN_name = torch.tensor([]).to(device)
    Strctr_name = torch.tensor([]).to(device)

    # List related GANs
    assert os.path.exists(save_addr), 'Wrong address!'
    filelist = os.listdir(save_addr)
    GANs_list = []
    for file in filelist:
        if os.path.isdir(os.path.join(save_addr, file)):
            if file.startswith(strctr[:-1]):
                if file.endswith('_'):
                    strctr_dataset_save_dir = os.path.join(save_addr, file)
                    dataset_strctr_path_list.append(strctr_dataset_save_dir)
                else:
                    GANs_list.append(file)
    assert GANs_list, 'No GAN saved in current address!'

    # ====== Testset generation of current GAN structure ====== #

    # Non-targeted GANs
    for gan in GANs_list:

        # List models Tested on current GAN
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
            if (model.endswith('8')) or (model.endswith('9')):
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
            information1 = torch.tensor([]).to(device)
            information2 = torch.tensor([]).to(device)
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(prob_loader):

                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)

                    # one-hot true labels (generated by GAN)
                    labels = torch.zeros(outputs.shape).to(device)
                    for i in range(len(targets)):
                        labels[i][targets[i]] = 1

                    # correctness of classification
                    _, predicted = outputs.max(1)
                    predicted = predicted.to(device)
                    corrects = predicted.eq(targets).float().unsqueeze(-1)

                    outputs_sorted = outputs.sort(dim=1)[0]

                    # concatenate UN-SORTED posterior and true label
                    post_l = torch.cat([outputs, labels], 1)
                    # concatenate SORTED posterior and correct bit
                    post_c = torch.cat([outputs_sorted, corrects], 1)
                    information0 = torch.cat((information0, outputs_sorted), 0)
                    information1 = torch.cat((information1, post_l), 0)
                    information2 = torch.cat((information2, post_c), 0)
                information0 = information0.view(1, -1)
                information1 = information1.view(1, -1)
                information2 = information2.view(1, -1)
                # update dataset
                posterior_sorted = torch.cat((posterior_sorted, information0), 0)
                posterior_label = torch.cat((posterior_label, information1), 0)
                posterior_correct = torch.cat((posterior_correct, information2), 0)
                gan_name = GAN_classes_to_index_tensor[gan]
                gan_name = gan_name.repeat(len(information0), 1).to(device)
                GAN_name = torch.cat((GAN_name, gan_name), 0)


            #break# since current tested model
        #break# since gan in a strctr
    Testset_strctr = torch.utils.data.TensorDataset(posterior_sorted, posterior_label, posterior_correct, GAN_name)
    saveData(strctr_dataset_save_dir, 'real_GANship_backdoor_Testset_25', Testset_strctr)
    print(len(Testset_strctr))
    del Testset_strctr


# Combine datasets among all structures
Testsets_strctr_list = []
for data_path_strctr in dataset_strctr_path_list:
    Testset = loadData(os.path.join(data_path_strctr, 'real_GANship_backdoor_Testset_25'))
    Testsets_strctr_list.append(Testset)

Testset_AllStrctr = torch.utils.data.ConcatDataset(Testsets_strctr_list)
saveData(save_addr, 'real_GANship_backdoor_Testset_25_AllStrctr', Testset_AllStrctr)
print(len(Testset_AllStrctr))
