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
                    default='/home/c01geha/PytorchProjects/GANInference/GANs/Conditional-GANs-Pytorch-master/SVHN_outputs')
parser.add_argument('--gan_name', dest='gan_name', type=str, help='gan name',
                    default='DCGAN_gan_')# DCGAN_wgan-gp_  DCGAN_gan_  ACGAN_hinge2_lr_  CGAN_gan_
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
'''
# Real data for testing
parser.add_argument('--test_real_path', dest='test_real_path', type=str, help='real test data',
                    default='/home/c01geha/Datasets/CelebA_png_align_Attr_22_21_20_sets/TestSet')
'''
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='/home/c01geha/PytorchProjects/GANInference/models/SVHN_outputs/')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='SmplCNN_')
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
#test_real_path = args.test_real_path
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


# Training
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
        _, predicted = outputs.max(1)#torch.round(outputs)
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


# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================


# Check save address
logger_addr = os.path.join(save_addr, gan_name)
if not os.path.exists(logger_addr):
    os.makedirs(logger_addr)
sys.stdout = Logger("%s/%sTrainLog.txt" % (logger_addr, save_name))

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
     #torchvision.transforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     torchvision.transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)

testset_real = torchvision.datasets.SVHN('/home/c01geha/Datasets/SVHN', download=True, transform=transform, split='test')
test_real_loader = torch.utils.data.DataLoader(
    dataset=testset_real,
    batch_size=2,
    shuffle=True,
    num_workers=2,
    drop_last=True
)
#classes = ('000', '001', '010', '011', '100', '101', '110', '111')

for gan in gan_list:
    gan_path = os.path.join(gan_addr, gan)

    # Prepare generated test set
    testset_gen_path = os.path.join(gan_path, 'TestSet')
    if not os.path.exists(testset_gen_path+'_dat'):
        testset_gen = GenerateDataSet(gan_path, z_dim, c_dim, testset_gen_size_perclass, testset_gen_path)
    else:
        testset_gen = loadData(testset_gen_path+'_dat')

    # Train models
    for i in range(num_models):

        # Directory to save model
        model_dir = os.path.join(os.path.join(save_addr, gan), save_name + '%d/' % (i))
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        # Building model
        net = structures.lenet.SmplCNN(c_dim=c_dim).to(device)

        criterion = nn.CrossEntropyLoss()
        net_optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=5e-4)

        # Load checkpoint
        model_ckpt_dir = os.path.join(model_dir, 'checkpoints')
        trainset_path = os.path.join(model_dir, 'TrainSet')

        try:
            ckpt = load_checkpoint(model_ckpt_dir)
            start_epoch = ckpt['epoch']
            if start_epoch != epochs:
                if not os.path.exists(trainset_path + '_dat'):
                    trainset = GenerateDataSet(gan_path, z_dim, c_dim, trainset_size_perclass, trainset_path)
                    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

                    print('Train from beginning.')
                    start_epoch = 0
                    best_acc = 0
                else:
                    trainset = loadData(trainset_path + '_dat')
                    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

                    try:
                        ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
                        best_acc = ckpt_best['acc']

                        start_epoch = ckpt_best['epoch']
                        net.load_state_dict(ckpt_best['net'])
                        net_optimizer.load_state_dict(ckpt_best['optimizer'])
                    except:
                        best_acc = ckpt['acc']

                        start_epoch = ckpt['epoch']
                        net.load_state_dict(ckpt['net'])
                        net_optimizer.load_state_dict(ckpt['optimizer'])
            else:
                net.load_state_dict(ckpt['net'])
                net_optimizer.load_state_dict(ckpt['optimizer'])
                best_acc = ckpt['acc']
        except:
            print(' [*] No checkpoint!\nTrain from beginning.')
            start_epoch = 0
            best_acc = 0
            try:
                trainset = loadData(trainset_path + '_dat')
                train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

            except:
                trainset = GenerateDataSet(gan_path, z_dim, c_dim, trainset_size_perclass, trainset_path)
                train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

        # Train one model

        # Data
        test_gen_loader = torch.utils.data.DataLoader(testset_gen, batch_size=args.batch_size, shuffle=False)

        ep = start_epoch
        for ep in range(start_epoch, epochs):
            train(net, net_optimizer, train_loader, ep)
            acc = test(net, test_gen_loader, test_real_loader)

            # Save checkpoint for best accuracy
            if acc > best_acc:
                print('Best accuracy updated from %.3f%% to %.3f%%.' % (best_acc, acc))
                best_acc = acc
                print('Saving...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': ep+1,
                    'optimizer': net_optimizer.state_dict()
                }
                if not os.path.isdir(model_ckpt_dir):
                    os.mkdir(model_ckpt_dir)
                save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep + 1),
                                      max_keep=2, is_best=True)
            else:
                # Save checkpoint every 10 epochs
                if ep % 10 == 0:
                    print('Saving...')
                    state = {
                        'net': net.state_dict(),
                        'acc': acc,
                        'epoch': ep+1,
                        'optimizer': net_optimizer.state_dict()
                    }
                    if not os.path.isdir(model_ckpt_dir):
                        os.mkdir(model_ckpt_dir)
                    save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep + 1),
                                          max_keep=2, is_best=False)

        acc = test(net, test_gen_loader, test_real_loader)
        print('Saving...')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': ep + 1,
            'optimizer': net_optimizer.state_dict()
        }
        if not os.path.isdir(model_ckpt_dir):
            os.mkdir(model_ckpt_dir)
        save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep + 1),
                        max_keep=2, is_best=False)

        try:
            os.remove(trainset_path + '_dat')
            shutil.rmtree(trainset_path)
            print("Train data removed.")
        except:
            print('No dataset to remove.')

