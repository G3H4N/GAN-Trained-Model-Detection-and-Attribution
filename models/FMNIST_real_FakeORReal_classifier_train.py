'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import os
import argparse
import pickle
import sys
import shutil
import structures
import GAN_structures
from sklearn.metrics import roc_auc_score

# ==============================================================================
# =                                  Settings                                  =
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
# Train setting
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.00001)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=20)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=600)
parser.add_argument('--model_addr', dest='model_addr', type=str, help='target and shadow model save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/FM_outputs/')
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/attackers/FM_outputs/')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='real_FakeORReal_withReal_balanced')
args = parser.parse_args()

# Train settings
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
model_addr = args.model_addr
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
    #print(' [*] Loading checkpoint succeeds! Copy variables from % s!' % ckpt_path)
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
    #D = GAN_structures.DiscriminatorDCGAN(x_dim=3, c_dim=c_dim, norm=norm, weight_norm=weight_norm).to(device)
    #G = GAN_structures.GeneratorDCGAN(nch_in=z_dim, n_class=c_dim).to(device)
    G = GAN_structures.GeneratorACGAN(z_dim=z_dim, c_dim=c_dim).to(device)
    #d_loss_fn, g_loss_fn = GAN_structures.get_losses_fn(loss_mode)
    #d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
    #g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))

    # Load GAN
    assert os.path.exists(gan_path), 'Wrong directory for loading GAN!'
    ckpt_dir = os.path.join(gan_path, 'checkpoints')
    ckpt = load_checkpoint(ckpt_dir)
    #start_ep = ckpt['epoch']
    #D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    #d_optimizer.load_state_dict(ckpt['d_optimizer'])
    #g_optimizer.load_state_dict(ckpt['g_optimizer'])

    # Make directories
    for i in range(c_dim):
        path = '%s/%d/' % (dataset_path, i)
        if not os.path.exists(path):
            os.makedirs(path)

    #init_weights(netG)
    '''
    def model_load(dir_chck, netG, epoch=[]):
        if epoch == -1:
            ckpt = os.listdir(dir_chck)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        dict_net = torch.load('%s/model_epoch%04d.pth' % (dir_chck, epoch))

        print('Loaded %dth network' % epoch)

        netG.load_state_dict(dict_net['netG'])

        return netG, epoch

    netG, st_epoch = model_load(model_dir, netG, load_epoch)
    '''
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
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print('batch_idx %d, len(trainloader) %d, Loss: %.6f | Acc: %.5f%% (%d/%d)'
                     % (batch_idx, len(train_loader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return net

def test(net, i_index, t_index, test_loader):
    # global best_acc
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data[i_index].to(device)
            targets = data[t_index].to(device)
            targets = targets.squeeze(-1)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total

    print('>> Test\nAccuracy: %.5f%%' % (acc))

    return acc

def test_binary(net, i_index, t_index, test_loader):
    # global best_acc
    net.eval()

    test_loss = 0
    outputs_test = []
    members_test = []
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data[i_index].to(device)
            targets = data[t_index].to(device)

            outputs = net(inputs)
            outputs = m(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            outputs_test.extend(outputs)
            members_test.extend(targets)

    auc = roc_auc_score(torch.BoolTensor(members_test), outputs_test)

    print('>> Test\nAUC: %.5f' % (auc))

    return auc


# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

# Save print log
logger_addr = os.path.join(save_addr, save_name)
if not os.path.exists(logger_addr):
    os.makedirs(logger_addr)
sys.stdout = Logger("%s/TrainLog.txt" % (logger_addr))

# several lists for testing phase
targetGAN_list = []
All_GAN_list = []
assert os.path.exists(model_addr), 'Wrong model address!'
filelist = os.listdir(model_addr)
for file in filelist:
    if os.path.isdir(os.path.join(model_addr, file)):
        if not file.endswith('_'):
            All_GAN_list.append(file)
            if file.endswith('T'):
                targetGAN_list.append(file)

# Datasets preparation
trainset_path = os.path.join(model_addr, 'real_FakeORReal_Trainset_AllStrctr')
trainset = loadData(trainset_path)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
testset_path = os.path.join(model_addr, 'real_FakeORReal_Testset_AllStrctr')
testset = loadData(testset_path)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

content_list = ['posterior_sorted', 'posterior_label', 'posterior_correct', 'Fake']
information_list = ['posterior_sorted', 'posterior_label', 'posterior_correct']
target_list = ['Fake']

# ====== Train and Test a model for each of information ====== #
for i in information_list:

    information_index = content_list.index(i)

    for t in target_list:
        print('\n--* %s --> %s *--' % (i, t))
        target_index = content_list.index(t)

        # Directory to save model
        model_dir = os.path.join(os.path.join(save_addr, save_name), i)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if i == 'posterior_sorted':
            # Building model
            net = structures.lenet.attacker_real(nch_info=1000, nch_output=1).to(device)
        elif i == 'posterior_label':
            # Building model
            net = structures.lenet.attacker_real(nch_info=2000, nch_output=1).to(device)
        elif i == 'posterior_correct':
            # Building model
            net = structures.lenet.attacker_real(nch_info=1100, nch_output=1).to(device)

        criterion = nn.BCELoss()
        m = nn.Sigmoid()
        net_optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=5e-4)

        # Load checkpoint
        model_ckpt_dir = os.path.join(model_dir, 'checkpoints')

        try:
            ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
            start_epoch = ckpt_best['epoch']
            best_acc = ckpt_best['acc']
            net.load_state_dict(ckpt_best['net'])
            net_optimizer.load_state_dict(ckpt_best['optimizer'])
        except:
            try:
                ckpt = load_checkpoint(model_ckpt_dir, load_best=False)
                start_epoch = ckpt['epoch']
                best_acc = ckpt['acc']
                net.load_state_dict(ckpt['net'])
                net_optimizer.load_state_dict(ckpt['optimizer'])
            except:
                print(' [*] No checkpoint!\nTrain from beginning.')
                start_epoch = 0
                best_acc = 0

        for ep in range(start_epoch, epochs):

            # TRAIN
            print('\n>> TRAIN Epoch: %d' % ep)
            net.train()

            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, data in enumerate(train_loader):
                inputs = data[information_index].to(device)
                targets = data[target_index].to(device)

                net_optimizer.zero_grad()
                outputs = net(inputs)
                outputs = m(outputs)
                loss = criterion(outputs, targets)
                loss.backward()
                net_optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx % 20 == 0) or (batch_idx == (len(train_loader)-1)):
                    print('batch_idx %d, len(trainloader) %d, Loss: %.6f | Acc: %.5f%% (%d/%d)'
                             % (batch_idx, len(train_loader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                #break

            # after all batch-iterations in current epoch

            # test during training
            acc = test_binary(net, information_index, target_index, test_loader)

            if acc > best_acc:
                print('Saving best...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': ep + 1,
                    'optimizer': net_optimizer.state_dict()
                }
                if not os.path.isdir(model_ckpt_dir):
                    os.mkdir(model_ckpt_dir)
                save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep), max_keep=2,
                                is_best=True)
                best_acc = acc

            # Save checkpoint every 10 epochs
            elif ep % 5 == 0:
                print('Saving...')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': ep + 1,
                    'optimizer': net_optimizer.state_dict()
                }
                if not os.path.isdir(model_ckpt_dir):
                    os.mkdir(model_ckpt_dir)
                save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep), max_keep=2,
                                is_best=False)
            #break

        # after all epochs
        acc = test_binary(net, information_index, target_index, test_loader)
        if start_epoch != epochs:
            print('Saving...')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': ep + 1,
                'optimizer': net_optimizer.state_dict()
            }
            if not os.path.isdir(model_ckpt_dir):
                os.mkdir(model_ckpt_dir)
            save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep), max_keep=2, is_best=False)
        else:
            acc = test_binary(net, information_index, target_index, test_loader)
        print('Finished!')# best_acc is %.5f%%\n\n' % best_acc)
        # release current network
        del net, net_optimizer
    #break# since current target
    # after all targets