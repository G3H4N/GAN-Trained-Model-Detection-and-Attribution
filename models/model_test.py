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
sys.path.append("../..")
from models import structures
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

# ==============================================================================
# =                                  Settings                                  =
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
# Train setting

parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=200)
args = parser.parse_args()

# Train settings
batch_size = args.batch_size

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
            loss = criterion(outputs, targets.long())

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
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            outputs_test.extend(outputs)
            members_test.extend(targets)

    auc = roc_auc_score(torch.BoolTensor(members_test), outputs_test)

    print('>> Test\nAUC: %.5f' % (auc))

    return auc


def test_f1score(net, test_loader):
    # global best_acc
    net.eval()

    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        y_true = torch.tensor([]).to(device)
        y_pred = torch.tensor([]).to(device)
        for batch_idx, data in enumerate(test_loader):
            inputs = data[1].to(device)
            targets = data[-1].to(device)
            targets = targets.squeeze(-1)
            outputs = net(inputs)
            #loss = criterion(outputs, targets.long())

            #test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            y_true = torch.cat((y_true, targets), 0)
            y_pred = torch.cat((y_pred, predicted), 0)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
    acc = 100. * correct / total

    print('>> Test\nAccuracy: %.5f%%' % (acc))
    print('Global F1-score: %.5f%%' % (metrics.f1_score(y_true, y_pred, average='micro')))
    print('Average F1-score: %.5f%%' % (metrics.f1_score(y_true, y_pred, average='macro')))
    print('Weighted F1-score: %.5f%%' % (metrics.f1_score(y_true, y_pred, average='weighted')))
    print('Separate F1-score: ', metrics.f1_score(y_true, y_pred, average=None))

    return acc
# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

model_dir_list = {'/data/gehan/PytorchProjects/GANInference/models/attackers/outputs/real_GANship_withReal/posterior_label':[1600, 41, '/data/gehan/PytorchProjects/GANInference/models/outputs/real_GANship_Testset_AllStrctr'],
                  '/data/gehan/PytorchProjects/GANInference/models/attackers/FM_outputs/real_GANship_withReal/posterior_label':[2000, 41, '/data/gehan/PytorchProjects/GANInference/models/FM_outputs/real_GANship_Testset_AllStrctr'],
                  '/data/gehan/PytorchProjects/GANInference/models/attackers/SVHN_outputs/real_GANship_withReal/posterior_label':[2000, 41, '/data/gehan/PytorchProjects/GANInference/models/SVHN_outputs/real_GANship_Testset_AllStrctr'],
                  '/data/gehan/PytorchProjects/GANInference/models/attackers/CIFAR10_outputs/real_GANship_withReal/posterior_label':[2000, 9, '/data/gehan/PytorchProjects/GANInference/models/CIFAR10_outputs/real_GANship_Testset_AllStrctr'],
                  '/data/gehan/PytorchProjects/GANInference/models/attackers/CIFAR10_HALF_outputs/real_GANship_withReal/posterior_label':[2000, 9, '/data/gehan/PytorchProjects/GANInference/models/CIFAR10_HALF_outputs/real_GANship_Testset_AllStrctr']}

# ====== Train and Test a model for each of information ====== #
for model_dir in model_dir_list:

    assert os.path.isdir(model_dir), 'No such trained model!'
    # Datasets preparation
    testset_path = model_dir_list[model_dir][2]
    testset = loadData(testset_path)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

    # Building model
    net = structures.lenet.attacker_real(nch_info=model_dir_list[model_dir][0], nch_output=model_dir_list[model_dir][1]).to(device)

    net.eval()
    # Load checkpoint
    model_ckpt_dir = os.path.join(model_dir, 'checkpoints')
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
    print('Model loaded, acc = %.2f' % (best_acc))

    with torch.no_grad():
        test_f1score(net, test_loader)
