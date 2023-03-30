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

# ==============================================================================
# =                                  Settings                                  =
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch CelebA Training')
# Train setting
parser.add_argument('--lr', dest='lr', type=float, help='learning rate',
                    default=0.00005)
parser.add_argument('--batch_size', dest='batch_size', type=float, help='batch size',
                    default=100)
parser.add_argument('--epochs', dest='epochs', type=int, help='number of epochs',
                    default=100)
parser.add_argument('--model_addr', dest='model_addr', type=str, help='target and shadow model save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/outputs/')

'''
parser.add_argument('--trainset_size_perclass', dest='trainset_size_perclass', type=int, help='size of train set',
                    default=10000)
parser.add_argument('--testset_gen_size_perclass', dest='testset_gen_size_perclass', type=int, help='size of generated test set',
                    default=100)
parser.add_argument('--c_dim', dest='c_dim', type=int, help='number of classes',
                    default=8)
parser.add_argument('--num_models', dest='num_models', type=int, help='number of models to be trained for each GAN',
                    default=5)
# GANs
parser.add_argument('--gan_addr', dest='gan_addr', type=str, help='gan address',
                    default='/data/gehan/PytorchProjects/GANInference/GANs/Conditional-GANs-Pytorch-master/outputs/')
parser.add_argument('--gan_name', dest='gan_name', type=str, help='gan name',
                    default='CGAN_gan_lr_')
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
'''
# Save settings
parser.add_argument('--save_addr', dest='save_addr', type=str, help='save address',
                    default='/data/gehan/PytorchProjects/GANInference/models/attackers/outputs/')
parser.add_argument('--save_name', dest='save_name', type=str, help='save name',
                    default='Classifier')
args = parser.parse_args()

# Train settings
lr = args.lr
batch_size = args.batch_size
epochs = args.epochs
model_addr = args.model_addr
#resume = args.resume
'''
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
'''
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
            print('batch_idx %d, len(trainloader) %d, Loss: %.6f | Acc: %.3f%% (%d/%d)'
                     % (batch_idx, len(train_loader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return net

def attacker_GANship_test(attacker, information_name, targetGAN_list, All_GAN_list, model_addr='/data/gehan/PytorchProjects/GANInference/models/outputs/', z_dim=100, c_dim=8):

    attacker.eval()
    gan_addr = '/data/gehan/PytorchProjects/GANInference/GANs/Conditional-GANs-Pytorch-master/outputs'
    tgan_counts_list = {}# model_str:[corrects, total]
    M_counts = {}# model_str:[corrects, total]
    for tgan in targetGAN_list:
        corrects_tgan = 0
        total_tgan = 0

        # List models trained on current GAN
        model_gan_dir = os.path.join(model_addr, tgan)
        assert os.path.exists(model_gan_dir), 'Wrong address!'
        model_gan = os.listdir(model_gan_dir)
        model_list = []

        for model in model_gan:
            contents = os.listdir(os.path.join(model_gan_dir, model))
            if 'TrainSet_dat' in contents:
                continue
            if model.startswith('Ale') or model.startswith('Goo'):
                continue
            model_list.append(model)

        assert model_list, 'No usable model saved for current GAN: %s!' % tgan

        for model in model_list:

            model_str = model[:3]

            if not M_counts.get(model_str):
                M_counts.update({model_str:[0, 0]})

            # Directory to save model
            model_dir = os.path.join(os.path.join(model_addr, tgan), model)

            # Building model
            if model.startswith('LeN'):
                net = structures.lenet.LeNet().to(device)
            elif model.startswith('Smp'):
                net = structures.lenet.SmplCNN().to(device)
            elif model.startswith('VGG'):
                net = structures.vgg.VGG('VGG11').to(device)
            elif model.startswith('Res'):
                net = structures.resnet.ResNet18().to(device)
            else:
                continue
            '''
            elif model.startswith('Ale'):
                net = structures.alexnet().to(device)
            elif model.startswith('Goo'):
                net = structures.googlenet.GoogLeNet().to(device)
            '''

            # Load checkpoint
            model_ckpt_dir = os.path.join(model_dir, 'checkpoints')

            ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
            if ckpt_best:
                #best_acc = ckpt_best['acc']
                net.load_state_dict(ckpt_best['net'])
            else:
                ckpt = load_checkpoint(model_ckpt_dir)
                if ckpt:
                    #best_acc = ckpt['acc']
                    net.load_state_dict(ckpt['net'])
                else:
                    print(' [Wrong] No checkpoint!')
                    continue

            net.eval()

            attacker_prediction_list = {}
            for gan_query in All_GAN_list:
                gan_query_addr = os.path.join(gan_addr, gan_query)
                if gan_query.startswith('DCGAN'):
                    G = GAN_structures.GeneratorDCGAN(nch_in=z_dim, n_class=c_dim).to(device)
                else:
                    G = GAN_structures.GeneratorACGAN(z_dim=z_dim, c_dim=c_dim).to(device)

                # Load GAN
                assert os.path.exists(gan_query_addr), 'Wrong directory for loading GAN!'
                ckpt_dir = os.path.join(gan_query_addr, 'checkpoints')
                ckpt = load_checkpoint(ckpt_dir)
                G.load_state_dict(ckpt['G'])

                # generate image
                noise = torch.randn(10, z_dim).to(device)
                gan_label = torch.rand(10).to(device)
                gan_label = torch.floor(gan_label * c_dim).int()
                gan_label_onehot = torch.tensor(np.eye(c_dim)[[gan_label.cpu().numpy()]], dtype=noise.dtype).to(device)
                if gan_query.startswith('DCGAN'):
                    noise = noise.unsqueeze(-1).unsqueeze(-1)
                    gan_label_onehot = gan_label_onehot.unsqueeze(-1).unsqueeze(-1)

                image = G(noise, gan_label_onehot)
                #output = G(input_noise, input_labels)

                image = Denormalize(image)
                #for i in range(image.shape[0]):
                #    im = image[i].cpu().clone()
                    #im = im.squeeze(0)
                # query the model
                output = net(image)

                if information_name == 'posterior_sorted':
                    information = output.sort(dim=-1)[0]
                elif information_name == 'posterior_label':
                    # one-hot true label (generated by GAN)
                    #label = torch.zeros(output.shape).to(device)
                    #label[gan_label] = gan_label
                    label = torch.tensor(np.eye(c_dim)[[gan_label.cpu().numpy()]], dtype=noise.dtype).to(device)
                    # concatenate UN-SORTED posterior and true label
                    information = torch.cat([output, label], 1)
                elif information_name == 'posterior_correct':
                    # correctness of classification
                    _, predicted = output.max(1)
                    correctness = predicted.eq(gan_label).float()
                    correctness = correctness.unsqueeze(-1)
                    information = torch.cat([output.sort(dim=-1)[0], correctness], -1)

                attacker_output = attacker(information)
                temp = attacker_output.mean()
                attacker_prediction_list.update({gan_query:attacker_output.mean()})
            # after query a model by all gans
            attacker_prediction_sorted = sorted(attacker_prediction_list.items(), key=lambda x:x[1])
            attacker_predicted_top = attacker_prediction_sorted[len(attacker_prediction_sorted)-1]

            # evaluate correctness of attacker prediction on current model
            # and count the number
            if attacker_predicted_top[0] == tgan:
                corrects_tgan += 1
                M_counts[model_str][0] += 1

            total_tgan += 1
            M_counts[model_str][1] += 1
            print('attacker_prediction_sorted ', attacker_prediction_sorted)
        # after test on all models generated by current tgan
        tgan_counts_list.update({tgan: [corrects_tgan, total_tgan]})
    # after all target gans
    correct_all = 0
    total_all = 0
    tgan_acc_list = {}
    for key in tgan_counts_list:
        correct_all += tgan_counts_list[key][0]
        total_all += tgan_counts_list[key][1]
        tgan_acc_list.update({key:tgan_counts_list[key][0]/tgan_counts_list[key][1]})
    acc_all = correct_all/total_all

    model_str_acc_list = {}
    for key in M_counts:
        model_str_acc_list.update({key:M_counts[key][0]/M_counts[key][1]})

    return acc_all, tgan_counts_list, M_counts



# ==============================================================================
# =                              Main procedure                                =
# ==============================================================================

# Save print log
logger_addr = save_addr#os.path.join(save_addr, save_name)
if not os.path.exists(logger_addr):
    os.makedirs(logger_addr)
sys.stdout = Logger("%s/%s/together_multiclass_TrainLog.txt" % (logger_addr, save_name))

# several lists for testing phase
GAN_list = []
assert os.path.exists(model_addr), 'Wrong model address!'
filelist = os.listdir(model_addr)
for file in filelist:
    if os.path.isdir(os.path.join(model_addr, file)):
        if (not file.endswith('_')) and (not file.endswith('T')):
            GAN_list.append(file)

# Datasets preparation
trainset_path = os.path.join(model_addr, 'Trainset_postGAN_together_AllStrctr')
trainset = loadData(trainset_path)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

content_list = ['posterior_sorted', 'posterior_label', 'posterior_correct', 'GANship', 'StrctrShip', 'GAN_name', 'Strctr_name']
information_list = ['posterior_sorted', 'posterior_label', 'posterior_correct']
target_list = ['GAN_name']#, 'StrctrShip']

# ====== Train and Test a model for each of information ====== #
for i in information_list:

    information_index = content_list.index(i)

    for t in target_list:
        print('\n--* %s --> %s *--' % (i, t))
        target_index = content_list.index(t)

        # Directory to save model
        model_dir = os.path.join(os.path.join(os.path.join(save_addr, save_name), t), i)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)

        if i == 'posterior_sorted':
            # Building model
            net = structures.lenet.attacker(nch_info=8, nch_output=len(All_GAN_list)).to(device)
        elif i == 'posterior_label':
            # Building model
            net = structures.lenet.attacker(nch_info=16, nch_output=len(All_GAN_list)).to(device)
        elif i == 'posterior_correct':
            # Building model
            net = structures.lenet.attacker(nch_info=9, nch_output=len(All_GAN_list)).to(device)

        criterion = nn.CrossEntropyLoss()
        net_optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=5e-4)

        # Load checkpoint
        model_ckpt_dir = os.path.join(model_dir, 'checkpoints')

        try:
            ckpt_best = load_checkpoint(model_ckpt_dir, load_best=True)
            start_epoch = ckpt_best['epoch']
            #best_acc = ckpt_best['acc']
            net.load_state_dict(ckpt_best['net'])
            net_optimizer.load_state_dict(ckpt_best['optimizer'])
        except:
            try:
                ckpt = load_checkpoint(model_ckpt_dir, load_best=False)
                start_epoch = ckpt['epoch']
                #best_acc = ckpt['acc']
                net.load_state_dict(ckpt['net'])
                net_optimizer.load_state_dict(ckpt['optimizer'])
            except:
                print(' [*] No checkpoint!\nTrain from beginning.')
                start_epoch = 0
                #best_acc = 0

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
                #targets = data[-1].to(device)
                targets = targets.unsqueeze(-1)

                net_optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                net_optimizer.step()

                train_loss += loss.item()
                predicted = torch.round(outputs)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                '''
                # test during training
                if t == 'GANship':
                    acc_all, tgan_acc_list, model_str_acc_list = attacker_GANship_test(net, i, targetGAN_list, All_GAN_list, model_addr='/data/gehan/PytorchProjects/GANInference/models/outputs/', z_dim=100, c_dim=8)
                '''
                if batch_idx % 100 == 0:
                    print('batch_idx %d, len(trainloader) %d, Loss: %.6f | Acc: %.3f%% (%d/%d)'
                             % (batch_idx, len(train_loader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))
                #break

            # after all batch-iterations in current epoch

            # test during training
            if t == 'GANship':
                if (ep != 0) and (ep%30 == 0):
                    acc_all, tgan_acc_list, model_str_acc_list = attacker_GANship_test(net, i, targetGAN_list, All_GAN_list,
                                                                                   model_addr='/data/gehan/PytorchProjects/GANInference/models/outputs/',
                                                                                   z_dim=100, c_dim=8)
                    print('Accuracy cross all: %.3f' % (acc_all))
                    print('Accuracy on each GAN structure:\n', tgan_acc_list)
                    print('Accuracy on each model structure:\n', model_str_acc_list)

            # Save checkpoint every 10 epochs
            if ep % 5 == 0:
                print('Saving...')
                state = {
                    'net': net.state_dict(),
                    #'acc': acc_all,
                    'epoch': ep + 1,
                    'optimizer': net_optimizer.state_dict()
                }
                if not os.path.isdir(model_ckpt_dir):
                    os.mkdir(model_ckpt_dir)
                save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep), max_keep=2,
                                is_best=False)
            #break

        # after all epochs
        if start_epoch != epochs:
            print('Saving...')
            state = {
                'net': net.state_dict(),
                #'acc': acc_all,
                'epoch': ep + 1,
                'optimizer': net_optimizer.state_dict()
            }
            if not os.path.isdir(model_ckpt_dir):
                os.mkdir(model_ckpt_dir)
            save_checkpoint(state, '%s/Epoch_(%d).ckpt' % (model_ckpt_dir, ep), max_keep=2, is_best=False)

        # test during training
        if t == 'GANship':
            acc_all, tgan_acc_list, model_str_acc_list = attacker_GANship_test(net, i, targetGAN_list, All_GAN_list,
                                                                               model_addr='/data/gehan/PytorchProjects/GANInference/models/outputs/',
                                                                               z_dim=100, c_dim=8)
            print('Accuracy cross all: %.3f' % (acc_all))
            print('Accuracy on each GAN structure:\n', tgan_acc_list)
            print('Accuracy on each model structure:\n', model_str_acc_list)

        print('Finished!')# best_acc is %.3f%%\n\n' % best_acc)
        # release current network
        del net, net_optimizer
    #break# since current target
    # after all targets