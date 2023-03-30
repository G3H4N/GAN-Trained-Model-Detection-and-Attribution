from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json

import model
import numpy as np
import pylib
import PIL.Image as Image
import tensorboardX
import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as tforms
import torchlib

import pickle


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

# command line arguments
parser = argparse.ArgumentParser()
# model
parser.add_argument('--z_dim', dest='z_dim', type=int, help='size of noise', default=100)
parser.add_argument('--c_dim', dest='c_dim', type=int, help='# of classes', default=8)
# training
parser.add_argument('--epoch', dest='epoch', type=int, default=400)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
parser.add_argument('--d_learning_rate', dest='d_learning_rate', type=float, default=0.0001)
parser.add_argument('--g_learning_rate', dest='g_learning_rate', type=float, default=0.0001)
parser.add_argument('--n_d', dest='n_d', type=int, help='# of d updates per g update', default=1)
parser.add_argument('--loss_mode', dest='loss_mode', choices=['gan', 'lsgan', 'wgan', 'hinge_v1', 'hinge_v2'], default='wgan')
parser.add_argument('--gp_mode', dest='gp_mode', choices=['none', 'dragan', 'wgan-gp'], default='wgan-gp')
parser.add_argument('--gp_coef', dest='gp_coef', type=float, default=1.0)
parser.add_argument('--norm', dest='norm', choices=['none', 'batch_norm', 'instance_norm'], default='none')
parser.add_argument('--weight_norm', dest='weight_norm', choices=['none', 'spectral_norm', 'weight_norm'], default='spectral_norm')
# others
parser.add_argument('--experiment_name', dest='experiment_name', default='CGAN_wgan-gp_D_0')
parser.add_argument('--trainset_dir', dest='trainset_dir', default='/data/gehan/Datasets/CelebA/png/align/CelebA_png_align_Attr_22_21_20_sets/TrainSet')
parser.add_argument('--trainset_size', dest='trainset_size', type=int, default=80000)
parser.add_argument('--image_size', dest='image_size', type=int, default=64)

# parse arguments
args = parser.parse_args()
# model
z_dim = args.z_dim
# training
epoch = args.epoch
batch_size = args.batch_size
d_learning_rate = args.d_learning_rate
g_learning_rate = args.g_learning_rate
n_d = args.n_d
loss_mode = args.loss_mode
gp_mode = args.gp_mode
gp_coef = args.gp_coef
norm = args.norm
weight_norm = args.weight_norm
# ohterss
experiment_name = args.experiment_name
trainset_dir = args.trainset_dir
trainset_size = args.trainset_size
c_dim = args.c_dim
image_size = args.image_size

# save settings
pylib.mkdir('./outputs/%s' % experiment_name)
with open('./outputs/%s/setting.txt' % experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# others
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


# ==============================================================================
# =                                   setting                                  =
# ==============================================================================

# data
transform = tforms.Compose(
    [tforms.Resize(size=(image_size, image_size), interpolation=Image.BICUBIC),
     tforms.ToTensor(),
     tforms.Lambda(lambda x: torch.cat((x, x, x), dim=0)),
     tforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
)

# loading dataset from file
def loadData(filename):
    with open(filename, 'rb') as file:
        print('Open file', filename, '......ok')
        obj = pickle.load(file, encoding='latin1')
        file.close()
        return obj

CelebA_attr_train = loadData(trainset_dir)
split = [trainset_size, len(CelebA_attr_train) - trainset_size]
trainset, _ = torch.utils.data.random_split(CelebA_attr_train, split)
train_loader = torch.utils.data.DataLoader(
    dataset=trainset,#dsets.FashionMNIST('data/FashionMNIST', train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=use_gpu,
    drop_last=True
)

# model
D = model.DiscriminatorCGAN(x_dim=3, c_dim=c_dim, norm=norm, weight_norm=weight_norm).to(device)
G = model.GeneratorCGAN(z_dim=z_dim, c_dim=c_dim).to(device)

# gan loss function
d_loss_fn, g_loss_fn = model.get_losses_fn(loss_mode)

# optimizer
d_optimizer = torch.optim.Adam(D.parameters(), lr=d_learning_rate, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=g_learning_rate, betas=(0.5, 0.999))


# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# load checkpoint
ckpt_dir = './outputs/%s/checkpoints' % experiment_name
pylib.mkdir(ckpt_dir)
try:
    ckpt = torchlib.load_checkpoint(ckpt_dir)
    start_ep = ckpt['epoch']
    D.load_state_dict(ckpt['D'])
    G.load_state_dict(ckpt['G'])
    d_optimizer.load_state_dict(ckpt['d_optimizer'])
    g_optimizer.load_state_dict(ckpt['g_optimizer'])
except:
    print(' [*] No checkpoint!')
    start_ep = 0

# writer
writer = tensorboardX.SummaryWriter('./outputs/%s/summaries' % experiment_name)

# run
z_sample = torch.randn(c_dim * 10, z_dim).to(device)
c_sample = torch.tensor(np.concatenate([np.eye(c_dim)] * 10), dtype=z_sample.dtype).to(device)
for ep in range(start_ep, epoch):
    for i, (x, c_dense) in enumerate(train_loader):
        step = ep * len(train_loader) + i + 1
        D.train()
        G.train()

        # train D
        x = x.to(device)
        c_dense = c_dense.to(device)
        z = torch.randn(batch_size, z_dim).to(device)
        c = torch.tensor(np.eye(c_dim)[[c_dense.cpu().numpy()]], dtype=z.dtype).to(device)

        x_f = G(z, c).detach()
        x_gan_logit = D(x, c)
        x_f_gan_logit = D(x_f, c)

        d_x_gan_loss, d_x_f_gan_loss = d_loss_fn(x_gan_logit, x_f_gan_logit)
        gp = model.gradient_penalty_c(D, c, x, x_f, mode=gp_mode)
        d_loss = d_x_gan_loss + d_x_f_gan_loss + gp * gp_coef

        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        writer.add_scalar('D/d_gan_loss', (d_x_gan_loss + d_x_f_gan_loss).data.cpu().numpy(), global_step=step)
        writer.add_scalar('D/gp', gp.data.cpu().numpy(), global_step=step)

        # train G
        if step % n_d == 0:
            z = torch.randn(batch_size, z_dim).to(device)

            x_f = G(z, c)
            x_f_gan_logit = D(x_f, c)

            g_gan_loss = g_loss_fn(x_f_gan_logit)
            g_loss = g_gan_loss

            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            writer.add_scalar('G/g_gan_loss', g_gan_loss.data.cpu().numpy(), global_step=step)

        # display
        if step % 50 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (ep, i + 1, len(train_loader)))

        '''
        # sample
        if step % 250 == 0:
            G.eval()
            x_f_sample = (G(z_sample, c_sample) + 1) / 2.0

            save_dir = './outputs/%s/sample_training' % experiment_name
            pylib.mkdir(save_dir)
            torchvision.utils.save_image(x_f_sample, '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, ep, i + 1, len(train_loader)), nrow=10)
        '''

    G.eval()
    x_f_sample = (G(z_sample, c_sample) + 1) / 2.0

    save_dir = './outputs/%s/sample_training' % experiment_name
    pylib.mkdir(save_dir)
    torchvision.utils.save_image(x_f_sample, '%s/Epoch_(%d).jpg' % (save_dir, ep), nrow=10)

    torchlib.save_checkpoint({'epoch': ep + 1,
                              'D': D.state_dict(),
                              'G': G.state_dict(),
                              'd_optimizer': d_optimizer.state_dict(),
                              'g_optimizer': g_optimizer.state_dict()},
                             '%s/Epoch_(%d).ckpt' % (ckpt_dir, ep + 1),
                             max_keep=2)
