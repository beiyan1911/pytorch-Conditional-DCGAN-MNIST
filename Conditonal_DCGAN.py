import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import time


class ModelD(nn.Module):
    def __init__(self):
        super(ModelD, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 28 * 28 + 1000, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.fc3 = nn.Linear(10, 1000)

    def forward(self, x, labels):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 28, 28)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(batch_size, 64 * 28 * 28)
        y_ = self.fc3(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class ModelG(nn.Module):
    def __init__(self, z_dim):
        self.z_dim = z_dim
        super(ModelG, self).__init__()
        self.fc2 = nn.Linear(10, 1000)
        self.fc = nn.Linear(self.z_dim + 1000, 64 * 28 * 28)
        self.bn1 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, 1, 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        y_ = self.fc2(labels)
        y_ = F.relu(y_)
        x = torch.cat([x, y_], 1)
        x = self.fc(x)
        x = x.view(batch_size, 64, 28, 28)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = torch.sigmoid(x)
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args, device):
    batch_size = args.batch_size
    net_G = ModelG(args.nz).to(device)
    net_G.apply(weights_init)
    net_D = ModelD().to(device)
    net_D.apply(weights_init)
    criterion = nn.BCELoss()

    # load model
    if args.netG != '':
        net_G.load_state_dict(torch.load(args.netG)['state_dict'])
    if args.netD != '':
        net_D.load_state_dict(torch.load(args.netD)['state_dict'])

    # dataloader
    train_dataset = datasets.MNIST(root='datasets', train=True, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    fixed_noise = torch.randn(SAMPLE_SIZE, args.nz).to(device)
    fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS).to(device)
    real_label = torch.ones((batch_size, 1), dtype=torch.float32).to(device)
    fake_label = torch.zeros((batch_size, 1), dtype=torch.float32).to(device)

    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i * (SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0

    optim_d = optim.SGD(net_D.parameters(), lr=args.lr)
    optim_g = optim.SGD(net_G.parameters(), lr=args.lr)

    for epoch_idx in range(args.epoch_start, args.epochs):
        net_D.train()
        net_G.train()

        d_loss, g_loss = 0.0, 0.0

        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            if train_x.size(0) != batch_size:
                continue
            input = train_x.view(-1, INPUT_SIZE).to(device)
            one_hot_labels = torch.zeros(batch_size, NUM_LABELS).scatter_(1, train_y.view(batch_size, 1), 1).to(device)

            # optim D
            output = net_D(input, one_hot_labels)
            optim_d.zero_grad()
            errD_real = criterion(output, real_label)
            errD_real.backward()
            realD_mean = output.data.cpu().mean()

            rand_y = torch.from_numpy(np.random.randint(0, NUM_LABELS, size=(batch_size, 1), dtype='int64'))
            one_hot_labels = torch.zeros(batch_size, NUM_LABELS).scatter_(1, rand_y.view(batch_size, 1), 1).to(device)
            noise = torch.randn(batch_size, args.nz).to(device)

            g_out = net_G(noise, one_hot_labels)
            output = net_D(g_out, one_hot_labels)
            errD_fake = criterion(output, fake_label)

            fakeD_mean = output.data.cpu().mean()
            errD = errD_real + errD_fake
            errD_fake.backward()
            optim_d.step()

            # optim G
            noise = torch.randn(batch_size, args.nz).to(device)
            rand_y = torch.from_numpy(np.random.randint(0, NUM_LABELS, size=(batch_size, 1), dtype='int64'))
            one_hot_labels = torch.zeros(batch_size, NUM_LABELS).scatter_(1, rand_y.view(batch_size, 1), 1).to(device)
            g_out = net_G(noise, one_hot_labels)
            output = net_D(g_out, one_hot_labels)
            errG = criterion(output, real_label)
            optim_g.zero_grad()
            errG.backward()
            optim_g.step()

            d_loss += errD.item()
            g_loss += errG.item()
            if batch_idx % args.print_every == 0:
                print(
                    "\tEpoch {} ({} / {}) mean D(fake) = {:.4f}, mean D(real) = {:.4f}".
                        format(epoch_idx, batch_idx, len(train_loader), fakeD_mean,
                               realD_mean))

                g_out = net_G(fixed_noise, fixed_labels).view(SAMPLE_SIZE, 1, 28, 28).cpu()
                save_image(g_out,
                           '{}/{}_{}.png'.format(
                               args.samples_dir, epoch_idx, batch_idx))

        print('Epoch {} - D loss = {:.4f}, G loss = {:.4f}'.format(epoch_idx,
                                                                   d_loss, g_loss))
        if epoch_idx % args.save_every == 0:
            torch.save({'state_dict': net_D.state_dict()}, '{}/model_d_epoch_{}.pth'.format(args.save_dir, epoch_idx))
            torch.save({'state_dict': net_G.state_dict()}, '{}/model_g_epoch_{}.pth'.format(args.save_dir, epoch_idx))


def test(args, device):
    net_G = ModelG(args.nz).to(device)

    assert args.netG != '', 'netG must not null'
    # load model
    net_G.load_state_dict(torch.load(args.netG)['state_dict'])

    # test input
    fixed_noise = torch.randn(TEST_SIZE, args.nz).to(device)
    fixed_labels = torch.zeros(TEST_SIZE, NUM_LABELS).to(device)
    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i * (SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0

    net_G.eval()

    g_out = net_G(fixed_noise, fixed_labels).view(SAMPLE_SIZE, 1, 28, 28).cpu()
    date = time.strftime("%Y-%m-%d_%H_%M_%S")
    save_image(g_out, '{}/{}.png'.format(args.test_dir, date))


INPUT_SIZE = 784
SAMPLE_SIZE = 80
TEST_SIZE = 80
NUM_LABELS = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DCGAN')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default=128)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default=0.01)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--nz', type=int, default=100, help='Number of dimensions for input noise.')
    parser.add_argument('--cuda', default=True, help='Enable cuda')
    parser.add_argument('--save_every', type=int, default=1, help='After how many epochs to save the model.')
    parser.add_argument('--print_every', type=int, default=50,
                        help='After how many epochs to print loss and save output samples.')
    parser.add_argument('--out_paths', default='outputs', type=str, help='Path to save the trained models.')
    parser.add_argument('--save_dir', type=str, default='models', help='Path to save the trained models.')
    parser.add_argument('--samples_dir', type=str, default='samples', help='Path to save the output samples.')
    parser.add_argument('--test_dir', type=str, default='tests', help='Path to save the output samples.')
    # outputs/models/model_g_epoch_9.pth
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--epoch_start', default=2, help="epoch count")
    parser.add_argument('--is_train', default=False, type=bool, help="train or test")
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.cuda else "cpu")
    # make folders
    args.save_dir = os.path.join(args.out_paths, args.save_dir)
    args.samples_dir = os.path.join(args.out_paths, args.samples_dir)
    args.test_dir = os.path.join(args.out_paths, args.test_dir)
    if not os.path.exists(args.out_paths):
        os.mkdir(args.out_paths)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.samples_dir):
        os.mkdir(args.samples_dir)
    if not os.path.exists(args.test_dir):
        os.mkdir(args.test_dir)

    if args.is_train:
        train(args, device)
    else:
        test(args, device)
