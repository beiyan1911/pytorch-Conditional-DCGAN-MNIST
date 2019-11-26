import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DCGAN')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default=128)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default=0.01)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--nz', type=int, default=100, help='Number of dimensions for input noise.')
    parser.add_argument('--cuda', default=False, help='Enable cuda')
    parser.add_argument('--save_every', type=int, default=1, help='After how many epochs to save the model.')
    parser.add_argument('--print_every', type=int, default=50,
                        help='After how many epochs to print loss and save output samples.')
    parser.add_argument('--out_paths', default='outputs', type=str, help='Path to save the trained models.')
    parser.add_argument('--save_dir', type=str, default='models', help='Path to save the trained models.')
    parser.add_argument('--samples_dir', type=str, default='samples', help='Path to save the output samples.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if args.cuda else "cpu")
    args.save_dir = os.path.join(args.out_paths, args.save_dir)
    args.samples_dir = os.path.join(args.out_paths, args.samples_dir)

    if not os.path.exists(args.out_paths):
        os.mkdir(args.out_paths)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.exists(args.samples_dir):
        os.mkdir(args.samples_dir)

    INPUT_SIZE = 784
    SAMPLE_SIZE = 80
    NUM_LABELS = 10
    train_dataset = datasets.MNIST(root='datas', train=True, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)

    model_d = ModelD().to(device)
    model_g = ModelG(args.nz).to(device)
    criterion = nn.BCELoss()
    # torch.random

    fixed_noise = torch.randn(SAMPLE_SIZE, args.nz).to(device)
    fixed_labels = torch.zeros(SAMPLE_SIZE, NUM_LABELS).to(device)

    for i in range(NUM_LABELS):
        for j in range(SAMPLE_SIZE // NUM_LABELS):
            fixed_labels[i * (SAMPLE_SIZE // NUM_LABELS) + j, i] = 1.0

    optim_d = optim.SGD(model_d.parameters(), lr=args.lr)
    optim_g = optim.SGD(model_g.parameters(), lr=args.lr)

    real_label = 1
    fake_label = 0

    for epoch_idx in range(args.epochs):
        model_d.train()
        model_g.train()

        d_loss = 0.0
        g_loss = 0.0
        for batch_idx, (train_x, train_y) in enumerate(train_loader):
            batch_size = train_x.size(0)
            input = train_x.view(-1, INPUT_SIZE).to(device)
            real_label = torch.ones((batch_size, 1), dtype=torch.float32).to(device)
            fake_label = torch.zeros((batch_size, 1), dtype=torch.float32).to(device)
            one_hot_labels = torch.zeros(batch_size, NUM_LABELS).scatter_(1, train_y.view(batch_size, 1), 1).to(device)

            # optim D
            output = model_d(input, one_hot_labels)
            optim_d.zero_grad()
            errD_real = criterion(output, real_label)
            errD_real.backward()
            realD_mean = output.data.cpu().mean()

            rand_y = torch.from_numpy(np.random.randint(0, NUM_LABELS, size=(batch_size, 1), dtype='int64'))
            one_hot_labels = torch.zeros(batch_size, NUM_LABELS).scatter_(1, rand_y.view(batch_size, 1), 1).to(device)
            noise = torch.randn(batch_size, args.nz).to(device)

            g_out = model_g(noise, one_hot_labels)
            output = model_d(g_out, one_hot_labels)
            errD_fake = criterion(output, fake_label)

            fakeD_mean = output.data.cpu().mean()
            errD = errD_real + errD_fake
            errD_fake.backward()
            optim_d.step()

            # optim G
            noise = torch.randn(batch_size, args.nz).to(device)
            rand_y = torch.from_numpy(np.random.randint(0, NUM_LABELS, size=(batch_size, 1), dtype='int64'))
            one_hot_labels = torch.zeros(batch_size, NUM_LABELS).scatter_(1, rand_y.view(batch_size, 1), 1).to(device)
            g_out = model_g(noise, one_hot_labels)
            output = model_d(g_out, one_hot_labels)
            errG = criterion(output, real_label)
            optim_g.zero_grad()
            errG.backward()
            optim_g.step()

            d_loss += errD.item()
            g_loss += errG.item()
            if batch_idx % args.print_every == 0:
                print(
                    "\t{} ({} / {}) mean D(fake) = {:.4f}, mean D(real) = {:.4f}".
                        format(epoch_idx, batch_idx, len(train_loader), fakeD_mean,
                               realD_mean))

                g_out = model_g(fixed_noise, fixed_labels).view(SAMPLE_SIZE, 1, 28, 28).cpu()
                save_image(g_out,
                           '{}/{}_{}.png'.format(
                               args.samples_dir, epoch_idx, batch_idx))

        print('Epoch {} - D loss = {:.4f}, G loss = {:.4f}'.format(epoch_idx,
                                                                   d_loss, g_loss))
        if epoch_idx % args.save_every == 0:
            torch.save({'state_dict': model_d.state_dict()},
                       '{}/model_d_epoch_{}.pth'.format(
                           args.save_dir, epoch_idx))
            torch.save({'state_dict': model_g.state_dict()},
                       '{}/model_g_epoch_{}.pth'.format(
                           args.save_dir, epoch_idx))
