from __future__ import print_function
import argparse
import torch
import torch.utils.data
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

"""
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=True, download=True,
                     transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)
"""

class NormalLogProb(nn.Module):
    def __init__(self):
        super(NormalLogProb,self).__init__()

    def forward(self, loc, scale, z):
        var = torch.pow(scale, 2)
        return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)

class VAE(nn.Module):
    def __init__(self,hidden_size,intermediate_size):
        super(VAE, self).__init__()

        # Encoder
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.latent_dim = intermediate_size
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, self.intermediate_size)

        # Latent space
        self.fc21 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.fc22 = nn.Linear(self.intermediate_size, self.hidden_size)

        # Decoder
        self.fc3 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.fc4 = nn.Linear(self.intermediate_size, 32 * 32 * 32)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid1 = nn.Sigmoid()
        self.log_q_z = NormalLogProb()
        self.log_p_z = NormalLogProb()
        self.log_p_x = NormalLogProb()

    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        # import pdb; pdb.set_trace()
        out = out.view(out.size(0), 32, 32, 32)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        mu_out = self.sigmoid(self.conv5(out))
        logvar_out = self.sigmoid1(self.conv6(out))
        out = self.reparameterize(mu_out,logvar_out)
        return out, mu_out, logvar_out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out, mu_out, logvar_out = self.decode(z)
        log_p_z = self.log_p_z(torch.zeros(self.hidden_size).cuda().float(),torch.ones(self.hidden_size).cuda().float(), z)
        # log_p_z = self.log_p_z(torch.zeros(self.hidden_size).float(),torch.ones(self.hidden_size).float(), z)
        log_p_x = self.log_p_x(mu_out, torch.sqrt(torch.exp(logvar_out)), x)
        log_q_z = self.log_q_z(mu,torch.sqrt(torch.exp(logvar)),z).sum(-1)
        # return out, log_p_z.sum(-1) + log_p_x.sum(-1).sum(-1).sum(-1) - log_q_z
        return out, mu_out, logvar_out

"""
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 32 * 32 * 3), x.view(-1, 32 * 32 * 3), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))



def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if epoch == args.epochs and i == 0:
            n = min(data.size(0), 8)
            comparison = torch.cat([data[:n],
                                   recon_batch[:n]])
            save_image(comparison.data.cpu(),
                       'snapshots/conv_vae/reconstruction_' + str(epoch) +
                       '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
    if epoch == args.epochs:
        sample = Variable(torch.randn(64, args.hidden_size))
        if args.cuda:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()
        save_image(sample.data.view(64, 3, 32, 32),
                   'snapshots/conv_vae/sample_' + str(epoch) + '.png')
"""