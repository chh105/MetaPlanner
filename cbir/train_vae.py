import argparse
from math import log10
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data

from model import *
from multi_channel_data_loader import *

M_N = 0.001

def train():
    '''train'''
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_mmd_loss = 0
    epoch_kld_loss = 0
    for idx, batch in enumerate(training_data_loader):
        input, target = batch[0].to(device), batch[1].to(device)

        recon, input, z, mu, log_var = model(input)
        optimizer.zero_grad()

        loss_dict = model.loss_function(recon, target, z, mu, log_var, M_N = M_N)
        loss = loss_dict['loss']
        recon_loss = loss_dict['Reconstruction_Loss']
        mmd_loss = loss_dict['MMD']
        kld_loss = loss_dict['KLD']

        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_mmd_loss += mmd_loss.item()
        epoch_kld_loss += kld_loss.item()

        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(epoch, idx+1, len(training_data_loader),
                                                                                loss.item(),
                                                                                recon_loss.item(),
                                                                                mmd_loss.item(),
                                                                                kld_loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(epoch,
                                                                                  epoch_loss / len(training_data_loader),
                                                                                  epoch_recon_loss / len(training_data_loader),
                                                                                  epoch_mmd_loss / len(training_data_loader),
                                                                                  epoch_kld_loss / len(training_data_loader),
                                                                                  ))

def val():
    '''val'''
    model.eval()
    avg_loss = 0
    avg_recon_loss = 0
    avg_mmd_loss = 0
    avg_kld_loss = 0
    with torch.no_grad():
        for batch in val_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            recon, input, z, mu, log_var = model(input)
            loss_dict = model.loss_function(recon, target, z, mu, log_var, M_N=M_N)

            loss = loss_dict['loss']
            recon_loss = loss_dict['Reconstruction_Loss']
            mmd_loss = loss_dict['MMD']
            kld_loss = loss_dict['KLD']

            avg_loss += loss.item()
            avg_recon_loss += recon_loss.item()
            avg_mmd_loss += mmd_loss.item()
            avg_kld_loss += kld_loss.item()

    print("===> Avg. Validation Losses: {:.3f} {:.3f} {:.3f} {:.3f}".format(avg_loss / len(val_data_loader),
                                                                            avg_recon_loss / len(val_data_loader),
                                                                            avg_mmd_loss / len(val_data_loader),
                                                                            avg_kld_loss / len(val_data_loader),))
    return avg_loss

def checkpoint(best_val_loss, curr_loss):
    '''checkpoint'''
    if curr_loss < best_val_loss:
        model_out_path = "./model_checkpoints_vae/model_epoch_{}.pth".format(epoch)
        torch.save(model, model_out_path)
        print("Validation loss improved from {:.3f} to {:.3f}! ".format(best_val_loss,
                                                                        curr_loss))
        print("Checkpoint saved to {}".format(model_out_path))
        return curr_loss
    else:
        return best_val_loss

if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--batchSize', type=int, default=2, help='training batch size')
    parser.add_argument('--valBatchSize', type=int, default=2, help='validation batch size')
    parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--threads', type=int, default=2, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    opt = parser.parse_args()

    print(opt)

    if not os.path.exists('./model_checkpoints_vae'):
        os.makedirs('./model_checkpoints_vae')

    torch.manual_seed(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('===> Loading datasets')
    training_data_loader, val_data_loader = get_loader(train_bs=opt.batchSize,
                                                       val_bs=opt.valBatchSize,
                                                       train_num_samples_per_epoch=200,
                                                       val_num_samples_per_epoch=40,
                                                       num_works=opt.threads)

    print('===> Building model')
    model = InfoVAE(in_channels=2,latent_dim=1024).to(device)
    if os.path.isfile('./model_checkpoints_vae/init_model.pth'):
        model=torch.load('./model_checkpoints_vae/init_model.pth')
        print('loading initial model')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    best_val_loss = np.infty
    for epoch in range(1, opt.nEpochs + 1):
        train()
        val_loss = val()
        best_val_loss = checkpoint(best_val_loss,val_loss)




