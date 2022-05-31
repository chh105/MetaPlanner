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
from sim_siam_data_loader import *

def train(epoch):
    '''train'''
    model.train()
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_feature_loss = 0
    epoch_triplet_loss = 0
    for idx, batch in enumerate(training_data_loader):
        input1, input2, input3, pos_img, neg_img = batch[0].to(device).float(), batch[1].to(device).float(), \
                                           batch[2].to(device).float(), \
                                           batch[3].to(device).float(), batch[4].to(device).float()

        p1, p2, z1, z2, input1, recon, z_pos, z_neg = model(input1, input3, pos_img, neg_img)
        optimizer.zero_grad()

        loss_dict = model.loss_function(p1, p2, z1, z2, input1, recon, z_pos, z_neg)
        loss = loss_dict['loss']
        recon_loss = loss_dict['Recon_loss']
        feature_loss = loss_dict['Feature_loss']
        triplet_loss = loss_dict['Triplet_loss']

        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_feature_loss += feature_loss.item()
        epoch_triplet_loss += triplet_loss.item()

        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(epoch, idx+1, len(training_data_loader),
                                                                                loss.item(),
                                                                                recon_loss.item(),
                                                                                feature_loss.item(),
                                                                                triplet_loss.item(),
                                                           ))
    print("===> Epoch {} Complete: Avg. Loss: {:.3f} {:.3f} {:.3f} {:.3f}".format(epoch,
                                                             epoch_loss / len(training_data_loader),
                                                             epoch_recon_loss / len(training_data_loader),
                                                             epoch_feature_loss / len(training_data_loader),
                                                             epoch_triplet_loss / len(training_data_loader),
                                                             ))

def val(epoch):
    '''val'''
    model.eval()
    avg_loss = 0
    avg_recon_loss = 0
    avg_feature_loss = 0
    avg_triplet_loss = 0
    with torch.no_grad():
        for batch in val_data_loader:
            input1, input2, input3, pos_img, neg_img = batch[0].to(device).float(), batch[1].to(device).float(), \
                                               batch[2].to(device).float(), \
                                               batch[3].to(device).float(), batch[4].to(device).float()

            p1, p2, z1, z2, input1, recon, z_pos, z_neg = model(input1, input3, pos_img, neg_img)

            loss_dict = model.loss_function(p1, p2, z1, z2, input1, recon, z_pos, z_neg)
            loss = loss_dict['loss']
            recon_loss = loss_dict['Recon_loss']
            feature_loss = loss_dict['Feature_loss']
            triplet_loss = loss_dict['Triplet_loss']

            avg_loss += loss.item()
            avg_recon_loss += recon_loss.item()
            avg_feature_loss += feature_loss.item()
            avg_triplet_loss += triplet_loss.item()

    print("===> Avg. Validation Losses: {:.3f} {:.3f} {:.3f} {:.3f}".format(avg_loss / len(val_data_loader),
                                                                     avg_recon_loss / len(val_data_loader),
                                                                     avg_feature_loss / len(val_data_loader),
                                                                     avg_triplet_loss / len(val_data_loader),))
    return avg_loss

def checkpoint(best_val_loss, curr_loss):
    '''checkpoint'''
    if curr_loss < best_val_loss:
        model_out_path = "./model_checkpoints_simsiam/model_epoch_{}.pth".format(epoch)
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
    parser.add_argument('--nEpochs', type=int, default=700, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
    parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    opt = parser.parse_args()

    print(opt)

    torch.manual_seed(opt.seed)

    if not os.path.exists('./model_checkpoints_simsiam'):
        os.makedirs('./model_checkpoints_simsiam')

    torch.manual_seed(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('===> Loading datasets')
    training_data_loader, val_data_loader = get_loader(train_bs=opt.batchSize,
                                                       val_bs=opt.valBatchSize,
                                                       train_num_samples_per_epoch=200,
                                                       val_num_samples_per_epoch=80,
                                                       num_works=opt.threads)

    print('===> Building model')
    model = SimSiam(in_channels=2,latent_dim=1024).to(device)
    init_model_files = sorted([int(i.split('/')[-1].split('.')[0].split('_')[-1]) \
                               for i in glob.glob(os.path.join('./model_checkpoints_simsiam/', 'model_epoch_*.pth'))])
    if not init_model_files:
        start = 0
    else:
        start = init_model_files[-1]
        model_pth = os.path.join('./model_checkpoints_simsiam/','model_epoch_'+str(start)+'.pth')
        state_dict_pth = os.path.join('./model_checkpoints_simsiam/','model_epoch_'+str(start)+'.pt')
        loaded_model = torch.load(model_pth)
        state_dict = loaded_model.state_dict()
        torch.save(state_dict,state_dict_pth)
        model.load_state_dict(torch.load(state_dict_pth))
        print('loading initial model')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    best_val_loss = np.infty
    for epoch in range(start+1, opt.nEpochs + 1):
        train(epoch)
        val_loss = val(epoch)
        best_val_loss = checkpoint(best_val_loss,val_loss)




