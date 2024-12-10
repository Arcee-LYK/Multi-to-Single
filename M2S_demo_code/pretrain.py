from M2S import Multi_to_Single
from itertools import chain
from CLUB import CLUBSample_group
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm
import torch
import warnings
from torch.utils.data import DataLoader
from data import SEED_multi_modal_time
import os.path as osp
import argparse
warnings.filterwarnings('ignore')

def experiment(args, train_loader):
    epochs = 200
    model = Multi_to_Single(eeg_dim=310, eye_dim=33, eeg_output_dim=128, eye_output_dim=128, embedding_dim=128).cuda()
    EEG_mi_net = CLUBSample_group(x_dim=128, y_dim=128, hidden_dim=64).cuda()
    EYE_mi_net = CLUBSample_group(x_dim=128, y_dim=128, hidden_dim=64).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_eeg_mi_net = torch.optim.Adam(EEG_mi_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_eye_mi_net = torch.optim.Adam(EYE_mi_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))

    for epoch in tqdm(range(1, epochs+1)):
        train_epoch(model, train_loader, optimizer, scheduler, EEG_mi_net, EYE_mi_net, optimizer_eeg_mi_net, optimizer_eye_mi_net)
    return model.state_dict()

def train_epoch(model, train_loader, optimizer, scheduler, EEG_mi_net, EYE_mi_net, optimizer_eeg_mi_net, optimizer_eye_mi_net):
    model.train()
    loss_eeg = nn.CrossEntropyLoss().cuda()
    loss_eye = nn.CrossEntropyLoss().cuda()
    mi_iters = 3
    for n_iter, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        eeg, eye, label = batch_data
        eeg = eeg.float().cuda()
        eye = eye.float().cuda()

        for i in range(mi_iters):
            optimizer_eeg_mi_net, optimizer_eye_mi_net = mi_first_forward(eeg, eye, model, EEG_mi_net, EYE_mi_net, optimizer_eeg_mi_net, optimizer_eye_mi_net)

        mi_eeg_loss, mi_eye_loss, cpc_loss, recon_loss, logits_eeg, logits_eye = mi_second_forward(eeg, eye, model, EEG_mi_net, EYE_mi_net)
        mi_loss = 0.3 * (mi_eeg_loss + mi_eye_loss)

        ground_truth = torch.arange(label.shape[0]).long().cuda()
        contra_loss = (loss_eeg(logits_eeg, ground_truth) + loss_eye(logits_eye, ground_truth)) / 2
        loss = contra_loss + cpc_loss + recon_loss + mi_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

def mi_first_forward(eeg_feature, eye_feature, model, EEG_mi_net, EYE_mi_net, optimizer_eeg_mi_net, optimizer_eye_mi_net):
    optimizer_eeg_mi_net.zero_grad()
    optimizer_eye_mi_net.zero_grad()
    cpc_loss, recon_loss, logits_eeg, logits_eye, eeg_semantic, eye_semantic, eeg_encoder_result, eye_encoder_result \
                                                  = model(eeg_feature, eye_feature)
    eeg_encoder_result = eeg_encoder_result.detach()
    eye_encoder_result = eye_encoder_result.detach()
    eeg_semantic = eeg_semantic.detach()
    eye_semantic = eye_semantic.detach()

    lld_eeg_loss = -EEG_mi_net.loglikeli(eeg_semantic, eeg_encoder_result)
    lld_eeg_loss.backward()
    optimizer_eeg_mi_net.step()

    lld_eye_loss = -EYE_mi_net.loglikeli(eye_semantic, eye_encoder_result)
    lld_eye_loss.backward()
    optimizer_eye_mi_net.step()

    return optimizer_eeg_mi_net, optimizer_eye_mi_net

def mi_second_forward(eeg_feature, eye_feature, model, EEG_mi_net, EYE_mi_net):
    cpc_loss, recon_loss, logits_eeg, logits_eye, eeg_semantic, eye_semantic, eeg_encoder_result, eye_encoder_result \
                                  = model(eeg_feature, eye_feature)
    mi_eeg_loss = EEG_mi_net.mi_est(eeg_semantic, eeg_encoder_result)
    mi_eye_loss = EYE_mi_net.mi_est(eye_semantic, eye_encoder_result)
    return mi_eeg_loss, mi_eye_loss, cpc_loss, recon_loss, logits_eeg, logits_eye

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--weight_decay', default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Running on GPU: {args.gpu}")
    #demo.npz  EEG data
    #demo EYE data
    file = 'demo'
    model_save_path = 'models_seed/{}'.format(file)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    train_dataset = SEED_multi_modal_time(file, time_window=5, is_training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    model = experiment(args, train_dataloader)
    torch.save(model, osp.join(model_save_path, 'pretrained_model.pt'))