from main_model import EE_VQVAE_Encoder
# from ..utils import get_logger, AverageMeter, metricsContainer
from itertools import chain
from CLUB import CLUBSample_group
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import os
from tqdm import tqdm
import torch
import warnings
from torch.utils.data import DataLoader
from data_cross_subject import SEED_multi_modal_time, LabelFeature
import os.path as osp
import argparse
import sys
from copy import deepcopy
warnings.filterwarnings('ignore')

def experiment(args, train_loader, eval_loader, lr, weight_decay, label_feature):
    epochs = 200

    model = EE_VQVAE_Encoder(eeg_dim=310, eye_dim=33, eeg_output_dim=128, eye_output_dim=128, embedding_dim=128).cuda()
    EEG_mi_net = CLUBSample_group(x_dim=128, y_dim=128, hidden_dim=64).cuda()
    EYE_mi_net = CLUBSample_group(x_dim=128, y_dim=128, hidden_dim=64).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_eeg_mi_net = torch.optim.Adam(EEG_mi_net.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_eye_mi_net = torch.optim.Adam(EYE_mi_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))
    max_eeg2eye_acc = 0
    max_eeg2eeg_acc = 0
    max_eye2eeg_acc = 0
    max_eye2eye_acc = 0
    max_eeg2eye_model = None
    max_eeg2eeg_model = None
    max_eye2eeg_model = None
    max_eye2eye_model = None
    for epoch in tqdm(range(1, epochs+1)):
        train_epoch(args, model, train_loader, optimizer, scheduler, EEG_mi_net, EYE_mi_net, optimizer_eeg_mi_net, optimizer_eye_mi_net)
        eeg2eye_acc, eeg2eeg_acc, eye2eeg_acc, eye2eye_acc = eval_epoch(model, eval_loader, label_feature)
        if eeg2eye_acc > max_eeg2eye_acc:
            max_eeg2eye_acc = eeg2eye_acc
            max_eeg2eye_model = deepcopy(model.state_dict())
        if eeg2eeg_acc > max_eeg2eeg_acc:
            max_eeg2eeg_acc = eeg2eeg_acc
            max_eeg2eeg_model = deepcopy(model.state_dict())
        if eye2eeg_acc > max_eye2eeg_acc:
            max_eye2eeg_acc = eye2eeg_acc
            max_eye2eeg_model = deepcopy(model.state_dict())
        if eye2eye_acc > max_eye2eye_acc:
            max_eye2eye_acc = eye2eye_acc
            max_eye2eye_model = deepcopy(model.state_dict())
    return max_eeg2eye_acc, max_eeg2eye_model, max_eeg2eeg_acc, max_eeg2eeg_model, \
           max_eye2eeg_acc, max_eye2eeg_model, max_eye2eye_acc, max_eye2eye_model

def train_epoch(args, model, train_loader, optimizer, scheduler, EEG_mi_net, EYE_mi_net, optimizer_eeg_mi_net, optimizer_eye_mi_net):
    model.train()
    loss_eeg = nn.CrossEntropyLoss().cuda()
    loss_eye = nn.CrossEntropyLoss().cuda()
    mi_iters = 3
    for n_iter, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        eeg, eye, label = batch_data
        eeg = eeg.float().cuda()
        eye = eye.float().cuda()
        if args.ablation_loss == 'mi':
            cpc_loss, recon_loss, logits_eeg, logits_eye, _, _, _, _ = model(eeg, eye)
            mi_loss = None
        else:
            for i in range(mi_iters):
                optimizer_eeg_mi_net, optimizer_eye_mi_net = mi_first_forward(eeg, eye, model, EEG_mi_net, EYE_mi_net, optimizer_eeg_mi_net, optimizer_eye_mi_net)

            mi_eeg_loss, mi_eye_loss, cpc_loss, recon_loss, logits_eeg, logits_eye = mi_second_forward(eeg, eye, model, EEG_mi_net, EYE_mi_net)
            mi_loss = 0.3 * (mi_eeg_loss + mi_eye_loss)

        ground_truth = torch.arange(label.shape[0]).long().cuda()
        contra_loss = (loss_eeg(logits_eeg, ground_truth) + loss_eye(logits_eye, ground_truth)) / 2

        if args.ablation_loss == 'none':
            loss = contra_loss + cpc_loss + recon_loss + mi_loss
        elif args.ablation_loss == 'cpc':
            loss = contra_loss + recon_loss + mi_loss
        elif args.ablation_loss == 'recon':
            loss = contra_loss + cpc_loss + mi_loss
        elif args.ablation_loss == 'contra':
            loss = mi_loss + cpc_loss + recon_loss
        else:
            loss = cpc_loss + recon_loss + contra_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

def eval_epoch(model, eval_loader, label_feature):
    model.eval()
    with torch.no_grad():
        eeg2eye_accuracies = []
        eeg2eeg_accuracies = []
        eye2eeg_accuracies = []
        eye2eye_accuracies = []
        for _ in range(10):
            for i, batch_data in enumerate(eval_loader):
                eeg, eye, label = batch_data
                eeg = eeg.float().cuda()
                eye = eye.float().cuda()

                eeg_feature_label = label_feature.eeg_feature_label.cuda()
                _, eeg2eye_out = model.single_modal_forward(eeg_feature_label, eye)
                eeg2eeg_out = model.same_eeg_modal(eeg, eeg_feature_label)

                eye_feature_label = label_feature.eye_feature_label.cuda()
                eye2eeg_out, _ = model.single_modal_forward(eeg, eye_feature_label)
                eye2eye_out = model.same_eye_modal(eye, eye_feature_label)

                eeg2eye_pred = eeg2eye_out.softmax(dim=-1).cpu()
                eeg2eeg_pred = eeg2eeg_out.softmax(dim=-1).cpu()
                eye2eeg_pred = eye2eeg_out.softmax(dim=-1).cpu()
                eye2eye_pred = eye2eye_out.softmax(dim=-1).cpu()
                eeg2eye_pred = torch.argmax(eeg2eye_pred, dim=-1)
                eeg2eeg_pred = torch.argmax(eeg2eeg_pred, dim=-1)
                eye2eeg_pred = torch.argmax(eye2eeg_pred, dim=-1)
                eye2eye_pred = torch.argmax(eye2eye_pred, dim=-1)

                eeg2eye_accuracies.append(accuracy_score(label, eeg2eye_pred))
                eeg2eeg_accuracies.append(accuracy_score(label, eeg2eeg_pred))
                eye2eeg_accuracies.append(accuracy_score(label, eye2eeg_pred))
                eye2eye_accuracies.append(accuracy_score(label, eye2eye_pred))
                label_feature.shuffle()
        eeg2eye_acc = np.mean(eeg2eye_accuracies)
        eeg2eeg_acc = np.mean(eeg2eeg_accuracies)
        eye2eeg_acc = np.mean(eye2eeg_accuracies)
        eye2eye_acc = np.mean(eye2eye_accuracies)
    return eeg2eye_acc, eeg2eeg_acc, eye2eeg_acc, eye2eye_acc

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
    eeg_path = '../../data/SEED/EEG/eeg_used_4s/'
    eye_path = '../../data/SEED/Eye/eye_tracking_feature/'
    file_list = ['1_1', '2_1', '3_1', '4_1', '5_1', '8_1', '9_1', '10_1', '11_1', '12_1', '13_1', '14_1']

    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_loss', type=str, default='none', choices=['none', 'cpc', 'recon','contra', 'mi'])
    parser.add_argument("--gpu", type=int, default=1, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Running on GPU: {args.gpu}")
    # print(file_list)

    for file in file_list:
        target_list = [file]
        source_list = deepcopy(file_list)
        source_list.remove(file)
        model_save_path = 'models_seed/{}/{}'.format(args.ablation_loss, file)
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        train_dataset = SEED_multi_modal_time(eeg_path, eye_path, source_list, time_window=5)
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        eval_dataset = SEED_multi_modal_time(eeg_path, eye_path, target_list, time_window=5)
        eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

        label_feature = LabelFeature(train_dataset.eeg, train_dataset.eye, train_dataset.label, time_window=5)
        max_eeg2eye_acc = 0
        max_eeg2eeg_acc = 0
        max_eye2eeg_acc = 0
        max_eye2eye_acc = 0
        max_eeg2eye_model = None
        max_eeg2eeg_model = None
        max_eye2eeg_model = None
        max_eye2eye_model = None
        for lr in [1e-4]:
            for weight_decay in [1e-3]:
                eeg2eye_acc, eeg2eye_model, eeg2eeg_acc, eeg2eeg_model, eye2eeg_acc, eye2eeg_model, \
                eye2eye_acc, eye2eye_model = experiment(args, train_dataloader, eval_dataloader, lr, weight_decay, label_feature)
                if eeg2eye_acc > max_eeg2eye_acc:
                    max_eeg2eye_acc = eeg2eye_acc
                    max_eeg2eye_model = eeg2eye_model
                if eeg2eeg_acc > max_eeg2eeg_acc:
                    max_eeg2eeg_acc = eeg2eeg_acc
                    max_eeg2eeg_model = eeg2eeg_model
                if eye2eeg_acc > max_eye2eeg_acc:
                    max_eye2eeg_acc = eye2eeg_acc
                    max_eye2eeg_model = eye2eeg_model
                if eye2eye_acc > max_eye2eye_acc:
                    max_eye2eye_acc = eye2eye_acc
                    max_eye2eye_model = eye2eye_model
        torch.save(max_eeg2eye_model, osp.join(model_save_path, 'eeg2eye_pretrained_model.pt'))
        torch.save(max_eeg2eeg_model, osp.join(model_save_path, 'eeg2eeg_pretrained_model.pt'))
        torch.save(max_eye2eeg_model, osp.join(model_save_path, 'eye2eeg_pretrained_model.pt'))
        torch.save(max_eye2eye_model, osp.join(model_save_path, 'eye2eye_pretrained_model.pt'))