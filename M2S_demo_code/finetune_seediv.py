from main_model import EE_VQVAE_Encoder, Semantic_Decoder
# from utils import get_logger, AverageMeter, metricsContainer
from itertools import chain
from torch import nn
from copy import deepcopy
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
import torch.nn.functional as F
import os
from tqdm import tqdm
import torch
import warnings
from torch.utils.data import DataLoader
from data import SEEDIV_multi_modal_time
import os.path as osp
import argparse
import sys
warnings.filterwarnings('ignore')

def experiment(args, train_loader, eval_loader, lr, weight_decay, model_save_path):
    y_dim = 4
    epochs = 200

    model = EE_VQVAE_Encoder(eeg_dim=310, eye_dim=31, eeg_output_dim=128, eye_output_dim=128, embedding_dim=128).cuda()
    model_pt = args.mode + '_pretrained_model.pt'
    model_file = osp.join(model_save_path, model_pt)
    model.load_state_dict(torch.load(model_file))
    for param in model.parameters():
        param.requires_grad = False

    decoder = Semantic_Decoder(class_num=y_dim).cuda()

    optimizer = torch.optim.Adam(chain(filter(lambda p: p.requires_grad, model.parameters()), decoder.parameters()),
                                 lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    max_acc = 0
    max_kappa = 0
    max_model = None
    for epoch in tqdm(range(1, epochs + 1)):
        train_epoch(args, model, decoder, train_loader, optimizer, scheduler)
        acc, kappa = eval_epoch(args, model, decoder, eval_loader)
        if acc > max_acc:
            max_acc = acc
            max_kappa = kappa
            max_model = [deepcopy(model), deepcopy(decoder)]
    return max_acc, max_kappa, max_model


def to_train(models):
    for m in models:
        m = m.train()


def train_epoch(args, model, decoder, train_loader, optimizer, scheduler):
    models = [model, decoder]
    to_train(models)
    for n_iter, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        eeg, eye, label = batch_data
        eeg = eeg.float().cuda()
        eye = eye.float().cuda()
        label = label.long().cuda()
        if args.mode == 'eeg2eye' or args.mode == 'eeg2eeg':
            eeg_vq = model.EEG_VQ_Encoder(eeg)
            eeg_pred = decoder(eeg_vq)
            loss = F.cross_entropy(eeg_pred, label)
        else:
            eye_vq = model.Eye_VQ_Encoder(eye)
            eye_pred = decoder(eye_vq)
            loss = F.cross_entropy(eye_pred, label)

        loss.backward()
        optimizer.step()
        scheduler.step()

def to_eval(models):
    for m in models:
        m = m.eval()

def eval_epoch(args, model, decoder, eval_loader):
    model.eval()
    decoder.eval()
    labels = []
    predicted = []
    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            eeg, eye, label = batch_data
            eeg = eeg.float().cuda()
            eye = eye.float().cuda()
            if args.mode == 'eeg2eye' or args.mode == 'eye2eye':
                vq = model.Eye_VQ_Encoder(eye)
                pred = decoder(vq)
            else:
                vq = model.EEG_VQ_Encoder(eeg)
                pred = decoder(vq)

            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
            predicted.append(pred)
            labels.append(label)
    acc = accuracy_score(np.concatenate(labels), np.concatenate(predicted))
    kappa = cohen_kappa_score(np.concatenate(labels), np.concatenate(predicted))
    return acc, kappa


def save_models(Encoder, Decoder, path):
    state_dict = {
        'Encoder': Encoder.state_dict(),
        'Decoder': Decoder.state_dict()
    }
    torch.save(state_dict, path)


if __name__ == '__main__':
    eeg_path = '../../data/SEED_IV/eeg_feature_smooth'
    eye_path = '../../data/SEED_IV/eye_feature_smooth'
    file_list = os.listdir(eye_path)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eeg2eye', choices=['eeg2eye', 'eeg2eeg', 'eye2eeg', 'eye2eye'])
    parser.add_argument('--ablation_loss', type=str, default='none', choices=['none', 'cpc', 'mi', 'recon', 'contra'])
    parser.add_argument("--gpu", type=int, default=0, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Running on GPU: {args.gpu}")

    final_acc = []
    final_kappa = []
    for session in [1, 2, 3]:
        eeg_new_path = osp.join(eeg_path, str(session))
        eye_new_path = osp.join(eye_path, str(session))
        file_list = os.listdir(eye_new_path)
        for file in file_list:
            model_save_path = 'models_seediv/{}/{}_{}'.format(args.ablation_loss, str(session), file)
            train_dataset = SEEDIV_multi_modal_time(eeg_new_path, eye_new_path, file, session=session, time_window=5, is_training=True)
            train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            eval_dataset = SEEDIV_multi_modal_time(eeg_new_path, eye_new_path, file, session=session, time_window=5, is_training=False)
            eval_dataloader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

            max_acc = 0
            max_kappa = 0
            max_model = None
            for lr in [1e-3, 1e-4, 1e-5]:
                for weight_decay in [1e-6, 1e-4, 1e-2]:
                    acc, kappa, model = \
                        experiment(args, train_dataloader, eval_dataloader, lr, weight_decay, model_save_path)
                    if acc > max_acc:
                        max_acc = acc
                        max_kappa = kappa
                        max_model = model
            save_pt = args.mode + '_model.pt'
            save_models(max_model[0], max_model[1], osp.join(model_save_path, save_pt))

            final_acc.append(max_acc)
            final_kappa.append(max_kappa)
    print("-------------")
    print(np.mean(final_acc), np.std(final_acc))
    print(np.mean(final_kappa), np.std(final_kappa))

    txt_file = 'seediv_{}_{}.txt'.format(args.mode, args.ablation_loss)
    result = [np.mean(final_acc), np.std(final_acc)]
    kappa_result = [np.mean(final_kappa), np.std(final_kappa)]
    with open(txt_file, 'w') as txtfile:
        txtfile.write(' '.join(map(str, result)) + '\n')
        txtfile.write(' '.join(map(str, kappa_result)) + '\n')