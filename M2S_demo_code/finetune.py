from M2S import Multi_to_Single, Classifier
# from utils import get_logger, AverageMeter, metricsContainer
from itertools import chain
from torch import nn
from copy import deepcopy
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
import torch.nn.functional as F
import os
from tqdm import tqdm
import torch
import warnings
from torch.utils.data import DataLoader
from data import SEED_multi_modal_time
import os.path as osp
import argparse
import sys
warnings.filterwarnings('ignore')

def experiment(args, train_loader, eval_loader, lr, weight_decay, model_save_path):
    y_dim = 3
    epochs = 200

    model = Multi_to_Single(eeg_dim=310, eye_dim=33, eeg_output_dim=128, eye_output_dim=128, embedding_dim=128).cuda()
    model_pt = 'pretrained_model.pt'
    model_file = osp.join(model_save_path, model_pt)
    model.load_state_dict(torch.load(model_file))
    for param in model.parameters():
        param.requires_grad = False

    cls = Classifier(class_num=y_dim).cuda()
    optimizer = torch.optim.Adam(chain(model.parameters(), cls.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))
    max_acc = 0
    max_kappa = 0
    max_model = None
    for epoch in tqdm(range(1, epochs+1)):
        train_epoch(args, model, cls, train_loader, optimizer, scheduler)
        acc, kappa = eval_epoch(args, model, cls, eval_loader)
        if acc > max_acc:
            max_acc = acc
            max_kappa = kappa
            max_model = [deepcopy(model), deepcopy(cls)]
    return max_acc, max_kappa, max_model

def to_train(models):
    for m in models:
        m = m.train()
def train_epoch(args, model, classifier, train_loader, optimizer, scheduler):
    models = [model, classifier]
    to_train(models)
    for n_iter, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        eeg, eye, label = batch_data
        eeg = eeg.float().cuda()
        eye = eye.float().cuda()
        label = label.long().cuda()
        if args.mode == 'eeg2eye' or args.mode == 'eeg2eeg':
            with torch.no_grad():
                eeg_out = model.EEG_finetune_Encoder(eeg)
            eeg_pred = classifier(eeg_out)
            loss = F.cross_entropy(eeg_pred, label)
        else:
            with torch.no_grad():
                eye_out = model.Eye_finetune_Encoder(eye)
            eye_pred = classifier(eye_out)
            loss = F.cross_entropy(eye_pred, label)

        loss.backward()
        optimizer.step()
        scheduler.step()

def to_eval(models):
    for m in models:
        m = m.eval()
def eval_epoch(args, model, classifier, eval_loader):
    model.eval()
    classifier.eval()
    labels = []
    predicted = []
    with torch.no_grad():
        for i, batch_data in enumerate(eval_loader):
            eeg, eye, label = batch_data
            eeg = eeg.float().cuda()
            eye = eye.float().cuda()
            if args.mode == 'eeg2eye' or args.mode == 'eye2eye':
                out = model.Eye_finetune_Encoder(eye)
                pred = classifier(out)
            else:
                out = model.EEG_finetune_Encoder(eeg)
                pred = classifier(out)

            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy()
            predicted.append(pred)
            labels.append(label)
    acc = balanced_accuracy_score(np.concatenate(labels), np.concatenate(predicted))
    kappa = cohen_kappa_score(np.concatenate(labels), np.concatenate(predicted))
    return acc, kappa

def save_models(Encoder, Classifier, path):
    state_dict = {
        'Encoder': Encoder.state_dict(),
        'Classifier': Classifier.state_dict()
    }
    torch.save(state_dict, path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eeg2eeg', choices=['eeg2eye', 'eeg2eeg', 'eye2eeg', 'eye2eye'])
    parser.add_argument("--gpu", type=int, default=0, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Running on GPU: {args.gpu}")

    file = 'demo'
    model_save_path = 'models_seed/{}'.format(file)
    train_dataset = SEED_multi_modal_time(file, time_window=5, is_training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    eval_dataset = SEED_multi_modal_time(file, time_window=5, is_training=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=512, shuffle=False)

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
    print(max_acc)
    print(max_kappa)
    save_pt = args.mode + '_model.pt'
    save_models(max_model[0], max_model[1], osp.join(model_save_path, save_pt))

