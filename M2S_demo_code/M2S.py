import torch
import torch.nn as nn
import torch.nn.functional as F
from M2MCPC import M2M_CPC
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np
import math
from models import EncoderLayer, Encoder, EEG_EncoderLayer
from torch import Tensor
import random
from einops import rearrange, repeat

class EEG_ER_Encoder(nn.Module):
    def __init__(self, input_dim, d_model):
        super(EEG_ER_Encoder, self).__init__()
        self.encoder_layer = EEG_EncoderLayer(d_model=input_dim, nhead=4)
        self.encoder = Encoder(self.encoder_layer, num_layers=4)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        feature = self.encoder(feature)
        feature = self.affine_matrix(feature)
        return feature

class Eye_ER_Encoder(nn.Module):
    def __init__(self, input_dim, d_model):
        super(Eye_ER_Encoder, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
        self.encoder = Encoder(self.encoder_layer, num_layers=4)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature):
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)
        return feature

class Classifier(nn.Module):
    def __init__(self,  class_num):
        super(Classifier, self).__init__()

        self.event_classifier = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, class_num)
        )

    def forward(self, input_vq):

        return self.event_classifier(input_vq)

class EEG_EI_Encoder(nn.Module):
    def __init__(self, eeg_dim, hidden_dim):
        super(EEG_EI_Encoder, self).__init__()
        self.eeg_dim = eeg_dim
        self.hidden_dim = hidden_dim
        self.eeg_linear = nn.Linear(eeg_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, eeg_feat):
        return self.relu(self.eeg_linear(eeg_feat))

class Eye_EI_Encoder(nn.Module):
    def __init__(self, eye_dim, hidden_dim):
        super(Eye_EI_Encoder, self).__init__()
        self.eye_dim = eye_dim
        self.hidden_dim = hidden_dim
        self.eye_linear = nn.Linear(eye_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, eye_feat):
        return self.relu(self.eye_linear(eye_feat))

class EEG_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super(EEG_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.eeg_rec = nn.Linear(input_dim * 2, output_dim)
        self.eeg_linear = nn.Linear(vq_dim, input_dim)

    def forward(self, eeg_semantic, eeg_encoder_result):
        eeg_vq_result = self.eeg_linear(eeg_semantic)
        eeg_encoder_result = torch.cat((eeg_vq_result, eeg_encoder_result), dim=2)
        eeg_decoder_result = self.eeg_rec(eeg_encoder_result)
        return eeg_decoder_result

class Eye_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super(Eye_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.eye_rec = nn.Linear(input_dim * 2, output_dim)
        self.eye_linear = nn.Linear(vq_dim, input_dim)

    def forward(self, eye_semantic, eye_encoder_result):
        eye_vq_result = self.eye_linear(eye_semantic)
        eye_encoder_result = torch.cat((eye_vq_result, eye_encoder_result), dim=2)
        eye_decoder_result = self.eye_rec(eye_encoder_result)
        return eye_decoder_result

class Multi_to_Single(nn.Module):
    def __init__(self, eeg_dim, eye_dim, eeg_output_dim, eye_output_dim, embedding_dim):
        super(Multi_to_Single, self).__init__()
        self.eeg_dim = eeg_dim
        self.eye_dim = eye_dim
        self.hidden_dim = embedding_dim

        self.cpc = M2M_CPC(embedding_dim, 64, 64, 2, 2, 1)
        self.EEG_ER_encoder = EEG_ER_Encoder(input_dim=eeg_dim, d_model=self.hidden_dim)
        self.Eye_ER_encoder = Eye_ER_Encoder(input_dim=eye_dim, d_model=self.hidden_dim)

        self.EEG_EI_encoder = EEG_EI_Encoder(eeg_dim, eeg_output_dim)
        self.Eye_EI_encoder = Eye_EI_Encoder(eye_dim, eye_output_dim)

        self.eeg_decoder = EEG_Decoder(eeg_output_dim, eeg_dim, self.hidden_dim)
        self.eye_decoder = Eye_Decoder(eye_output_dim, eye_dim, self.hidden_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.eeg_transform = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU()
        )
        self.eye_transform = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU()
        )

    def EEG_finetune_Encoder(self, eeg_feat):
        eeg_feat = eeg_feat.cuda()
        eeg_semantic_result = self.EEG_ER_encoder(eeg_feat)
        eeg_semantic_result = eeg_semantic_result.mean(dim=1)
        eeg_semantic_result = self.eeg_transform(eeg_semantic_result)
        return eeg_semantic_result

    def Eye_finetune_Encoder(self, eye_feat):
        eye_feat = eye_feat.cuda()
        eye_semantic_result = self.Eye_ER_encoder(eye_feat)
        eye_semantic_result = eye_semantic_result.mean(dim=1)
        eye_semantic_result = self.eye_transform(eye_semantic_result)
        return eye_semantic_result

    def same_eeg_modal(self, eeg_feat, eeg_label):
        eeg_semantic = self.EEG_ER_encoder(eeg_feat)
        eeg_label_semantic = self.EEG_ER_encoder(eeg_label)
        eeg_semantic_result = eeg_semantic.mean(dim=1)
        eeg_label_semantic_result = eeg_label_semantic.mean(dim=1)

        eeg_semantic_result = self.eeg_transform(eeg_semantic_result)
        eeg_label_semantic_result = self.eeg_transform(eeg_label_semantic_result)

        eeg_semantic_result = eeg_semantic_result / eeg_semantic_result.norm(dim=-1, keepdim=True)
        eeg_label_semantic_result = eeg_label_semantic_result / eeg_label_semantic_result.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_eeg_global = logit_scale * eeg_semantic_result @ eeg_label_semantic_result.t()
        return logits_per_eeg_global

    def same_eye_modal(self, eye_feat, eye_label):
        eye_semantic = self.Eye_ER_encoder(eye_feat)
        eye_label_semantic = self.Eye_ER_encoder(eye_label)
        eye_semantic_result = eye_semantic.mean(dim=1)
        eye_label_semantic_result = eye_label_semantic.mean(dim=1)

        eye_semantic_result = self.eye_transform(eye_semantic_result)
        eye_label_semantic_result = self.eye_transform(eye_label_semantic_result)

        eye_semantic_result = eye_semantic_result / eye_semantic_result.norm(dim=-1, keepdim=True)
        eye_label_semantic_result = eye_label_semantic_result / eye_label_semantic_result.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_eye_global = logit_scale * eye_semantic_result @ eye_label_semantic_result.t()
        return logits_per_eye_global

    def single_modal_forward(self, eeg_feat, eye_feat):
        eeg_semantic = self.EEG_ER_encoder(eeg_feat)
        eye_semantic = self.Eye_ER_encoder(eye_feat)
        eeg_semantic_result = eeg_semantic.mean(dim=1)
        eye_semantic_result = eye_semantic.mean(dim=1)

        eeg_semantic_result = self.eeg_transform(eeg_semantic_result)
        eye_semantic_result = self.eye_transform(eye_semantic_result)

        eeg_semantic_result = eeg_semantic_result / eeg_semantic_result.norm(dim=-1, keepdim=True)
        eye_semantic_result = eye_semantic_result / eye_semantic_result.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_eeg_global = logit_scale * eeg_semantic_result @ eye_semantic_result.t()
        logits_per_eye_global = logits_per_eeg_global.t()

        return logits_per_eeg_global ,logits_per_eye_global

    def forward(self, eeg_feat, eye_feat):
        # print("eeg_feat", eeg_feat.shape)
        eeg_ER_result = self.EEG_ER_encoder(eeg_feat)
        # print("eeg_semantic", eeg_semantic.shape)
        eye_ER_result = self.Eye_ER_encoder(eye_feat)

        eeg_encoder_result = self.EEG_EI_encoder(eeg_feat)
        eye_encoder_result = self.Eye_EI_encoder(eye_feat)

        eeg_recon_result = self.eeg_decoder(eeg_ER_result, eeg_encoder_result)
        eye_recon_result = self.eye_decoder(eye_ER_result, eye_encoder_result)
        eeg_recon_loss = F.mse_loss(eeg_recon_result, eeg_feat)
        eye_recon_loss = F.mse_loss(eye_recon_result, eye_feat)

        cpc_loss = self.cpc(eeg_ER_result, eye_ER_result)

        recon_loss = 0.1 * (eeg_recon_loss + eye_recon_loss)

        eeg_semantic_result = eeg_ER_result.mean(dim=1)
        eye_semantic_result = eye_ER_result.mean(dim=1)

        eeg_semantic_result = self.eeg_transform(eeg_semantic_result)
        eye_semantic_result = self.eye_transform(eye_semantic_result)

        eeg_semantic_result = eeg_semantic_result / eeg_semantic_result.norm(dim=-1, keepdim=True)
        eye_semantic_result = eye_semantic_result / eye_semantic_result.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_eeg_global = logit_scale * eeg_semantic_result @ eye_semantic_result.t()
        logits_per_eye_global = logits_per_eeg_global.t()

        return cpc_loss, recon_loss, logits_per_eeg_global, logits_per_eye_global, eeg_ER_result, eye_ER_result, \
                                                     eeg_encoder_result, eye_encoder_result


