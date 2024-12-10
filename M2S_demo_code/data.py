from sklearn import preprocessing
import torch
import os
import pickle
import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader
import random
import os.path as osp

class SEED_multi_modal_time():
    def __init__(self, file_name, time_window, is_training):
        npz_eeg = np.load(file_name + '.npz')
        npz_eye = pickle.load(open(file_name, 'rb'))

        eeg_train_data = pickle.loads(npz_eeg['train_data'])
        eeg_test_data = pickle.loads(npz_eeg['test_data'])
        eeg_train_label = npz_eeg['train_label']
        eeg_test_label = npz_eeg['test_label']
        eye_train_data = npz_eye['train_data_eye']
        eye_test_data = npz_eye['test_data_eye']

        eeg_X_train, eeg_X_test = [], []

        for key in list(eeg_train_data.keys()):
            eeg_X_train.append(eeg_train_data[key])
        for key in list(eeg_test_data.keys()):
            eeg_X_test.append(eeg_test_data[key])

        eeg_X_train = np.array(eeg_X_train).transpose(1, 0, 2)
        eeg_X_test = np.array(eeg_X_test).transpose(1, 0, 2)
        _, p, b = eeg_X_train.shape
        eeg_X_train = np.reshape(eeg_X_train, [eeg_X_train.shape[0], p * b])
        eeg_X_test = np.reshape(eeg_X_test, [eeg_X_test.shape[0], p * b])
        eeg_scaler = preprocessing.StandardScaler().fit(eeg_X_train)
        eeg_X_train = eeg_scaler.transform(eeg_X_train)
        eeg_X_test = eeg_scaler.transform(eeg_X_test)

        eye_scaler = preprocessing.StandardScaler().fit(eye_train_data)
        eye_train_data = eye_scaler.transform(eye_train_data)
        eye_test_data = eye_scaler.transform(eye_test_data)

        self.time_window = time_window

        if is_training:
            self.eeg = torch.from_numpy(eeg_X_train)
            self.eye = torch.from_numpy(eye_train_data)
            self.label = torch.from_numpy(eeg_train_label)
            self.dic = {}
            self.cnt = 0
            for i in range(self.label.shape[0] - time_window):
                if self.label[i] == self.label[i + time_window]:
                    self.dic[self.cnt] = i
                    self.cnt += 1
            self.len = self.cnt
        else:
            self.eeg = torch.from_numpy(eeg_X_test)
            self.eye = torch.from_numpy(eye_test_data)
            self.label = torch.from_numpy(eeg_test_label)
            self.dic = {}
            self.cnt = 0
            for i in range(self.label.shape[0] - time_window):
                if self.label[i] == self.label[i + time_window]:
                    self.dic[self.cnt] = i
                    self.cnt += 1
            self.len = self.cnt

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        start = self.dic[item]
        end = start + self.time_window
        return self.eeg[start:end], self.eye[start:end], self.label[start]