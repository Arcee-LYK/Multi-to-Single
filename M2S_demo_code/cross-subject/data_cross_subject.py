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
    def __init__(self, eeg_path, eye_path, file_name_list, time_window):
        EEG = []
        EYE = []
        Label = []
        for file_name in file_name_list:
            npz_eeg = np.load(os.path.join(eeg_path, file_name + '.npz'))
            npz_eye = pickle.load(open(os.path.join(eye_path, file_name), 'rb'))

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
            EEG.append(eeg_X_train)
            EEG.append(eeg_X_test)
            EYE.append(eye_train_data)
            EYE.append(eye_test_data)
            Label.append(eeg_train_label)
            Label.append(eeg_test_label)
        EEG = np.concatenate(EEG, axis=0)
        EYE = np.concatenate(EYE, axis=0)
        Label = np.concatenate(Label, axis=0)

        eeg_scaler = preprocessing.StandardScaler().fit(EEG)
        EEG = eeg_scaler.transform(EEG)

        eye_scaler = preprocessing.StandardScaler().fit(EYE)
        EYE = eye_scaler.transform(EYE)

        self.time_window = time_window

        self.eeg = torch.from_numpy(EEG)
        self.eye = torch.from_numpy(EYE)
        self.label = torch.from_numpy(Label)
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

class LabelFeature():
    def __init__(self, train_eeg, train_eye, train_label, time_window):
        self.train_eeg = train_eeg
        self.train_eye = train_eye
        self.train_label = train_label
        self.time_window = time_window

        self.eeg_feature_label = []
        self.eye_feature_label = []
        eeg_feature_label = []
        eye_feature_label = []
        while True:
            random_num = random.choice(np.where(self.train_label == 0)[0])
            if random_num + time_window < self.train_label.shape[0] and train_label[random_num] == train_label[
                random_num + self.time_window]:
                eeg_feature_label.append(self.train_eeg[random_num:(random_num + time_window)])
                eye_feature_label.append(self.train_eye[random_num:(random_num + time_window)])
                break
        while True:
            random_num = random.choice(np.where(self.train_label == 1)[0])
            if random_num + time_window < self.train_label.shape[0] and train_label[random_num] == train_label[
                random_num + self.time_window]:
                eeg_feature_label.append(self.train_eeg[random_num:(random_num + time_window)])
                eye_feature_label.append(self.train_eye[random_num:(random_num + time_window)])
                break
        while True:
            random_num = random.choice(np.where(self.train_label == 2)[0])
            if random_num + time_window < self.train_label.shape[0] and train_label[random_num] == train_label[
                random_num + self.time_window]:
                eeg_feature_label.append(self.train_eeg[random_num:(random_num + time_window)])
                eye_feature_label.append(self.train_eye[random_num:(random_num + time_window)])
                break

        self.eeg_feature_label = torch.stack(eeg_feature_label, dim=0).float()
        self.eye_feature_label = torch.stack(eye_feature_label, dim=0).float()

    def shuffle(self):
        eeg_feature_label = []
        eye_feature_label = []
        while True:
            random_num = random.choice(np.where(self.train_label == 0)[0])
            if random_num + self.time_window < self.train_label.shape[0] and self.train_label[random_num] == self.train_label[
                random_num + self.time_window]:
                eeg_feature_label.append(self.train_eeg[random_num:(random_num + self.time_window)])
                eye_feature_label.append(self.train_eye[random_num:(random_num + self.time_window)])
                break
        while True:
            random_num = random.choice(np.where(self.train_label == 1)[0])
            if random_num + self.time_window < self.train_label.shape[0] and self.train_label[random_num] == self.train_label[
                random_num + self.time_window]:
                eeg_feature_label.append(self.train_eeg[random_num:(random_num + self.time_window)])
                eye_feature_label.append(self.train_eye[random_num:(random_num + self.time_window)])
                break
        while True:
            random_num = random.choice(np.where(self.train_label == 2)[0])
            if random_num + self.time_window < self.train_label.shape[0] and self.train_label[random_num] == self.train_label[
                random_num + self.time_window]:
                eeg_feature_label.append(self.train_eeg[random_num:(random_num + self.time_window)])
                eye_feature_label.append(self.train_eye[random_num:(random_num + self.time_window)])
                break
        self.eeg_feature_label = torch.stack(eeg_feature_label, dim=0).float()
        self.eye_feature_label = torch.stack(eye_feature_label, dim=0).float()