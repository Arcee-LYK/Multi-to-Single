from sklearn import preprocessing
import torch
import os
import pickle
import numpy as np
import scipy.io as scio
from torch.utils.data import DataLoader
import random
import os.path as osp
class SEED_multi_modal():
    def __init__(self, eeg_path, eye_path, file_name, is_training):
        npz_eeg = np.load(os.path.join(eeg_path, file_name+'.npz'))
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

        eeg_X_train = np.array(eeg_X_train).transpose(1,0,2)
        eeg_X_test = np.array(eeg_X_test).transpose(1,0,2)
        # eeg_X_train = np.pad(eeg_X_train, ((0, 0), (0, 0), (1, 1)), mode='mean')
        # eeg_X_test = np.pad(eeg_X_test, ((0,0),(0,0),(1,1)), mode='mean')
        _, p, b = eeg_X_train.shape
        eeg_X_train = np.reshape(eeg_X_train, [eeg_X_train.shape[0], p * b])
        eeg_X_test = np.reshape(eeg_X_test, [eeg_X_test.shape[0], p * b])
        eeg_scaler = preprocessing.StandardScaler().fit(eeg_X_train)
        eeg_X_train = eeg_scaler.transform(eeg_X_train)
        eeg_X_test = eeg_scaler.transform(eeg_X_test)
        # eeg_X_train = np.reshape(eeg_X_train, [eeg_X_train.shape[0], p, b])
        # eeg_X_test = np.reshape(eeg_X_test, [eeg_X_test.shape[0], p, b])

        eye_scaler = preprocessing.StandardScaler().fit(eye_train_data)
        eye_train_data = eye_scaler.transform(eye_train_data)
        eye_test_data = eye_scaler.transform(eye_test_data)

        if is_training:
            self.eeg = torch.from_numpy(eeg_X_train)
            self.eye = torch.from_numpy(eye_train_data)
            self.label = torch.from_numpy(eeg_train_label)
            self.len = eeg_X_train.shape[0]
        else:
            self.eeg = torch.from_numpy(eeg_X_test)
            self.eye = torch.from_numpy(eye_test_data)
            self.label = torch.from_numpy(eeg_test_label)
            self.len = eeg_X_test.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.eye[item], self.label[item]

class SEED_multi_modal_time():
    def __init__(self, eeg_path, eye_path, file_name, time_window, is_training):
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

class SEEDIV_multi_modal_time():
    def __init__(self, eeg_path, eye_path, file, session, time_window, is_training=True):
        self.session_label = {
            1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
            2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
            3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
        }
        eeg_data_path = osp.join(eeg_path, file)
        eye_data_path = osp.join(eye_path, file)

        eeg_data = scio.loadmat(eeg_data_path)
        eye_data = scio.loadmat(eye_data_path)
        eeg_train = []
        eeg_test = []
        eye_train = []
        eye_test = []
        label_train = []
        label_test = []
        for i in range(16):
            eeg = eeg_data[f'de_LDS{i + 1}'].astype('float')
            eye = eye_data[f'eye_{i + 1}'].astype('float')
            print(eeg.shape)
            eeg_train.append(eeg)
            eye_train.append(eye)
            for _ in range(eeg.shape[1]):
                label_train.append(self.session_label[session][i])
        eeg_train = np.concatenate(eeg_train, axis=1).transpose(1, 0, 2)
        eye_train = np.concatenate(eye_train, axis=1).transpose(1, 0)
        label_train = np.array(label_train)

        for i in range(16, 24):
            eeg = eeg_data[f'de_LDS{i + 1}'].astype('float')
            eye = eye_data[f'eye_{i + 1}'].astype('float')
            eeg_test.append(eeg)
            eye_test.append(eye)
            for j in range(eeg.shape[1]):
                label_test.append(self.session_label[session][i])
        eeg_test = np.concatenate(eeg_test, axis=1).transpose(1, 0, 2)
        eye_test = np.concatenate(eye_test, axis=1).transpose(1, 0)
        label_test = np.array(label_test)

        _, p, b = eeg_train.shape
        eeg_train = np.reshape(eeg_train, [eeg_train.shape[0], p * b])
        eeg_test = np.reshape(eeg_test, [eeg_test.shape[0], p * b])
        eeg_scaler = preprocessing.StandardScaler().fit(eeg_train)
        eeg_train = eeg_scaler.transform(eeg_train)
        eeg_test = eeg_scaler.transform(eeg_test)

        eye_scaler = preprocessing.StandardScaler().fit(eye_train)
        eye_train = eye_scaler.transform(eye_train)
        eye_test = eye_scaler.transform(eye_test)

        self.time_window = time_window

        if is_training:
            self.eeg = torch.from_numpy(eeg_train)
            self.eye = torch.from_numpy(eye_train)
            self.label = torch.from_numpy(label_train)
            self.dic = {}
            self.cnt = 0
            for i in range(self.label.shape[0] - time_window):
                if self.label[i] == self.label[i + time_window]:
                    self.dic[self.cnt] = i
                    self.cnt += 1
            self.len = self.cnt
        else:
            self.eeg = torch.from_numpy(eeg_test)
            self.eye = torch.from_numpy(eye_test)
            self.label = torch.from_numpy(label_test)
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

class SEEDIV_multi_modal():
    def __init__(self, eeg_path, eye_path, file, session, is_training=True):
        self.session_label = {
            1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
            2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
            3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
        }
        eeg_data_path = osp.join(eeg_path, file)
        eye_data_path = osp.join(eye_path, file)

        eeg_data = scio.loadmat(eeg_data_path)
        eye_data = scio.loadmat(eye_data_path)
        eeg_train = []
        eeg_test = []
        eye_train = []
        eye_test = []
        label_train = []
        label_test = []
        for i in range(16):
            eeg = eeg_data[f'de_LDS{i + 1}'].astype('float')
            eye = eye_data[f'eye_{i + 1}'].astype('float')
            eeg_train.append(eeg)
            eye_train.append(eye)
            for _ in range(eeg.shape[1]):
                label_train.append(self.session_label[session][i])
        eeg_train = np.concatenate(eeg_train, axis=1).transpose(1, 0, 2)
        eye_train = np.concatenate(eye_train, axis=1).transpose(1, 0)
        label_train = np.array(label_train)

        for i in range(16, 24):
            eeg = eeg_data[f'de_LDS{i + 1}'].astype('float')
            eye = eye_data[f'eye_{i + 1}'].astype('float')
            eeg_test.append(eeg)
            eye_test.append(eye)
            for j in range(eeg.shape[1]):
                label_test.append(self.session_label[session][i])
        eeg_test = np.concatenate(eeg_test, axis=1).transpose(1, 0, 2)
        eye_test = np.concatenate(eye_test, axis=1).transpose(1, 0)
        label_test = np.array(label_test)

        _, p, b = eeg_train.shape
        eeg_train = np.reshape(eeg_train, [eeg_train.shape[0], p * b])
        eeg_test = np.reshape(eeg_test, [eeg_test.shape[0], p * b])
        eeg_scaler = preprocessing.StandardScaler().fit(eeg_train)
        eeg_train = eeg_scaler.transform(eeg_train)
        eeg_test = eeg_scaler.transform(eeg_test)

        eye_scaler = preprocessing.StandardScaler().fit(eye_train)
        eye_train = eye_scaler.transform(eye_train)
        eye_test = eye_scaler.transform(eye_test)

        if is_training:
            self.eeg = torch.from_numpy(eeg_train)
            self.eye = torch.from_numpy(eye_train)
            self.label = torch.from_numpy(label_train)
            self.len = self.eye.shape[0]
        else:
            self.eeg = torch.from_numpy(eeg_test)
            self.eye = torch.from_numpy(eye_test)
            self.label = torch.from_numpy(label_test)
            self.len = self.eye.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.eeg[item], self.eye[item], self.label[item]

class SEEDIV_LabelFeature():
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
        while True:
            random_num = random.choice(np.where(self.train_label == 3)[0])
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
        while True:
            random_num = random.choice(np.where(self.train_label == 3)[0])
            if random_num + self.time_window < self.train_label.shape[0] and self.train_label[random_num] == self.train_label[
                random_num + self.time_window]:
                eeg_feature_label.append(self.train_eeg[random_num:(random_num + self.time_window)])
                eye_feature_label.append(self.train_eye[random_num:(random_num + self.time_window)])
                break
        self.eeg_feature_label = torch.stack(eeg_feature_label, dim=0).float()
        self.eye_feature_label = torch.stack(eye_feature_label, dim=0).float()








