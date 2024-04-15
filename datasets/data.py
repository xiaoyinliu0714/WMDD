import mat4py as m4p
import numpy as np
from datasets.utils import Dataset
from sklearn import preprocessing
from torch.utils.data import DataLoader


class Datasets(object):
    def __init__(self, data_name='ENABL3S', batch_size=32, is_one_hot=False, is_normalized=False, is_resize=False,
                 sensor_num=0, X_dim=4,total_subject = 9):
        self.data_name = data_name
        self.data_path = 'datasets/data/' + data_name + '/subject_'
        self.batch_size = batch_size
        self.is_one_hot = is_one_hot
        self.is_normalized = is_normalized
        self.is_resize = is_resize
        self.sensor_num = sensor_num
        self.X_dim = X_dim
        self.data = dict([(k, iter([])) for k in range(0, 10)])
        self.total_subject = total_subject-1

    def load_one_mat(self, idx=0):
        data = m4p.loadmat(self.data_path + str(idx) + '.mat')
        return [np.array(data['x_train']), np.array(data['y_train']),
                np.array(data['x_test']), np.array(data['y_test'])]

    def resize_feature(self, x):
        data = m4p.loadmat(self.data_path + 'idx.mat')
        idx_mat = np.array(data['idx_mat']) + 1
        zero_vec = np.zeros((len(x), 1))
        x_concat = np.concatenate((zero_vec, x), axis=1)
        return x_concat[:, idx_mat]

    def one_hot(self, y, n_classes=10):
        y = y.reshape(len(y))
        return np.eye(n_classes)[np.array(y, dtype=np.int32)]

    def select_sensor(self, x):
        emg_idx = np.r_[np.arange(1, 4), np.arange(8, 12), np.arange(21, 25), np.arange(29, 32)]
        imu_idx = np.r_[np.arange(4, 7), np.arange(12, 21), np.arange(26, 29)]
        angle_idx = np.r_[0, 7, 25, 32]
        sensor_idx = [emg_idx, imu_idx, angle_idx, np.r_[emg_idx, imu_idx],
                      np.r_[emg_idx, angle_idx], np.r_[imu_idx, angle_idx]]
        x_select = x[:, sensor_idx[self.sensor_num - 1]]
        return x_select

    def process_data(self, dataset):
        for i in range(2):
            if self.is_resize:
                if self.data_name == 'ENABL3S':
                    dataset[2 * i] = self.resize_feature(dataset[2 * i])
                else:
                    dataset[2 * i] = dataset[2 * i].reshape((-1, 45, 6))
            if 0 != self.sensor_num:
                if self.data_name == 'ENABL3S':
                    dataset[2 * i] = self.select_sensor(dataset[2 * i])
                else:
                    dataset[2 * i] = dataset[2 * i][:, 9 * (self.sensor_num - 1):9 * self.sensor_num]
            if self.is_normalized:
                dataset[2 * i] = preprocessing.scale(dataset[2 * i])
            for _ in range(self.X_dim - len(dataset[2 * i].shape)):
                dataset[2 * i] = np.expand_dims(dataset[2 * i], axis=1)
            if self.is_one_hot:
                dataset[2 * i + 1] = self.one_hot(dataset[2 * i + 1], n_classes=int(1 + np.max(dataset[2 * i + 1])))

        return dataset

    def get_samples(self, idx, source = True):
        if source ==True:
            batch_size = self.batch_size
        else:
            batch_size = self.batch_size
        data_iterator = self.data[idx]
        try:
            imgs, labels = next(data_iterator)

        except StopIteration:
            dataset = self.load_one_mat(idx=idx)
            dataset = self.process_data(dataset)

            load_data = DataLoader(Dataset(dataset[0],dataset[1]), batch_size=batch_size, shuffle=True,num_workers=0)
            data_iterator = iter(load_data)
            imgs, labels = next(data_iterator)
            self.data[idx] = data_iterator

        return imgs, labels

    def get_test_samples(self, idx):
        dataset = self.load_one_mat(idx=idx)
        dataset = self.process_data(dataset)
        img, label = dataset[2], dataset[3]
        return img, label