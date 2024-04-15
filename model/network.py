import torch
import torch.nn as nn
import torch.nn.functional as F


class Feature(nn.Module):
    def __init__(self, dataset='ENABL3S', sensor_num=0):
        super(Feature, self).__init__()
        # NW, sensor_num: 33, feature_length = 12
        if 'ENABL3S' == dataset:
            sensor_count = [33, 14, 15, 4, 29, 18, 19]
            final_kernel_size = [sensor_count[sensor_num], 12]
        # UCI, sensor_num: 45, feature_length = 6
        elif 'DSADS' == dataset:
            sensor_count = [45, 9, 9, 9, 9, 9]
            final_kernel_size = [sensor_count[sensor_num], 6]

        self.conv1 = nn.Conv2d(1, 4, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0])
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0])
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 256, kernel_size=final_kernel_size, stride=[1, 1], padding=[0, 0])
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = F.max_pool2d(F.relu6(self.bn1(self.conv1(x))),
                         stride=[1, 1], kernel_size=[1, 1])
        x = F.max_pool2d(F.relu6(self.bn2(self.conv2(x))),
                         stride=[1, 1], kernel_size=[1, 1])
        x = F.relu6(self.bn3(self.conv3(x)))
        return x


class Predictor(nn.Module):
    def __init__(self, prob=0.5, dataset='ENABL3S'):
        super(Predictor, self).__init__()
        if 'ENABL3S' == dataset:
            class_num = 7
        elif 'DSADS' == dataset:
            class_num = 19
        self.fc1 = nn.Linear(256, 128)
        self.bn1_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2_fc = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, class_num)
        self.bn_fc3 = nn.BatchNorm1d(class_num)
        self.prob = prob

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu6(self.bn1_fc(self.fc1(x)))
        x = F.relu6(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x
