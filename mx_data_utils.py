from mxnet import gluon
from mxnet.gluon import data
from mxnet.gluon.data import Dataset
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
class Gluondataset(Dataset):
    def __init__(self, training=True):
        self.training = training
        if self.training == True:
            self.feature_A_train_path = sorted([x for x in os.listdir('./data/trainfeaturemap/train_A/')])
            self.feature_B_train_path = sorted([x for x in os.listdir('./data/trainfeaturemap/train_B/')])
        else:
            self.feature_A_valid_path = sorted([x for x in os.listdir('./data/validfeaturemap/valid_A/')])
            self.feature_B_valid_path = sorted([x for x in os.listdir('./data/validfeaturemap/valid_B/')])

    def __getitem__(self, item):
        if self.training == True:
            matrix_A = np.load('./data/trainfeaturemap/train_A/' + self.feature_A_train_path[item])
            training_A = nd.array(matrix_A)
            matrix_B = np.load('./data/trainfeaturemap/train_B/' + self.feature_B_train_path[item])
            training_B = nd.array(matrix_B)
            return (training_A, training_B)
        else:

            matrix_A = np.load('./data/validfeaturemap/valid_A/' + self.feature_A_valid_path[item])
            valid_A = nd.array(matrix_A)
            matrix_B = np.load('./data/validfeaturemap/valid_B/' + self.feature_B_valid_path[item])
            valid_B = nd.array(matrix_B)
            return (valid_A, valid_B)

    def __len__(self):
        if self.training == True:
            return len(self.feature_A_train_path)
        else:
            return len(self.feature_A_valid_path)



