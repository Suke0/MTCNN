# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 2:26
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : net.py
# @Software: PyCharm
import tensorflow as tf

from layer import Conv2_PReLU, Conv2_Softmax, Dense_ReLU, Dense_Softmax


class PNet(tf.keras.Model):
    def __init__(self):
        super(PNet,self).__init__()
        self.layers_1 = Conv2_PReLU(10,3,1)
        self.layers_2= tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='SAME')
        self.layers_3 = Conv2_PReLU(16, 3, 1)
        self.layers_4 = Conv2_PReLU(32, 3, 1)

        self.cls_layers = Conv2_Softmax(2, 1, 1)
        self.bbox_layers = tf.keras.layers.Conv2D(4, 1, 1)
        self.landmark_layers = tf.keras.layers.Conv2D(10, 1, 1)
        pass

    def call(self,inputs):
        x = self.layers_1(inputs)
        x = self.layers_2(x)
        x = self.layers_3(x)
        x = self.layers_4(x)

        cls_prob = self.cls_layers(x)
        bbox_pred = self.bbox_layers(x)
        landmark_pred = self.landmark_layers(x)
        # (batchsize, 1, 1, 2)
        # (batchsize, 1, 1, 4)
        # (batchsize, 1, 1, 10)
        return cls_prob, bbox_pred, landmark_pred
        pass
    pass


class RNet(tf.keras.Model):
    def __init__(self,batch_size=256):
        super(RNet, self).__init__()
        self.batch_size = batch_size
        self.layers_1 = Conv2_PReLU(28, 3, 1)
        self.layers_2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME')
        self.layers_3 = Conv2_PReLU(48, 3, 1)
        self.layers_4 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='VALID')
        self.layers_5 = Conv2_PReLU(64, 2, 1)
        self.layers_6 = tf.keras.layers.Flatten()
        self.layers_7 = Dense_ReLU(128)

        self.cls_layers = Dense_Softmax(2)
        self.bbox_layers = tf.keras.layers.Dense(4)
        self.landmark_layers = tf.keras.layers.Dense(10)
        pass
    # def predict(self,inputs):
    #     batch_size = self.batch_size
    #     minibatch = []
    #     cur = 0
    #     # 所有数据总数
    #     n = inputs.shape[0]
    #     # 将数据整理成固定batch
    #     while cur < n:
    #         minibatch.append(inputs[cur:min(cur + batch_size, n), :, :, :])
    #         cur += batch_size
    #         pass
    #     cls_prob_list = []
    #     bbox_pred_list = []
    #     landmark_pred_list = []
    #     for idx, data in enumerate(minibatch):
    #         m = data.shape[0]
    #         real_size = self.batch_size
    #         # 最后一组数据不够一个batch的处理
    #         if m < batch_size:
    #             keep_inds = np.arange(m)
    #             gap = self.batch_size - m
    #             while gap >= len(keep_inds):
    #                 gap -= len(keep_inds)
    #                 keep_inds = np.concatenate((keep_inds, keep_inds))
    #                 pass
    #             if gap != 0:
    #                 keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
    #                 pass
    #             data = data[keep_inds]
    #             real_size = m
    #             pass
    #
    #         tmp, cls_prob, bbox_pred, landmark_pred = self.call(data)
    #         tmp = tmp.numpy()
    #         cls_prob_list.append(cls_prob[:real_size])
    #         bbox_pred_list.append(bbox_pred[:real_size])
    #         landmark_pred_list.append(landmark_pred[:real_size])
    #
    #     return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(
    #         landmark_pred_list, axis=0)
    #     pass

    def call(self,inputs):
        x = self.layers_1(inputs)
        x = self.layers_2(x)
        x = self.layers_3(x)
        x = self.layers_4(x)
        x = self.layers_5(x)
        x = self.layers_6(x)
        x = self.layers_7(x)

        cls_prob = self.cls_layers(x)
        bbox_pred = self.bbox_layers(x)
        landmark_pred = self.landmark_layers(x)
        # (batchsize, 2)
        # (batchsize, 4)
        # (batchsize, 10)
        return cls_prob, bbox_pred, landmark_pred
        pass
    pass


class ONet(tf.keras.Model):
    def __init__(self, batch_size=16):
        super(ONet, self).__init__()
        self.batch_size = batch_size
        self.layers_1 = Conv2_PReLU(32, 3, 1)
        self.layers_2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='SAME')
        self.layers_3 = Conv2_PReLU(64, 3, 1)
        self.layers_4 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='VALID')
        self.layers_5 = Conv2_PReLU(64, 3, 1)
        self.layers_6 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='SAME')
        self.layers_7 = Conv2_PReLU(128, 2, 1)
        self.layers_8 = tf.keras.layers.Flatten()
        self.layers_9 = Dense_ReLU(256)

        self.cls_layers = Dense_Softmax(2)
        self.bbox_layers = tf.keras.layers.Dense(4)
        self.landmark_layers = tf.keras.layers.Dense(10)
        pass

    # def predict(self,inputs):
    #     batch_size = self.batch_size
    #     minibatch = []
    #     cur = 0
    #     # 所有数据总数
    #     n = inputs.shape[0]
    #     # 将数据整理成固定batch
    #     while cur < n:
    #         minibatch.append(inputs[cur:min(cur + batch_size, n), :, :, :])
    #         cur += batch_size
    #         pass
    #     cls_prob_list = []
    #     bbox_pred_list = []
    #     landmark_pred_list = []
    #     for idx, data in enumerate(minibatch):
    #         m = data.shape[0]
    #         real_size = self.batch_size
    #         # 最后一组数据不够一个batch的处理
    #         if m < batch_size:
    #             keep_inds = np.arange(m)
    #             gap = self.batch_size - m
    #             while gap >= len(keep_inds):
    #                 gap -= len(keep_inds)
    #                 keep_inds = np.concatenate((keep_inds, keep_inds))
    #                 pass
    #             if gap != 0:
    #                 keep_inds = np.concatenate((keep_inds, keep_inds[:gap]))
    #                 pass
    #             data = data[keep_inds]
    #             real_size = m
    #             pass
    #
    #         cls_prob, bbox_pred, landmark_pred = self.call(data)
    #         cls_prob_list.append(cls_prob[:real_size])
    #         bbox_pred_list.append(bbox_pred[:real_size])
    #         landmark_pred_list.append(landmark_pred[:real_size])
    #
    #     return np.concatenate(cls_prob_list, axis=0), np.concatenate(bbox_pred_list, axis=0), np.concatenate(
    #         landmark_pred_list, axis=0)
    #     pass

    def call(self, inputs):
        x = self.layers_1(inputs)
        x = self.layers_2(x)
        x = self.layers_3(x)
        x = self.layers_4(x)
        x = self.layers_5(x)
        x = self.layers_6(x)
        x = self.layers_7(x)
        x = self.layers_8(x)
        x = self.layers_9(x)

        cls_prob = self.cls_layers(x)
        bbox_pred = self.bbox_layers(x)
        landmark_pred = self.landmark_layers(x)
        # (batchsize, 2)
        # (batchsize, 4)
        # (batchsize, 10)
        return cls_prob, bbox_pred, landmark_pred
        pass
    pass

import numpy as np

def transfor_npy_to_h5(model):
    vars = model.variables
    # for var in model.variables:
    #     print(var.name.split('/')[-1]+'__'+str(var.shape))
    #     pass
    weights = np.load('./weight/ONet/onet_weight.npy', allow_pickle=True)
    # for v in weights:
    #     print(v.shape)
    #     pass

    arr = []
    for v in vars:
        arr.append(v)
        pass

    for v, v1 in zip(arr, weights):
        v.assign(v1)
        pass

    model.save_weights("./weight/ONet/onet_weight.h5")
    pass
if __name__=="__main__":
    model = PNet()
    model.build((None,None,None,3))
    vars = model.variables
    print(vars[0].value)
    model.load_weights("./weight/PNet/pnet_weight.h5")
    print(vars[0].value)
    for var in vars:
        print(var.name.split('/')[-1]+'__'+str(var.shape))
        pass
    pass
