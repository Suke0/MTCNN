# -*- coding: utf-8 -*-
# @Time    : 2020/3/22 5:39
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : train.py
# @Software: PyCharm
import os
import random

import tensorflow as tf

from net import PNet, RNet, ONet
from utils import create_p_net_target_data, create_r_o_net_target_data
from loss import cls_loss_fn, bbox_loss_fn, landmark_loss_fn, cal_accuracy



# pos数据：图片左上右下坐标和label的IOU > 0.65的图片
# part数据：图片左上右下坐标和label的0.65 > IOU > 0.4的图片
# neg数据：图片左上右下坐标和lable的IOU < 0.3的图片
# landmark数据：图片带有landmark label的图片
#
# 网络做人脸分类的时候，使用pos和neg的图片来做，这两种数据分得开，中间隔着个part face + 0.1IOU的距离，容易使模型收敛；
# 网络做人脸bbox的偏移量回归的时候，使用pos和part数据，
# 回归的时候，就只使用landmark数据。

def train(model,data_files,batch_size=6,epochs=100):
    base_path = './model'
    if isinstance(model,PNet):
        net_size = 12
        model_path = os.path.join(base_path,'PNet')
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            pass
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 0.5
        pass
    elif isinstance(model,RNet):
        net_size = 24
        model_path = os.path.join(base_path, 'RNet')
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            pass
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 0.5
        pass
    elif isinstance(model,ONet):
        net_size = 48
        model_path = os.path.join(base_path, 'ONet')
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            pass
        radio_cls_loss = 1.0
        radio_bbox_loss = 0.5
        radio_landmark_loss = 1
        pass

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    dataArr = []
    num_step = 0
    for data_file in data_files:
        with open(data_file, 'r') as f:
            datalines = f.readlines()
            dataArr.append(datalines)
            pass

        num_step += len(datalines) // batch_size
        pass

    dataArr.append(datalines)

    # label = tf.reshape(label, [batch_size])
    # roi = tf.reshape(roi, [batch_size, 4])
    # landmark = tf.reshape(landmark, [batch_size, 10])

    # 自己构造循环
    for epoch in range(epochs):
        print('epoch: ', epoch)
        for data in dataArr:
            random.shuffle(data)
            pass

        for step in range(num_step):
            # label = tf.reshape(label, [batch_size])
            # roi = tf.reshape(roi, [batch_size, 4])
            # landmark = tf.reshape(landmark, [batch_size, 10])
            if isinstance(model,PNet):
                image_batch, label_batch, bbox_batch, landmark_batch = create_p_net_target_data(dataArr[0], step, batch_size, net_size)
                pass
            else:
                image_batch, label_batch, bbox_batch, landmark_batch = create_r_o_net_target_data(dataArr, step, batch_size, net_size)
                pass

            # 开一个gradient tape, 计算梯度
            with tf.GradientTape() as tape:
                cls_prob, bbox_pred, landmark_pred = model(image_batch)

                cls_loss = cls_loss_fn(cls_prob, label_batch)

                bbox_loss = bbox_loss_fn(bbox_pred, bbox_batch, label_batch)

                landmark_loss = landmark_loss_fn(landmark_pred, landmark_batch, label_batch)

                #accuracy = cal_accuracy(cls_prob, label_batch)

                l2_loss = tf.add_n(tf.compat.v1.losses.get_regularization_losses())

                total_loss_value = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + radio_landmark_loss * landmark_loss + l2_loss

                grads = tape.gradient(total_loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(total_loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * batch_size))
                pass

            pass
        model.save_weights(model_path)
    pass


if __name__ == "__main__":
    # inputs = np.random.rand(1,48,48,3)
    # a,b,c = ONet()(inputs)
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)

    model = PNet()
    data_files = ["./data/train_pnet_landmark.txt"]
    train(model,data_files,batch_size=6,epochs=100)
    pass

