# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 2:28
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : layer.py
# @Software: PyCharm

import tensorflow as tf

def Conv2_PReLU(filters,kernel_size,strides):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,kernel_size=kernel_size,strides=strides,padding='VALID',kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)),
        PReLU()
    ])
    pass

def Conv2_Softmax(filters,kernel_size,strides):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters,kernel_size=kernel_size,strides=strides,padding='VALID',kernel_regularizer=tf.keras.regularizers.l2(l=0.0005)),
        tf.keras.layers.Softmax()
    ])
    pass

def Dense_Softmax(filters):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(filters),
        tf.keras.layers.Softmax()
    ])
    pass

def Dense_ReLU(filters):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(filters),
        tf.keras.layers.ReLU()
    ])
    pass

# 定义网络层就是：设置网络权重和输出到输入的计算过程
class PReLU(tf.keras.layers.Layer):
    def __init__(self):
        super(PReLU, self).__init__()
        pass

    def build(self, input_shape):
        self.alphas = self.add_weight(name='alphas',shape=input_shape[-1],
                                      initializer=tf.constant_initializer(0.25),
                                      trainable=True)

    def call(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = self.alphas * (inputs - abs(inputs)) * 0.5
        return pos + neg
        pass
