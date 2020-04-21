# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 4:13
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : loss.py
# @Software: PyCharm

import tensorflow as tf

def total_loss():
    pass

def cls_loss_fn(cls_prob, label):
    '''计算类别损失
    参数：
      cls_prob：预测类别，是否有人
      label：真实值
    返回值：
      损失
    '''
    num_keep_radio = 0.7
    zeros = tf.zeros_like(label)
    # 只把pos的label设定为1,其余都为0
    label_filter_invalid = tf.where(tf.math.less(label, 0), zeros, label)
    # 类别size[2*batch]
    num_cls_prob = tf.size(cls_prob)
    cls_prob_reshpae = tf.reshape(cls_prob, [num_cls_prob, -1])
    label_int = tf.cast(label_filter_invalid, tf.int32)
    # 获取batch数
    num_row = cls_prob.shape[0]
    # 对应某一batch而言，batch*2为非人类别概率，batch*2+1为人概率类别,indices为对应 cls_prob_reshpae
    # 应该的真实值，后续用交叉熵计算损失
    row = tf.range(num_row) * 2
    indices_ = row + label_int
    # 真实标签对应的概率
    label_prob = tf.squeeze(tf.gather(cls_prob_reshpae, indices_))
    loss = -tf.math.log(label_prob + 1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)
    # 统计neg和pos的数量
    valid_inds = tf.where(label < zeros, zeros, ones)
    num_valid = tf.math.reduce_sum(valid_inds)
    # 选取70%的数据
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)
    # 只选取neg，pos的70%损失
    loss = loss * valid_inds
    loss, _ = tf.math.top_k(loss, k=keep_num)
    return tf.math.reduce_mean(loss)
    pass


def bbox_loss_fn(bbox_pred, bbox_target, label):
    '''计算box的损失'''
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label, dtype=tf.float32)
    # 保留pos和part的数据
    valid_inds = tf.where(tf.math.equal(tf.math.abs(label), 1), ones_index, zeros_index)
    # 计算平方差损失
    square_error = tf.math.square(bbox_pred - bbox_target)
    square_error = tf.math.reduce_sum(square_error, axis=1)
    # 保留的数据的个数
    num_valid = tf.math.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # 保留pos和part部分的损失
    square_error = square_error * valid_inds
    square_error, _ = tf.math.top_k(square_error, k=keep_num)
    return tf.math.reduce_mean(square_error)
    pass


def landmark_loss_fn(landmark_pred, landmark_target, label):
    '''计算关键点损失'''
    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    # 只保留landmark数据
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    # 计算平方差损失
    square_error = tf.math.square(landmark_pred - landmark_target)
    square_error = tf.math.reduce_sum(square_error, axis=1)
    # 保留数据个数
    num_valid = tf.math.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    # 保留landmark部分数据损失
    square_error = square_error * valid_inds
    square_error, _ = tf.math.top_k(square_error, k=keep_num)
    return tf.math.reduce_mean(square_error)
    pass


# In[6]:


def cal_accuracy(cls_prob, label):
    '''计算分类准确率'''
    # 预测最大概率的类别，0代表无人，1代表有人
    pred = tf.math.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)
    # 保留label>=0的数据，即pos和neg的数据
    cond = tf.where(tf.math.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    # 获取pos和neg的label值
    label_picked = tf.math.gather(label_int, picked)
    pred_picked = tf.math.gather(pred, picked)
    # 计算准确率
    accuracy = tf.math.reduce_mean(tf.cast(tf.math.equal(label_picked, pred_picked), tf.float32))
    return accuracy
    pass