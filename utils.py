# -*- coding: utf-8 -*-
# @Time    : 2020/3/17 21:22
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : utils.py
# @Software: PyCharm
import os

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw


def iou_fun(box,boxes):
    '''裁剪的box和图片所有人脸box的iou值
       参数：
         box：裁剪的box,当box维度为4时表示box左上右下坐标，维度为5时，最后一维为box的置信度
         boxes：图片所有人脸box,[n,4]
       返回值：
         iou值，[n,]
       '''
    # box面积
    box_area = (box[2]-box[0]+1) * (box[3]-box[1]+1)
    #boxes面积，[n,]
    area = (boxes[:,2]-boxes[:,0]+1) * (boxes[:,3]-boxes[:,1]+1)
    # 重叠部分左上右下坐标
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # 重叠部分长宽
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    # 重叠部分面积
    inter = w * h
    return inter / (box_area + area - inter + 1e-10)
    pass


def nms(dets, thresh):
    '''剔除太相似的box'''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)

        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def convert_to_square(box):
    '''将box转换成更大的正方形
        参数：
          box：预测的box,[n,5]
        返回值：
          调整后的正方形box，[n,5]
    '''
    square_box = box.copy()
    h = box[:, 3] - box[:, 1] + 1
    w = box[:, 2] - box[:, 0] + 1
    #寻找正方形的最大边长
    max_side = np.maximum(w,h)
    square_box[:, 0] = box[:, 0] + w * 0.5 - max_side * 0.5
    square_box[:, 1] = box[:, 1] + h * 0.5 - max_side * 0.5
    square_box[:, 2] = square_box[:, 0] + max_side - 1
    square_box[:, 3] = square_box[:, 1] + max_side - 1
    return square_box
    pass


def read_annotation(base_dir, label_path):
    '''读取文件的image，box'''
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # 图像地址
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/images/' + imagepath
        images.append(imagepath)
        # 人脸数目
        nums = labelfile.readline().strip('\n')

        one_image_bboxes = []
        for i in range(int(nums)):
            bb_info = labelfile.readline().strip('\n').split(' ')
            # 人脸框
            face_box = [float(bb_info[i]) for i in range(4)]

            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]

            one_image_bboxes.append([xmin, ymin, xmax, ymax])

        bboxes.append(one_image_bboxes)

    data['images'] = images
    data['bboxes'] = bboxes
    return data

def getDataFromText(txt,data_path,with_landmark = True):
    '''获取txt中的图像路径，人脸box，人脸关键点
    参数：
      txt：数据txt文件
      data_path:数据存储目录
      with_landmark:是否留有关键点
    返回值：
      result包含(图像路径，人脸box，关键点)
    '''
    with open(txt,'r') as f:
        lines = f.readlines()
        pass
    result = []
    for line in lines:
        components = line.strip().split(' ')
        img_path = os.path.join(data_path,components[0]).replace('\\','/')
        box =np.array([components[1],components[3],components[2],components[4]],dtype=np.int32)
        if not with_landmark:
            result.append((img_path,BBox(box)))
            continue
            pass

        #五个关键点（x,y)
        landmark = np.zeros((5,2))
        for i in range(5):
            rv = (float(components[5+2*i]),float(components[5+2*i+1]))
            landmark[i] = rv
            pass
        result.append((img_path,BBox(box),landmark))
        pass
    pass


class BBox:
    # 人脸的box
    def __init__(self, box):
        self.x1 = box[0]
        self.y1 = box[1]
        self.x2 = box[2]
        self.y2 = box[3]

        self.x = box[0]
        self.y = box[1]
        self.w = box[2] - box[0] + 1
        self.h = box[3] - box[1] + 1

    def project(self, point):
        '''将关键点的绝对值转换为相对于左上角坐标偏移并归一化
        参数：
          point：某一关键点坐标(x,y)
        返回值：
          处理后偏移
        '''
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        '''将关键点的相对值转换为绝对值，与project相反
        参数：
          point:某一关键点的相对归一化坐标
        返回值：
          处理后的绝对坐标
        '''
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        '''对所有关键点进行reproject操作'''
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        '''对所有关键点进行project操作'''
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p



# 将级别结果显示在图片上
def draw_boxes(boxes, img_file):
    img = Image.open(img_file)
    draw = ImageDraw.Draw(img)

    for box in boxes: #box = [x1,y1,x2,y2]
        box = [box[0],box[1],box[2],box[3]]
        draw.rectangle(box, outline='red')
    img.save(f"output_img02.jpg")
    img.show()
    pass

def processed_img(img, scale):
    '''预处理数据，转化图像尺度并对像素归一到[-1,1]
    '''
    h,w,_ = img.shape
    n_h = int(h*scale)
    n_w = int(w*scale)
    dsize = (n_w,n_h)
    img_resized = cv2.resize(np.array(img), dsize,interpolation=cv2.INTER_LINEAR)
    img_resized = (img_resized - 127.5)/128
    return img_resized
    pass


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)
    return inputs
    pass


def create_p_net_target_data(datalines, step, batch_size, net_size):
    image_batch, label_batch, bbox_batch, landmark_batch = [], [], [], []
    for i in range(batch_size):
        dataArr = datalines[step * batch_size + i].strip().split(' ')

        bbox = np.zeros((4,), dtype=np.float32)
        landmark = np.zeros((10,), dtype=np.float32)

        img_file = dataArr[0]
        img = cv2.imread(img_file)
        img = tf.reshape(img, [net_size, net_size, 3])
        # 将值规划在[-1,1]内
        img = (tf.cast(img, tf.float32) - 127.5) / 128
        # 图像色相变换
        img = image_color_distort(img)
        image_batch.append(img)

        label_batch.append(np.array(dataArr[1], dtype=np.int32))

        if len(dataArr) > 2:
            bbox = np.array(dataArr[2:6], dtype=np.float32)
            pass
        bbox_batch.append(bbox)

        if len(dataArr) > 6:
            landmark = np.array(dataArr[7:], dtype=np.float32)
            pass
        landmark_batch.append(landmark)
        pass

    image_batch = np.concatenate(image_batch)
    label_batch = np.concatenate(label_batch)
    bbox_batch = np.concatenate(bbox_batch)
    landmark_batch = np.concatenate(landmark_batch)

    image_batch = tf.reshape(image_batch, [batch_size, net_size, net_size, 3])
    label_batch = tf.reshape(label_batch, [batch_size])
    bbox_batch = tf.reshape(bbox_batch, [batch_size, 4])
    landmark_batch = tf.reshape(landmark_batch, [batch_size, 10])
    return image_batch,label_batch,bbox_batch,landmark_batch
    pass

def create_r_o_net_target_data(dataArr, step, batch_size, net_size):
    # 各数据占比
    # 目的是使每一个batch的数据占比都相同
    pos_radio, part_radio, landmark_radio, neg_radio = 1.0 / 6, 1.0 / 6, 1.0 / 6, 3.0 / 6
    pos_batch_size = int(np.ceil(batch_size * pos_radio))
    part_batch_size = int(np.ceil(batch_size * part_radio))
    neg_batch_size = int(np.ceil(batch_size * neg_radio))
    landmark_batch_size = int(np.ceil(batch_size * landmark_radio))
    batch_sizes = [pos_batch_size, part_batch_size, neg_batch_size, landmark_batch_size]

    image_batch, label_batch, bbox_batch, landmark_batch = [], [], [], []
    for index, batch in enumerate(batch_sizes):
        datalines = dataArr[index]
        for i in range(batch):
            dataArr = datalines[step * batch_size + i].strip().split(' ')
            bbox = np.zeros((4,), dtype=np.float32)
            landmark = np.zeros((10,), dtype=np.float32)

            img_file = dataArr[0]
            img = cv2.imread(img_file)
            img = tf.reshape(img, [net_size, net_size, 3])
            # 将值规划在[-1,1]内
            img = (tf.cast(img, tf.float32) - 127.5) / 128
            # 图像色相变换
            img = image_color_distort(img)
            image_batch.append(img)

            label_batch.append(np.array(dataArr[1], dtype=np.int32))

            if len(dataArr) > 2:
                bbox = np.array(dataArr[2:6], dtype=np.float32)
                pass
            bbox_batch.append(bbox)

            if len(dataArr) > 6:
                landmark = np.array(dataArr[7:], dtype=np.float32)
                pass
            landmark_batch.append(landmark)
            pass
        pass

    image_batch = np.concatenate(image_batch)
    label_batch = np.concatenate(label_batch)
    bbox_batch = np.concatenate(bbox_batch)
    landmark_batch = np.concatenate(landmark_batch)

    image_batch = tf.reshape(image_batch, [batch_size, net_size, net_size, 3])
    label_batch = tf.reshape(label_batch, [batch_size])
    bbox_batch = tf.reshape(bbox_batch, [batch_size, 4])
    landmark_batch = tf.reshape(landmark_batch, [batch_size, 10])
    return image_batch, label_batch, bbox_batch, landmark_batch
    pass


if __name__ == '__main__':
    draw_boxes(np.array([[84,92,161,169]]),'./data/lfw_5590/Aaron_Eckhart_0001.jpg')

    pass