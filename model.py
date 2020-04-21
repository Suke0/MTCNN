# -*- coding: utf-8 -*-
# @Time    : 2020/3/15 1:55
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : model.py
# @Software: PyCharm

import cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import nms, convert_to_square, processed_img


class MTCNN(tf.keras.Model):
    def __init__(self,detectors,training = False, min_face_size = 20, threshold = [0.6, 0.7, 0.7], scale_factor = 0.79):
        super(MTCNN,self).__init__()
        self.pnet = detectors[0]
        self.rnet = detectors[1]
        self.onet = detectors[2]
        self.training = training

        self.min_face_size = min_face_size
        self.threshold = threshold
        self.scale_factor = scale_factor
        pass

    def call(self,inputs):
        if self.training:
            all_boxes = []
            landmarks = []
            batch_idx = 0
            empty_array = np.array([])
            for databatch in tqdm(inputs):
                batch_idx += 1
                img = databatch.numpy()
                if self.pnet:
                    boxes, boxes_c, _ = self.detect_pnet(img)
                    if boxes_c is None:
                        all_boxes.append(empty_array)
                        landmarks.append(empty_array)
                        continue
                        pass
                    pass

                if self.rnet:
                    boxes, boxes_c, _ = self.detect_rnet(img, boxes_c)
                    if boxes_c is None:
                        all_boxes.append(empty_array)
                        landmarks.append(empty_array)
                        continue
                        pass
                    pass

                if self.onet:
                    boxes, boxes_c, landmark = self.detect_onet(img, boxes_c)
                    if boxes_c is None:
                        all_boxes.append(empty_array)
                        landmarks.append(empty_array)
                        continue
                        pass
                    pass

                all_boxes.append(boxes_c)
                landmark = [1]
                landmarks.append(landmark)
                pass
            return all_boxes, landmarks
            pass
        else:
            '''用于测试'''
            boxes, boxes_c, landmark  = None,None,None
            # pnet
            if self.pnet:
                boxes, boxes_c, _ = self.detect_pnet(inputs)
                if boxes_c is None:
                    return np.array([]), np.array([])

            # rnet
            if self.rnet:
                boxes, boxes_c, _ = self.detect_rnet(inputs, boxes_c)
                if boxes_c is None:
                    return np.array([]), np.array([])

            # onet
            if self.onet:
                boxes, boxes_c, landmark = self.detect_onet(inputs, boxes_c)
                if boxes_c is None:
                    return np.array([]), np.array([])

            return boxes_c, landmark
            pass

        pass

    def detect_pnet(self,img):
        '''通过pnet筛选box和landmark
                参数：
                  im:输入图像[h,w,3]
        '''
        net_size = 12
        #人脸和输入图片的比率
        current_scale = float(net_size) / self.min_face_size
        img_resized = processed_img(img,current_scale)
        current_height, current_width, _ = img_resized.shape
        all_boxes = []
        #图像金字塔
        while min(current_height, current_width) > net_size:
            img_resized = tf.expand_dims(img_resized,axis=0)
            img_resized = tf.cast(img_resized,tf.float32)
            cls_prob, bbox_pred, _ = self.pnet(img_resized) #(1,h,w,2),(1,h,w,4)
            cls_prob = cls_prob[0].numpy()
            bbox_pred = bbox_pred[0].numpy()
            bboxes = self.generate_bbox(cls_prob[:,:,1],bbox_pred,current_scale,self.threshold[0])
            current_scale *= self.scale_factor  # 继续缩小图像做金字塔
            img_resized = processed_img(img, current_scale)
            current_height, current_width, _ = img_resized.shape

            if bboxes.size == 0:
                continue
                pass
            # 非极大值抑制留下重复低的box
            keep = nms(bboxes[:, :5], 0.5)
            bboxes = bboxes[keep]
            all_boxes.append(bboxes)
            pass
        if len(all_boxes) == 0:
            return None, None, None
            pass

        all_boxes = np.vstack(all_boxes)
        # 将金字塔之后的box也进行非极大值抑制
        keep = nms(all_boxes[:, :5], 0.7)
        all_boxes = all_boxes[keep]
        boxes =np.copy(all_boxes[:, :5])
        # box的长宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        x1Arr = all_boxes[:, 0] + all_boxes[:, 5] * bbw
        y1Arr = all_boxes[:, 1] + all_boxes[:, 6] * bbh
        x2Arr = all_boxes[:, 2] + all_boxes[:, 7] * bbw
        y2Arr = all_boxes[:, 3] + all_boxes[:, 8] * bbh
        scoreArr = all_boxes[:, 4]
        # 对应原图的box坐标和分数
        # boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
        #                      all_boxes[:, 1] + all_boxes[:, 6] * bbh,
        #                      all_boxes[:, 2] + all_boxes[:, 7] * bbw,
        #                      all_boxes[:, 3] + all_boxes[:, 8] * bbh,
        #                      all_boxes[:, 4]])
        boxes_c = np.concatenate([x1Arr.reshape(-1,1),y1Arr.reshape(-1,1),x2Arr.reshape(-1,1),y2Arr.reshape(-1,1),scoreArr.reshape(-1,1)],axis=-1)

        return boxes, boxes_c, None
        pass

    def detect_rnet(self, img,dets):
        '''通过rent选择box
        参数：
          im：输入图像
          dets:pnet选择的box，是相对原图的绝对坐标
        返回值：
          box绝对坐标
        '''
        h, w, _ = img.shape
        #将pnet的box变成包含他的正方形，可以避免信息损失
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, dw, dh] = self.pad(dets,w,h)
        delete_size = np.ones_like(dw) * 20
        ones = np.ones_like(dw)
        zeros = np.zeros_like(dw)
        num_boxes = np.sum(np.where((np.minimum(dw, dh) >= delete_size), ones, zeros))
        cropped_imgs = np.zeros((num_boxes, 24, 24, 3), dtype=np.float32)
        for i in range(num_boxes):
            # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
            # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
            if dh[i] < 20 or dw[i] < 20:
                continue
            tmp = np.zeros((dh[i],dw[i],3),dtype=np.uint8)
            tmp[dy[i]:edy[i]+1,dx[i]:edx[i]+1,:] = img[y[i]:ey[i]+1,x[i]:ex[i]+1,:]
            cropped_imgs[i,:,:,:] = (cv2.resize(tmp,(24,24)) - 127.5) / 128
            pass

        cls_scores, reg, _ = self.rnet(cropped_imgs)
        cls_scores, reg = cls_scores.numpy(), reg.numpy()
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.threshold[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:,4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            pass
        else:
            return None,None,None
            pass

        keep = nms(boxes,0.6)
        boxes = boxes[keep]
        # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
        boxes_c = self.calibrate_box(boxes, reg[keep])
        return boxes, boxes_c,None
        pass


    def calibrate_box(self, bbox, reg):
        '''校准box
        参数：
          bbox:pnet生成的box

          reg:rnet生成的box偏移值
        返回值：
          调整后的box是针对原图的绝对坐标
        '''

        bbox_c = bbox.copy()
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
        return bbox_c

    def pad(self,bboxes, w, h):
        '''将超出图像的box进行处理
        参数：
          bboxes:人脸框
          w,h:图像长宽
        返回值：
          dy, dx : 为调整后的box的左上角坐标相对于原box左上角的坐标
          edy, edx : 为调整后的box右下角相对原box左上角的相对坐标
          y, x : 调整后的box在原图上左上角的坐标
          ey, ex : 调整后的box在原图上右下角的坐标
        '''
        tw, th = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        n_box = bboxes.shape[0]

        dx, dy = np.zeros((n_box,)), np.zeros((n_box,))
        edx, edy = tw.copy() - 1, th.copy() - 1
        # box左上右下的坐标
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        # 找到超出右下边界的box并将ex,ey归为图像的w,h
        # edx,edy为调整后的box右下角相对原box左上角的相对坐标

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tw[tmp_index]  - 1 - (ex[tmp_index] - w + 1)
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = th[tmp_index] - 1 - (ey[tmp_index] - h + 1)
        ey[tmp_index] = h - 1

        # 找到超出左上角的box并将x,y归为0
        # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tw, th]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

        #return x.astype(np.int32), y.astype(np.int32), ex.astype(np.int32), ey.astype(np.int32), dx.astype(np.int32), dy.astype(np.int32), edx.astype(np.int32), edy.astype(np.int32)
        pass

    def detect_onet(self,img,dets):
        '''将onet的选框继续筛选基本和rnet差不多但多返回了landmark'''
        h, w, _ = img.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, dw, dh] = self.pad(dets,w,h)
        n_boxes = dets.shape[0]
        cropped_imgs = np.zeros((n_boxes,48,48,3),dtype=np.float32)
        for i in range(n_boxes):
            tmp = np.zeros((dh[i],dw[i],3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = img[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            cropped_imgs[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128
            pass
        #cropped_imgs = tf.cast(cropped_imgs,tf.float32)
        cls_scores, reg, landmark = self.onet(cropped_imgs)
        cls_scores = cls_scores[:,1]
        keep_inds = np.where(cls_scores > self.threshold[2])[0]
        if len(keep_inds) >0:
            boxes = dets[keep_inds]
            boxes[:,4] = cls_scores.numpy()[keep_inds]
            reg = reg.numpy()[keep_inds]
            landmark = landmark.numpy()[keep_inds]
            pass
        else:
            return None,None,None
            pass
        h = boxes[:, 3] - boxes[:, 1] + 1
        w = boxes[:, 2] - boxes[:, 0] + 1

        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = self.calibrate_box(boxes, reg)

        boxes = boxes[nms(boxes, 0.6)]
        keep = nms(boxes_c, 0.6)
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes, boxes_c, landmark
        pass

    def generate_bbox(self,cls_pro, bbox_pred, scale, threshold):
        """
        得到对应原图的box坐标，分类分数，box偏移量
        cls_pro.shape=[h,w,1],bbox_pred.shape=[h,w,4]
        """
        #pnet大致将图像size缩小2倍
        stride = 2
        cellsize = 12

        # 将置信度高的留下
        t_index = np.where(cls_pro > threshold)

        #没有人脸
        if len(t_index[0]) == 0:
            return np.array([])
            pass

        #偏移量
        bbox_pred = bbox_pred[t_index[0], t_index[1], :]
        bbox_pred = np.reshape(bbox_pred,(-1,4))
        score = cls_pro[t_index[0], t_index[1]]
        score = np.reshape(score,(-1,1))

        x1Arr = np.round((stride * t_index[1]) / scale)
        x1Arr = np.reshape(x1Arr,(-1,1))
        y1Arr = np.round((stride * t_index[0]) / scale)
        y1Arr = np.reshape(y1Arr,(-1,1))
        x2Arr = np.round((stride * t_index[1] + cellsize) / scale)
        x2Arr = np.reshape(x2Arr,(-1,1))
        y2Arr = np.round((stride * t_index[0] + cellsize) / scale)
        y2Arr = np.reshape(y2Arr,(-1,1))

        bboxes = np.concatenate([x1Arr,y1Arr,x2Arr,y2Arr,score,bbox_pred],-1)
        return bboxes
        pass

    pass





if __name__ == "__main__":
    # inputs = np.random.rand(1,48,48,3)
    # a,b,c = ONet()(inputs)
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)
    model = MTCNN()
    inputs = np.random.rand(1, 244, 244, 3) * 255
    all_boxes, landmarks = model(inputs)
    pass