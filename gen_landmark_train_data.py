# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 19:24
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : gen_p_net_train_data.py
# @Software: PyCharm
import os,cv2
import numpy as np
from tqdm import tqdm
from utils import iou_fun, getDataFromText, BBox

data_dir = './data'


def pro_landmark_train_data(input_size):
    if input_size == 12:
        net = 'pnet'
        pass
    elif input_size == 24:
        net = 'rnet'
        pass
    elif input_size == 48:
        net = 'onet'
        pass
    #数据输出路径
    path = os.path.join(data_dir,str(input_size))
    if not os.path.exists(path):
        os.mkdir(path)
        pass

    #图片处理后输出路径
    dstdir = os.path.join(path,'train_%s_landmark_aug'%(net))
    if not os.path.exists(dstdir):
        os.mkdir(dstdir)
        pass

    ftxt = os.path.join(data_dir,'trainImageList.txt')
    f = open(os.path.join(path,'landmark_%d_aug.txt'%(input_size)),'w')
    #获取图像路径，box，landmark
    data = getDataFromText(ftxt,data_dir)

    for (img_path, box, landmarkGt) in tqdm(data):
        #存储人脸图片和关键点
        f_imgs = []
        f_landmarks=[]
        img = cv2.imread(img_path)
        img_h,img_w,_=img.shape
        gt_box = np.array([box.x1,box.y1,box.x2,box.y2])
        gt_w = gt_box[2] - gt_box[0] + 1
        gt_h = gt_box[3] - gt_box[1] + 1
        #人脸图片
        f_face = img[box.y1:box.y2+1,box.x1:box.x2+1,3]
        #resize成网络输入大小
        f_face = cv2.resize(f_face,(input_size,input_size))
        landmark = np.zeros((5,2))

        for index, item in enumerate(landmarkGt):
            # 关键点相对于左上坐标偏移量并归一化
            rv = ((item[0]-gt_box[0])/gt_w,(item[1]-gt_box[1])/gt_h)
            landmark[index] = rv
            pass

        f_imgs.append(f_face)
        f_landmarks.append(landmark.resize(10))

        landmark = np.zeros((5, 2))
        # 对图像变换
        x1, y1, x2, y2 = gt_box

        # 除去过小图像
        if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
            continue
            pass

        for i in range(10):
            # 随机裁剪图像大小
            box_size = np.random.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
            # 随机左上坐标偏移量
            delta_x = np.random.randint(-gt_w * 0.2, gt_w * 0.2)
            delta_y = np.random.randint(-gt_h * 0.2, gt_h * 0.2)
            # 计算左上坐标
            nx1 = int(max(x1 + gt_w / 2 - box_size / 2 + delta_x, 0))
            ny1 = int(max(y1 + gt_h / 2 - box_size / 2 + delta_y, 0))
            nx2 = nx1 + box_size
            ny2 = ny1 + box_size
            # 除去超过边界的
            if nx2 > img_w or ny2 > img_h:
                continue
            # 裁剪边框，图片
            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[ny1:ny2 + 1, nx1:nx2 + 1, :]
            resized_im = cv2.resize(cropped_im, (input_size, input_size))
            iou = iou_fun(crop_box, np.expand_dims(gt_box, 0))
            # 只保留pos图像
            if iou > 0.65:
                f_imgs.append(resized_im)
                # 关键点相对偏移
                for index, one in enumerate(landmarkGt):
                    rv = ((one[0] - nx1) / box_size, (one[1] - ny1) / box_size)
                    landmark[index] = rv
                f_landmarks.append(landmark.reshape(10))
                landmark = np.zeros((5, 2))
                landmark_ = f_landmarks[-1].reshape(-1, 2)
                box = BBox([nx1, ny1, nx2, ny2])
                # 镜像
                if np.random.choice([0, 1]) > 0:
                    face_flipped, landmark_flipped = flip(resized_im, landmark_)
                    face_flipped = cv2.resize(face_flipped, (input_size, input_size))
                    f_imgs.append(face_flipped)
                    f_landmarks.append(landmark_flipped.reshape(10))
                # 逆时针翻转
                if np.random.choice([0, 1]) > 0:
                    face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), 5)
                    # 关键点偏移
                    landmark_rorated = box.projectLandmark(landmark_rorated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (input_size, input_size))
                    f_imgs.append(face_rotated_by_alpha)
                    f_landmarks.append(landmark_rorated.reshape(10))

                    # 左右翻转
                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                    face_flipped = cv2.resize(face_flipped, (input_size, input_size))
                    f_imgs.append(face_flipped)
                    f_landmarks.append(landmark_flipped.reshape(10))
                # 顺时针翻转
                if np.random.choice([0, 1]) > 0:
                    face_rotated_by_alpha, landmark_rorated = rotate(img, box, box.reprojectLandmark(landmark_), -5)
                    # 关键点偏移
                    landmark_rorated = box.projectLandmark(landmark_rorated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, (input_size, input_size))
                    f_imgs.append(face_rotated_by_alpha)
                    f_landmarks.append(landmark_rorated.reshape(10))

                    # 左右翻转
                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rorated)
                    face_flipped = cv2.resize(face_flipped, (input_size, input_size))
                    f_imgs.append(face_flipped)
                    f_landmarks.append(landmark_flipped.reshape(10))
            pass
        pass
    pass


def flip(face, landmark):
    # 镜像
    face_flipped_by_x = cv2.flip(face, 1)
    landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]
    landmark_[[3, 4]] = landmark_[[4, 3]]
    return (face_flipped_by_x, landmark_)


# In[5]:


def rotate(img, box, landmark, alpha):
    # 旋转
    center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    img_rotated_by_alpha = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0] * x + rot_mat[0][1] * y + rot_mat[0][2],
                             rot_mat[1][0] * x + rot_mat[1][1] * y + rot_mat[1][2]) for (x, y) in landmark])
    face = img_rotated_by_alpha[box.y1:box.y2 + 1, box.x1:box.x2 + 1]
    return (face, landmark_)


if __name__ == '__main__':
    #pro_pnet_train_data()
    # with open(new_anno_train_file, 'r') as f:
    #     annotations = f.readlines()
    # num = len(annotations)
    #
    # annotation = annotations[0].strip().split(' ')
    # img_path = annotation[0]
    # boxArr = list(map(float, annotation[1:]))
    # boxes = np.array(boxArr, dtype=np.float32).reshape(-1, 4)
    # draw_boxes(boxes,os.path.join(img_dir,img_path))

    pass