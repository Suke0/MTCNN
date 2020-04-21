# -*- coding: utf-8 -*-
# @Time    : 2020/3/16 19:24
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : gen_p_net_train_data.py
# @Software: PyCharm
import os,cv2
import numpy as np
from tqdm import tqdm
from PIL import ImageDraw,Image
from utils import iou_fun

anno_train_file = './data/wider_face_train_bbx_gt.txt'
new_anno_train_file = './data/wider_face_train.txt'
anno_val_file = './data/wider_face_val_bbx_gt.txt'
new_anno_val_file = './data/wider_face_val.txt'
img_dir = './data/wider_train/images'

pos_save_dir = './data/12/pos'
part_save_dir = './data/12/part'
neg_save_dir = './data/12/neg'

save_dir = './data/12'



def preprocess_txt(anno_file,new_anno_file):
    with open(anno_file, 'r') as f:  # 设置文件对象
        line = f.readline()
        line = line[:-1]
        if '.jpg' in line:
            print(line)
        while line:  # 直到读取完文件
            if '.jpg' in line:
                n_boxes = int(f.readline())
                for i in range(n_boxes):
                    boxline = f.readline()  # x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
                    arr = boxline.strip().split(' ')
                    box = np.array(arr[0:4],dtype=np.float32)
                    box[2] = box[0] + box[2] - 1
                    box[3] = box[1] + box[3] - 1
                    line += " "
                    line += " ".join(str(i) for i in box)
                    pass
                with open(new_anno_file, "a+") as wf:
                    wf.writelines(line + '\n')
                    pass
                pass
            line = f.readline()  # 读取一行文件，包括换行符
            line = line[:-1]  # 去掉换行符，也可以不去
            pass
        pass
    pass

def pro_pnet_bbox_train_data():
    f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
    f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
    f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')
    with open(new_anno_train_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print('总共的图片数： %d' % num)
    # 记录pos,neg,part三类生成数
    p_idx = 0
    n_idx = 0
    d_idx = 0
    # 记录读取图片数
    idx = 0

    for annotation in tqdm(annotations):
        annotation = annotation.strip().split(' ')
        img_path = annotation[0]
        boxArr = list(map(float, annotation[1:]))
        if len(boxArr) == 0:
            continue
            pass
        boxes = np.array(boxArr,dtype=np.float32).reshape(-1,4)
        img = cv2.imread(os.path.join(img_dir,img_path))
        idx += 1

        height, width, _ = img.shape
        neg_num = 0
        #先采样一定数量的neg图片
        while neg_num < 50:
            #随机选取截取图片大小
            size = np.random.randint(12, min(height,width)/2)
            #随机选取左上角坐标
            nx = np.random.randint(0,width-size)
            ny = np.random.randint(0,height-size)

            #截取box
            crop_box = np.array([nx,ny,nx+size,ny+size])
            iou = iou_fun(crop_box,boxes)

            #iou小于0.3判定为neg图像
            if np.max(iou) < 0.3:
                # 截取图片并resize成12X12
                cropped_img = img[ny:ny+size,nx:nx+size,:]
                resized_img = cv2.resize(cropped_img,(12,12),interpolation=cv2.INTER_LINEAR)
                save_file = os.path.join(neg_save_dir,"%s.jpg"%n_idx)
                f2.write(save_file+ ' 0\n')
                cv2.imwrite(save_file,resized_img)
                n_idx += 1
                neg_num += 1
                pass

            pass

        for box in boxes:
            #左上右下坐标
            x1,y1,x2,y2 = box
            w = x2-x1+1
            h = y2-y1+1

            #过滤掉图片过小和box在图片外的图片
            if max(w,h) < 20 or x1 < 0 or y1 < 0:
                continue
                pass
            for i in range(5):
                size = np.random.randint(12,min(height,width)/2)

                #随机生成关于x1，y1的偏移量，并且保证x1+delta_x > 0, y1+delta_y>0
                delta_x = np.random.randint(max(-size,-x1),w)
                delta_y = np.random.randint(max(-size,-y1),h)

                #截取后的左上角坐标
                nx1 = int(max(0,x1+delta_x))
                ny1 = int(max(0,y1+delta_y))

                #排除大于图片尺度的
                if nx1+size > width or ny1+size > height:
                    continue
                    pass
                crop_box = np.array([nx1,ny1,nx1+size,ny1+size])
                iou = iou_fun(crop_box, boxes)

                if np.max(iou) < 0.3:
                    cropped_img = img[ny1 : ny1 + size, nx1 : nx1 + size, :]
                    resized_img = cv2.resize(cropped_img, (12, 12), interpolation=cv2.INTER_LINEAR)
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    f2.write(save_file + ' 0\n')
                    cv2.imwrite(save_file, resized_img)
                    n_idx += 1
                    pass
                pass

            for i in range(20):
                #缩小随机选取size范围，更多截取pos和part图片
                size = np.random.randint(int(min(w,h)*0.8), np.ceil(1.25*max(w,h)))

                #除去尺度小的
                if w < 5:
                    continue
                    pass

                #偏移量，范围缩小了
                delta_x = np.random.randint(-w * 0.2, w * 0.2)
                delta_y = np.random.randint(-h * 0.2, h * 0.2)

                #截取图片左上角坐标计算是先计算x1+w/2表示的中心点，再加上delta_x偏移量，再减去size/2,即为新的左上角坐标
                nx1 = int(max(x1+w/2+delta_x-size/2,0))
                ny1 = int(max(y1+h/2+delta_y-size/2,0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                #排除超出的图片
                if nx2 > width or ny2 > height:
                    continue
                    pass
                crop_box = np.array([nx1,ny1,nx2,ny2])

                #人脸框相对于截取图片的偏移量并做归一化处理
                offset_x1 = (x1 - nx1)/float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_img = img[ny1:ny2,nx1:nx2,:]
                resized_img = cv2.resize(cropped_img,(12,12),interpolation=cv2.INTER_LINEAR)

                #box扩充一个维度作为iou输入
                box_ = box.reshape(1,-1)
                iou = iou_fun(crop_box,box_)
                if iou > 0.65:
                    save_file = os.path.join(pos_save_dir,"%s.jpg"%p_idx)
                    f1.write(save_file+' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1,offset_y1,offset_x2,offset_y2))
                    cv2.imwrite(save_file,resized_img)
                    p_idx += 1
                    pass
                elif iou >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_img)
                    d_idx += 1
                    pass
                pass
            pass
        pass
    print('%s 个图片已处理，pos：%s  part: %s neg:%s' % (idx, p_idx, d_idx, n_idx))
    f1.close()
    f2.close()
    f3.close()
    pass


if __name__ == '__main__':
    preprocess_txt(anno_train_file,new_anno_train_file)
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