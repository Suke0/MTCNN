# -*- coding: utf-8 -*-
# @Time    : 2020/3/24 1:34
# @Author  : Suke0
# @Email   : 652434288@qq.com
# @File    : test.py
# @Software: PyCharm
import os
import cv2
from model import MTCNN
from net import PNet, RNet, ONet
def predict(path,out_path):
    pnet_model = PNet()
    rnet_model = RNet()
    onet_model = ONet()
    pnet_model.build((None,None,None,3))
    rnet_model.build((None, 24, 24, 3))
    onet_model.build((None, 48, 48, 3))
    if os.path.exists("./weight/PNet/pnet_weight.h5"):
        pnet_model.load_weights("./weight/PNet/pnet_weight.h5")
        print(pnet_model.variables[0].value)
        for v in pnet_model.variables:
            print(v.name+"__"+str(v.shape))
            pass

        pass
    if os.path.exists("./weight/RNet/rnet_weight.h5"):
        rnet_model.load_weights("./weight/RNet/rnet_weight.h5")
        print(rnet_model.variables[0].value)
        for v in rnet_model.variables:
            print(v.name+"__"+str(v.shape))
            pass
        pass
    if os.path.exists("./weight/ONet/onet_weight.h5"):
        onet_model.load_weights("./weight/ONet/onet_weight.h5")
        print(onet_model.variables[0].value)
        for v in onet_model.variables:
            print(v.name+"__"+str(v.shape))
            pass
        pass
    model = MTCNN([pnet_model, rnet_model, onet_model])
    for item in os.listdir(path):
        img_path = os.path.join(path, item)
        image = cv2.imread(img_path)
        img = image
        #img = np.expand_dims(img,0).astype(np.float32)
        boxes_c, landmarks = model(img)
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # 画人脸框
            cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            # 判别为人脸的置信度
            cv2.putText(image, '{:.2f}'.format(score),
                        (corpbbox[0], corpbbox[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # 画关键点
        for i in range(landmarks.shape[0]):
            for j in range(len(landmarks[i]) // 2):
                cv2.circle(image, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 1, (0, 0, 255))
        cv2.imshow('im', image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.imwrite(out_path + item, image)
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    predict('./test_img','./output/')
    pass