from keras_frcnn.pascal_voc_salient import get_data_salient
from keras_frcnn.pascal_voc_parser import get_data

path = '/home/saiprajwalk/Desktop/testing/VOCdevkit/'
actual_gts, act_class_count, act_class_mapping = get_data(path)
salient_gts, sal_class_count, sal_class_mapping = get_data_salient(path)

import cv2

for indx in range(len(salient_gts)):
    if(indx%500 == 0):
        print("Processing for index: {}".format(indx))
    actual_bboxes = actual_gts[indx]['bboxes']
    img = cv2.imread(actual_gts[indx]['filepath'])
    for bbox in actual_bboxes:
        if(bbox in salient_gts[indx]['bboxes']):
            img = cv2.rectangle(img, (bbox['x1'], bbox['y1']),(bbox['x2'], bbox['y2']),(0,0,255),2)
        else:
            img = cv2.rectangle(img, (bbox['x1'], bbox['y1']),(bbox['x2'], bbox['y2']),(0,255,0),2)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_name = "Images_With_BBoxes/salient_{}.jpg".format(indx)
    cv2.imwrite(img_name, img)
