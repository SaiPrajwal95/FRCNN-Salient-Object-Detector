import json
import os.path
from keras_frcnn.pascal_voc_parser import get_data
from Salient_Regions_Detection.saliency import BMS_thresh
from Salient_Regions_Detection.findSalientRegions import find_sal_regions
import numpy as np

def findIOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    if(np.isnan(iou)):
        iou = 0
    # return the intersection over union value
    return iou

def salient_ground_truth_extraction(path, relative_saliency):
    all_imgs, classes_count, class_mapping = get_data(path)
    for ind in range(len(all_imgs)):
        if (ind%500 == 0):
            print("completed process for {} images".format(ind))
        impath = all_imgs[ind]['filepath']
        # img = cv2.imread(impath)
        sal_mask = BMS_thresh(impath,0.5)
        sal_regions = find_sal_regions(sal_mask,0.005)
        im_actual_gts = all_imgs[ind]['bboxes']
        gt_saliencies = []
        gt_areas = []
        keepIndex=[]
        for gt in im_actual_gts:
            box_gt = (gt['x1'], gt['y1'], gt['x2'], gt['y2'])
            max_sal_prob = 0
            gt_areas.append((gt['x2'] - gt['x1'])*(gt['y2'] - gt['y1']))
            for sal_reg in sal_regions:
                box_sal = sal_reg
                sal_prob = findIOU(box_gt,box_sal)
                #print(sal_prob,end="")
                if(sal_prob > max_sal_prob):
                    max_sal_prob = sal_prob
                gt_saliencies.append(max_sal_prob)
                #print(gt_saliencies)
        if(sum(gt_saliencies)==0):
            keepIndex.append(np.argmax(gt_areas))
        elif(len(gt_saliencies)==1):
            gt_saliencies[0]=1
            keepIndex.append(0)
        else:
            gt_saliencies = np.array(gt_saliencies)
            gt_saliencies = (gt_saliencies-min(gt_saliencies))/(max(gt_saliencies)-min(gt_saliencies))
            gt_saliencies = list(gt_saliencies)
            itemindex = np.where((np.array(gt_saliencies)>=relative_saliency)==1)
            keepIndex = list(itemindex[0])
        for box_ind in range(len(im_actual_gts)):
            if(box_ind not in keepIndex):
                if(box_ind >= len(all_imgs[ind]['bboxes'])):
                    continue
                # More stuff here
                classes_count[all_imgs[ind]['bboxes'][box_ind]['class']] -= 1
                del(all_imgs[ind]['bboxes'][box_ind])
                keepIndex = list(np.array(keepIndex)-1)
    return all_imgs, classes_count, class_mapping


def get_data_salient(path, relative_saliency=0.65):
    if(os.path.isdir('Salient_Ground_Truths')):
        print('Using the available salient GTs')
        with open('Salient_Ground_Truths/salient_ground_truths.json', 'r') as fp:
            all_imgs = json.load(fp)
        with open('Salient_Ground_Truths/classes_count.json', 'r') as fp:
            classes_count = json.load(fp)
        with open('Salient_Ground_Truths/class_mapping.json', 'r') as fp:
            class_mapping = json.load(fp)
        return all_imgs, classes_count, class_mapping
    else:
        print('Generating the salient GTs')
        return salient_ground_truth_extraction(path)
