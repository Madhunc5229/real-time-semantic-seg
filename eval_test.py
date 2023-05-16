import numpy as np
import torch
import torch.nn.functional as F
import cv2

def mIOU(label, pred, num_classes=19):
    pred = F.softmax(pred, dim=1) 
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()
    each_image = list()
    # print("pred argmax:", pred.shape, pred.size)
    # print("label:", label.shape, label.size)
    # pred = pred.view(-1)
    # label = label.view(-1)
    # print(pred.shape)
    og_pred = pred
    og_label = label
    for i in range(label.shape[0]):
        pred = og_pred[i]
        label = og_label[i]
        for sem_class in range(num_classes):
            pred_inds = (pred == sem_class)
            target_inds = (label == sem_class)
            if target_inds.long().sum().item() == 0:
                iou_now = float('nan')
            else: 
                intersection_now = (pred_inds[target_inds]).long().sum().item()
                union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
                iou_now = float(intersection_now) / float(union_now)
                present_iou_list.append(iou_now)
            iou_list.append(iou_now)
        each_image.append(np.mean(present_iou_list))
    return np.mean(each_image)