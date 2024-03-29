import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import cv2
from math import sqrt
from random import random


def generate_box(x, y, w, h):
    # box = np.array([x, y, w, h, x-w/2.0, y-h/2.0, x+w/2.0, y+h/2.0], dtype=float)
    # box = np.transpose(np.vstack((x, y, w, h, x-w/2.0, y-h/2.0, x+w/2.0, y+h/2.0)))
    box = np.column_stack((x, y, w, h, x-w/2.0, y-h/2.0, x+w/2.0, y+h/2.0))
    box = np.clip(box, a_min=0, a_max=1)
    return box

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    
    boxes = np.zeros((135*4, 8))
    box_index = 0

    for k in range(len(layers)):

        for i in range(layers[k]):
            for j in range(layers[k]):

                # row by row because of how torch.view work?? not sure??
                x_center = (0.5 + j) / layers[k]
                y_center = (0.5 + i) / layers[k]

                boxes[box_index, :] = generate_box(x_center, y_center, small_scale[k], small_scale[k])
                boxes[box_index+1, :] = generate_box(x_center, y_center, large_scale[k], large_scale[k])
                boxes[box_index+2, :] = generate_box(x_center, y_center, large_scale[k]*sqrt(2), large_scale[k]/sqrt(2))
                boxes[box_index+3, :] = generate_box(x_center, y_center, large_scale[k]/sqrt(2), large_scale[k]*sqrt(2))

                box_index = box_index + 4

    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    # ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    if cat_id == 0:
        one_hot_label = np.array([1, 0, 0, 0])
    elif cat_id == 1:
        one_hot_label = np.array([0, 1, 0, 0])
    elif cat_id == 2:
        one_hot_label = np.array([0, 0, 1, 0])

    gt_w = x_max - x_min
    gt_h = y_max - y_min
    gt_box = [x_min + gt_w / 2.0, 
              y_min + gt_h / 2.0, 
              gt_w, 
              gt_h]

    ious_true = (ious > threshold) | np.argmax(ious)

    ann_box[ious_true, :] = np.column_stack((
        (gt_box[0] - boxs_default[ious_true, 0]) / boxs_default[ious_true, 2], 
        (gt_box[1] - boxs_default[ious_true, 1]) / boxs_default[ious_true, 3],
        np.log(gt_box[2] / boxs_default[ious_true, 2]), 
        np.log(gt_box[3] / boxs_default[ious_true, 3])
    ))
    
    ann_confidence[ious_true, :] = one_hot_label
    
    # ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    
    # This part is done above



class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        
        #to use function "match":
        #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
        #where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        
        #note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        #For example, point (x=100, y=200) in a image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)
        
        img = cv2.imread(img_name)
        img_h, img_w, img_c = img.shape

        crop_threshold = 0.1
        if self.train:
            # a -> top left, b -> bottom right
            ax = int(random() * img_w * crop_threshold)
            ay = int(random() * img_h * crop_threshold)
            bx = int(img_w - random() * img_w * crop_threshold)
            by = int(img_h - random() * img_h * crop_threshold)

            img = img[ay:by, ax:bx, :]
            img_h = by - ay
            img_w = bx - ax

        img = cv2.resize(img, (320, 320))
        img = np.transpose(img, (2, 0, 1)) 

        annotations_txt = open(ann_name)
        annotations = annotations_txt.readlines()
        annotations_txt.close()

        for i in range(len(annotations)):
            line = annotations[i].split()
            cat_id = int(line[0])
            
            x_min = float(line[1])
            y_min = float(line[2])
            w = float(line[3])
            h = float(line[4])
            x_max = x_min + w
            y_max = y_min + h

            if self.train:
                x_min = x_min - ax
                y_min = y_min - ay
                x_max = x_max - ax
                y_max = y_max - ay
                if x_min < 0:
                    x_min = 0
                if y_min < 0:
                    y_min = 0
                if x_max > img_w:
                    x_max = img_w
                if y_max > img_h:
                    y_max = img_h
            
            x_min = x_min / img_w
            y_min = y_min / img_h
            x_max = x_max / img_w
            y_max = y_max / img_h

            match(ann_box, ann_confidence, self.boxs_default, self.threshold, cat_id, x_min, y_min, x_max, y_max)
            
        if self.train:
            return img, ann_box, ann_confidence
        else:
            return img, ann_box, ann_confidence, int(self.img_names[index][:-4])

if __name__ == "__main__":
    a = np.array([0.5, 0.5, 0.5, 0.5])
    print(a)
    print(a.shape)

    A1 = generate_box(0.5, 0.5, 0.5, 0.5)
    print(A1)
    print(A1.shape)
    print()

    A2 = generate_box(a, a, a, a)
    print(A2)
    print(A2.shape)
    print()

    a = np.ones((1, 4)) * 0.5
    print(a)
    print(a.shape)

    A2 = generate_box(a, a, a, a)
    print(A2)
    print(A2.shape)
    print()

    a = np.ones((4, 1)) * 0.5
    print(a)
    print(a.shape)

    A2 = generate_box(a, a, a, a)
    print(A2)
    print(A2.shape)