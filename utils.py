import numpy as np
import cv2
from dataset import iou, generate_box


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def get_actual_boxes(boxes, boxs_default):
    actual_boxes = np.column_stack((
        boxs_default[:, 2] * boxes[:, 0] + boxs_default[:, 0],
        boxs_default[:, 3] * boxes[:, 1] + boxs_default[:, 1],
        boxs_default[:, 2] * np.exp(boxes[:, 2]),
        boxs_default[:, 3] * np.exp(boxes[:, 3])
    ))

    A = generate_box(actual_boxes[:, 0], actual_boxes[:, 1], actual_boxes[:, 2], actual_boxes[:, 3])
    return A

def convert_to_real_scale(boxes, w, h):
    boxes1 = np.zeros(boxes.shape)
    boxes1[:, ::2] = boxes[:, ::2] * w
    boxes1[:, 1::2] = boxes[:, 1::2] * h
    return boxes1.astype(int)

# def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
#     #input:
#     #windowname      -- the name of the window to display the images
#     #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
#     #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
#     #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
#     #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
#     #image_          -- the input image to the network
#     #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
#     _, class_num = pred_confidence.shape
#     #class_num = 4
#     class_num = class_num-1
#     #class_num = 3 now, because we do not need the last class (background)
    
#     image = np.transpose(image_, (1,2,0)).astype(np.uint8)
#     image1 = np.zeros(image.shape,np.uint8)
#     image2 = np.zeros(image.shape,np.uint8)
#     image3 = np.zeros(image.shape,np.uint8)
#     image4 = np.zeros(image.shape,np.uint8)
#     image1[:]=image[:]
#     image2[:]=image[:]
#     image3[:]=image[:]
#     image4[:]=image[:]
#     #image1: draw ground truth bounding boxes on image1
#     #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
#     #image3: draw network-predicted bounding boxes on image3
#     #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
#     #draw ground truth
#     for i in range(len(ann_confidence)):
#         for j in range(class_num):
#             if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
#                 #TODO:
#                 #image1: draw ground truth bounding boxes on image1
#                 #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
#                 #you can use cv2.rectangle as follows:
#                 #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
#                 #end_point = (x2, y2) #bottom right corner
#                 #color = colors[j] #use red green blue to represent different classes
#                 #thickness = 2
#                 #cv2.rectangle(image?, start_point, end_point, color, thickness)
    
#     #pred
#     for i in range(len(pred_confidence)):
#         for j in range(class_num):
#             if pred_confidence[i,j]>0.5:
#                 #TODO:
#                 #image3: draw network-predicted bounding boxes on image3
#                 #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
#     #combine four images into one
#     h,w,_ = image1.shape
#     image = np.zeros([h*2,w*2,3], np.uint8)
#     image[:h,:w] = image1
#     image[:h,w:] = image2
#     image[h:,:w] = image3
#     image[h:,w:] = image4
#     cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
#     cv2.waitKey(1)
#     #if you are using a server, you may not be able to display the image.
#     #in that case, please save the image using cv2.imwrite and check the saved image for visualization.

def visualize_pred_custom(windowname, pred_boxes, cat_ids, corresponding_default_boxes, ann_confidence, ann_box, boxs_default, image_, prefix="result", image_id=0):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)

    h, w, c = image.shape

    gt_boxes = get_actual_boxes(ann_box, boxs_default)

    gt_boxes1 = convert_to_real_scale(gt_boxes, w, h)
    # boxs_default = convert_to_real_scale(boxs_default, w, h)

    # print(len(pred_boxes))
    if len(pred_boxes) == 0:
        print(f"Detection fail. image-[{image_id}] no box detected")
        return

    pred_boxes = convert_to_real_scale(pred_boxes, w, h)
    corresponding_default_boxes = convert_to_real_scale(corresponding_default_boxes, w, h)
    
    class_num = 3
    thickness = 2

    # print(boxs_default[:4, :])
    # print(gt_boxes)
    # print(gt_boxes1)

    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                color = colors[j]
                image1 = cv2.rectangle(image1, (gt_boxes1[i, 4], gt_boxes1[i, 5]), (gt_boxes1[i, 6], gt_boxes1[i, 7]), color, thickness)

                ious = iou(boxs_default, gt_boxes[i, 4], gt_boxes[i, 5], gt_boxes[i, 6], gt_boxes[i, 7])
                ious_true = ious > 0.5

                selected_boxes = boxs_default[ious_true, :]
                if len(selected_boxes) == 0:
                    best_index = np.argmax(ious)
                    selected_boxes = np.vstack((selected_boxes, boxs_default[best_index, :]))

                selected_boxes = convert_to_real_scale(selected_boxes, w, h)
                
                # print(selected_boxes.shape)
                # print(selected_boxes)
                
                for k in range(len(selected_boxes)):
                    image2 = cv2.rectangle(image2, (selected_boxes[k, 4], selected_boxes[k, 5]), (selected_boxes[k, 6], selected_boxes[k, 7]), color, thickness)

    #pred
    for i in range(len(pred_boxes)):
        #TODO:
        #image3: draw network-predicted bounding boxes on image3
        #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
        color = colors[cat_ids[i]]
        image3 = cv2.rectangle(image3, (pred_boxes[i, 4], pred_boxes[i, 5]), (pred_boxes[i, 6], pred_boxes[i, 7]), color, thickness)
        image4 = cv2.rectangle(image4, (corresponding_default_boxes[i, 4], corresponding_default_boxes[i, 5]), (corresponding_default_boxes[i, 6], corresponding_default_boxes[i, 7]), color, thickness)
    
    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    # cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    # cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.

    cv2.imwrite(f"results/{windowname}/{prefix}-{image_id}.jpg", image)
    # if windowname == "val":
    #     cv2.imwrite('', image)
    # elif windowname == "test":
    #     cv2.imwrite('', image)

def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.25, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    
    #TODO: non maximum suppression

    # actual_boxes = np.column_stack((
    #     boxs_default[:, 2] * box_[:, 0] + boxs_default[:, 0],
    #     boxs_default[:, 3] * box_[:, 1] + boxs_default[:, 1],
    #     boxs_default[:, 2] * np.exp(box_[:, 2]),
    #     boxs_default[:, 3] * np.exp(box_[:, 3])
    # ))

    # A = generate_box(actual_boxes[:, 0], actual_boxes[:, 1], actual_boxes[:, 2], actual_boxes[:, 3])

    actual_boxes = get_actual_boxes(box_, boxs_default)

    # suppressed_confidence = []
    suppressed_boxes = []
    pred_cat_ids = []
    corresponding_default_boxes = []

    while confidence_.shape[0] > 0:
        num_of_boxes, num_of_classes = confidence_.shape

        max_index = np.unravel_index(np.argmax(confidence_[:, :-1], axis=None), (num_of_boxes, num_of_classes-1))
        
        # print(num_of_boxes)
        # print(num_of_classes)
        # print(max_index)
        # print(confidence_[max_index])

        if confidence_[max_index] > threshold:
            x_confidence = confidence_[max_index[0], :-1] # discard background
            x_box = actual_boxes[max_index[0], :]
            corresponding_default_box = boxs_default[max_index[0], :]
            confidence_ = np.delete(confidence_, max_index[0], 0)
            actual_boxes = np.delete(actual_boxes, max_index[0], 0)
            boxs_default = np.delete(boxs_default, max_index[0], 0)

            # suppressed_confidence.append(x_confidence)
            suppressed_boxes.append(x_box)
            pred_cat_ids.append(np.argmax(x_confidence))
            corresponding_default_boxes.append(corresponding_default_box)

            ious = iou(actual_boxes, x_box[4], x_box[5], x_box[6], x_box[7])

            ious_true = ious > overlap

            actual_boxes = np.delete(actual_boxes, ious_true, 0)
            confidence_ = np.delete(confidence_, ious_true, 0)
            boxs_default = np.delete(boxs_default, ious_true, 0)

        else:
            break

    suppressed_boxes = np.array(suppressed_boxes)
    pred_cat_ids = np.array(pred_cat_ids)
    corresponding_default_boxes = np.array(corresponding_default_boxes)
    
    return suppressed_boxes, pred_cat_ids, corresponding_default_boxes

def no_suppression(confidence_, box_, boxs_default, threshold=0.5):
    actual_boxes = get_actual_boxes(box_, boxs_default)

    boxes = []
    pred_cat_ids = []
    corresponding_default_boxes = []

    while confidence_.shape[0] > 0:
        num_of_boxes, num_of_classes = confidence_.shape

        max_index = np.unravel_index(np.argmax(confidence_[:, :-1], axis=None), (num_of_boxes, num_of_classes-1))

        if confidence_[max_index] > threshold:
            boxes.append(actual_boxes[max_index[0], :])
            pred_cat_ids.append(np.argmax(confidence_[max_index[0], :-1]))
            corresponding_default_boxes.append(boxs_default[max_index[0], :])
            confidence_ = np.delete(confidence_, max_index[0], 0)
            actual_boxes = np.delete(actual_boxes, max_index[0], 0)
            boxs_default = np.delete(boxs_default, max_index[0], 0)

        else:
            break

    boxes = np.array(boxes)
    pred_cat_ids = np.array(pred_cat_ids)
    corresponding_default_boxes = np.array(corresponding_default_boxes)
    
    return boxes, pred_cat_ids, corresponding_default_boxes

def save_predicted_boxes(pred_boxes, cat_ids, image_id, tmp="test"):
    path = "data/"+tmp+"/images/"
    out_path = "predicted_boxes/"+tmp+"/"
    if image_id <= 9:
        img_name = path+"0000"+str(image_id)+".jpg"
        filename = out_path+"0000"+str(image_id)+".txt"
    elif image_id <= 99:
        img_name = path+"000"+str(image_id)+".jpg"
        filename = out_path+"000"+str(image_id)+".txt"
    elif image_id <= 999:
        img_name = path+"00"+str(image_id)+".jpg"
        filename = out_path+"00"+str(image_id)+".txt"
    else:
        img_name = path+"0"+str(image_id)+".jpg"
        filename = out_path+"0"+str(image_id)+".txt"

    img = cv2.imread(img_name)
    img_h, img_w, img_c = img.shape

    f = open(filename, "w")

    if len(pred_boxes) == 0:
        print(f"Detection failed for image {image_id}")
        f.close()
        return

    for i in range(len(pred_boxes)):
        x_min = pred_boxes[i, 4]
        y_min = pred_boxes[i, 5]
        x_max = pred_boxes[i, 6]
        y_max = pred_boxes[i, 7]

        x_c = (x_min + x_max) / 2.0
        y_c = (y_min + y_max) / 2.0
        w = x_max - x_min
        h = y_max - y_min

        x_min = x_min * img_w
        y_min = y_min * img_h
        x_c = x_c * img_w
        y_c = y_c * img_h
        w = w * img_w
        h = h * img_h

        f.write(f"{cat_ids[i]} {x_min} {y_min} {w} {h}\n")
    
    f.close()
