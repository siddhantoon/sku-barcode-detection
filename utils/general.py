import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 4
thickness = 5
def draw_bbox_barcode(image, corners, color):
    """Creates Bbox in image with Corner Coordinates"""
    image = cv2.line(image, corners[0], corners[1], color, 20)
    image = cv2.line(image, corners[1], corners[2], color, 20)
    image = cv2.line(image, corners[2], corners[3], color, 20)
    image = cv2.line(image, corners[3], corners[0], color, 20)
    return image

def preprocess(path):
    """Read Image and Preprocess for YOLO"""
    # Read image
    img0 = cv2.imread(path)  # BGR
    assert img0 is not None, f'Image Not Found {path}'
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    
    # resize
    img = cv2.resize(img0,(640,640),interpolation = cv2.INTER_CUBIC)
    # Convert
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img/=255
    img = img[None] #batch dim
    return img0, img

def draw_range_idx(img0,pred):
    imgresult = img0.copy()
    for i in range(len(pred)):
        x1,y1,x2,y2 = pred[i,:4].astype(np.int32)
        obj = pred[i,4] #object confidence
        obj = int(100*obj)
#         print(obj)
        imgresult = cv2.rectangle(imgresult, (x1,y1), (x2,y2), (255,0,0), 20)
        cv2.putText(imgresult, str(obj), (x1, y1), font, font_scale, (0, 0, 0), thickness)
    return imgresult

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Scale detected co-ordinates to original image size"""
    for i in img1_shape:
        i = float(i)
    for i in img0_shape:
        i = float(i)
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # np.array (faster grouped)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right"""
    
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def nms(dets, thresh=0.25, conf=0.1):
    """Non Max Suppresion with IOU threshold and Object Confidence"""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    x = scores > conf
    dets = dets[x]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    dets = dets[keep]
    return dets
