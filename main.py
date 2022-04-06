import cv2
import argparse
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import glob
from utils.general import xywh2xyxy, scale_coords,nms,preprocess,draw_bbox_barcode

red = (255,0,0)
blue = (0,0,255)
yellow = (255, 255,0)

def run(image_path, yolo_sku_model, bardet, draw_all_objects=False):
    # Run Inference
    orig, img = preprocess(image_path)
    yolo_sku_model.setInput(img)
    prediction = yolo_sku_model.forward()
#     print(prediction.shape)
    
    # x y w h --> x1 y1 x2 y2
    pred = prediction[0] #Prediction on first image in batch
    pred[:,:4] = xywh2xyxy(pred[:,:4])
    # scale coordinates to image size
    pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], orig.shape).round()
    # Non Max Suppresion
    pred_nms = nms(pred, thresh = 0.1, conf = 0.1)
    
    boxes = pred_nms.copy()
    result = orig.copy()
    if not len(boxes):
        print('I did not see any items here, I need more training, help me learn :)\n\n')
    else:
        total_barcodes = {}
        for idx, box in enumerate(boxes):
            # Crop Object detected
            x1,y1,x2,y2 = box[:4].astype(np.uint16)
            item = orig[y1:y2,x1:x2,:].copy()
            if 0 in item.shape:
                continue
#             print(item.shape)
#             plt.imshow(item)
#             plt.show()
            item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
            found, decoded_info, decoded_type, detections = bardet.detectAndDecode(item)
            if draw_all_objects:
                result = cv2.rectangle(result, (x1,y1), (x2,y2), red, 20)
            # Checking barcodes detected by OpenCV
            if detections is None:
                # Red Bbox around object with no barcode
                result = cv2.rectangle(result, (x1,y1), (x2,y2), red, 20)
            else:
                detections = np.round_(detections).astype(np.uint16)
                for i in range(detections.shape[0]):
#                     print(f'{idx}th object {i}th decoded info is {decoded_info[i]}')
                    y = total_barcodes.get(decoded_info[i])
                    if decoded_info[i] == '':
                        # Barcode detected But Not decoded
                        # draw Bbox for barcode
                        corners = detections[i]
                        orig_corners = corners.copy()
                        orig_corners[:,0]+=x1
                        orig_corners[:,1]+=y1
                        result = draw_bbox_barcode(result, orig_corners, yellow)
                    # check if barcode is already found
                    elif total_barcodes.get(decoded_info[i]) is not None:
#                         print('Already found')
                        total_barcodes[decoded_info[i]]+=1
                        
                        # draw Bbox for barcode
                        corners = detections[i].copy()
                        corners[:,0]+=x1
                        corners[:,1]+=y1
                        result = draw_bbox_barcode(result, corners, blue)
                    else:
                        # Found a new Barcode

                        # draw Bbox for barcode
                        total_barcodes[decoded_info[i]]=1
                        corners = detections[i].copy()
                        corners[:,0]+=x1
                        corners[:,1]+=y1
                        result = draw_bbox_barcode(result, corners, blue)
        
        if len(total_barcodes):
            print('Info of Item     |  Count of Item')
            for item_code, count in total_barcodes.items():
                print(item_code,count,sep='    |  ')
            print('\n\n')
        else:
            print('I could not find any barcodes here, I need more training, help me learn :)\n\n')

    return orig,result

if __name__=='__main__':

    my_parser = argparse.ArgumentParser(description='Using Computer Vision to scan Barcodes in Images')

    my_parser.add_argument('--images', action='store', type=str, default='../sample_barcode/all_barcode', help='Path to the image folder not file please. \n Example: sample/all_barcodes')
    my_parser.add_argument('--weight', action='store', type=str, default='../trained/sku_last.onnx', help='Path to the weight file')
    args = my_parser.parse_args()

    # folder path
    folder_path = Path(args.images)
    weight = args.weight
    assert Path(folder_path).exists() ,f'{folder_path} Does not exist, Input the path correctly'
    images = glob.glob(folder_path.__str__()+'/*.jpg')
    # Check OpenCV version    
    assert int(cv2.__version__.split('.')[0]) >= 4 ,f'OpenCV-Contrib-Python greater than 4 is required, Use command"pip install opencv-contrib==4.5.4"'
    # Load Model
    
    # weight = '../trained/sku_last.onnx'
    assert Path(weight).exists(), f'{weight} Path is wrong, Recheck path to weights file'
    yolo_sku_model = cv2.dnn.readNetFromONNX(weight)

    # Initialize Barcode Detector
    bardet = cv2.barcode_BarcodeDetector()
    if not len(images):
        print('No jpg images folder specified')
    for image in images:
        orig, result =  run(image, yolo_sku_model, bardet)
        plt.imshow(result)
        plt.show()