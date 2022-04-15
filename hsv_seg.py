import cv2
import numpy as np
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import argparse
from utils.general import draw_bbox_barcode

red = (255,0,0)
blue = (0,0,255)
yellow = (255, 255,0)


def hsv_seg(orig):
    """HSV + Edge Segmentation for white background removal. Returns Foreground mask."""
    
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # hsv mask
    mask = cv2.inRange(img, (0,0,100), (179,100,255)) # hsv mask with white background
    mask = cv2.bitwise_not(mask) # mask to remove background

    # edge 
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray,100,255)
    
    # Dilate edges
    dilatation_size = 25
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),(-1, -1))
    diledge = cv2.dilate(edges, element) 
    
    # Sum of HSV and dilated Edges
    summed = diledge + mask 
    dilatemask = cv2.dilate(summed, element)
    
    # eroded
    ero = cv2.erode(dilatemask, element, iterations=2)

    
    # Specify size on horizontal axis
    cols = orig.shape[1]
    horizontal_size = cols // 30
    horizontal = ero
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 20))
    # Apply morphology operations
    horizontal = cv2.morphologyEx(horizontal,cv2.MORPH_OPEN, horizontalStructure, iterations=2)

    # Specify size on vertical axis
    rows = orig.shape[0]
    vertical_size = rows // 30
    vertical = ero
    # Create structure element for extracting horizontal lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (20,vertical_size))
    # Apply morphology operations
    vertical = cv2.morphologyEx(vertical,cv2.MORPH_OPEN, verticalStructure, iterations=2)
    
    nmask = cv2.bitwise_and(horizontal, vertical)
    
    # Close image to remove holes in objects
    dilatation_size = 25
    element = cv2.getStructuringElement(cv2.MORPH_RECT, ( dilatation_size + 1, dilatation_size + 1),(-1, -1))
    
    nmask = cv2.morphologyEx(nmask, cv2.MORPH_CLOSE, element, iterations=2)
    
    return nmask

def cont(path):
    orig = cv2.imread(path)
    img = np.copy(orig)
    fgmask = hsv_seg(img)
  
    # Contours in Foreground mask
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgcont = cv2.drawContours(np.copy(orig), contours, -1, (255,0,0), 10)
    contlist=[]
    boxes = []
    for i,cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        if w>100 and h>100:
            if (w>h and w//h<5) or (h>=w and h//w < 5):
                contlist.append(cnt)
                boxes.append([x,y,w,h])
                cv2.rectangle(imgcont,(x,y),(x+w,y+h),(0,255,0),20)
                cv2.putText(imgcont, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 
                           5, (255,0,255), 10, cv2.LINE_AA)
    boxes = np.array(boxes)
    return contlist,boxes, imgcont, orig

def run(image_path, bardet, draw_all_objects=False):

    contlist , boxes, _, orig= cont(image_path)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    result = np.copy(orig)
    if not len(boxes):
        print('I did not see any items here, I need more training, help me learn :)\n\n')
    else:
        total_barcodes = {}
        for idx, box in enumerate(boxes):
            # Crop Object detected
            x1,y1,w,h = box
            x2, y2 = x1+w, y1+h
            item = orig[y1:y2,x1:x2,:].copy()
            if 0 in item.shape:
                continue
#             print(item.shape)
            # plt.imshow(item)
            # plt.show()
            item = cv2.cvtColor(item, cv2.COLOR_RGB2BGR)
            found, decoded_info, decoded_type, detections = bardet.detectAndDecode(item)
            if draw_all_objects:
                cnt = contlist[idx]
                area = cv2.contourArea(cnt)
                hull = cv2.convexHull(cnt)
                hullarea = cv2.contourArea(hull)
                if hullarea/area > 1.3:
                    cv2.putText(result, 'Multiple Objects', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,5, (255,255,255), 10, cv2.LINE_AA)

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
    
    return orig, result

if __name__=='__main__':

    my_parser = argparse.ArgumentParser(description='Using Computer Vision to scan Barcodes in Images')

    my_parser.add_argument('--images', action='store', type=str, default='../sample_barcode/all_barcode', help='Path to the image folder not file please. \n Example: sample/all_barcodes')
    args = my_parser.parse_args()

    # folder path
    folder_path = Path(args.images)
    assert Path(folder_path).exists() ,f'{folder_path} Does not exist, Input the path correctly'
    images = glob.glob(folder_path.__str__()+'/*.jpg')
    # Check OpenCV version    
    assert int(cv2.__version__.split('.')[0]) >= 4 ,f'OpenCV-Contrib-Python greater than 4 is required, Use command"pip install opencv-contrib==4.5.4"'

    # Initialize Barcode Detector
    bardet = cv2.barcode_BarcodeDetector()
    if not len(images):
        print('No jpg images folder specified')
    for image in images:
        orig, result =  run(image, bardet, True)
        plt.imshow(result)
        plt.show()