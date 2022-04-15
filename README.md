# sku-barcode-detection
Barcode detection on grocery items

Steps to Run:

1. Run hsv_seg.py --images path/to/image/folder

Requirements: OpenCV-Contrib version 4.5.4

Input Image             |  Predicted Result
:-------------------------:|:-------------------------:
<img src="/assets/inp.jpg?raw=true" alt="drawing" width="400" height="300"/>  |  <img src="/assets/op.jpg?raw=true" alt="drawing" width="400" height="300"/></div>

Prediction Print

<img src="/assets/res.png?raw=true" alt="drawing" width="300" height="60"/>

[Link to Result images](https://drive.google.com/drive/folders/1RJzk1vweJnjmk7jmZ_ovHw6IQKaY296k?usp=sharing)
<hr>

### Methodology

**Object Detection**
1. HSV Segmentation + edge detection to extract foreground.
2. Morphological operations to get a better mask.
3. Find contours and fit Rectangle to get Bounding Box.
4. Crop item using B-Box
5. Send the detected boxes(groery items) for barcode detection.

**Barcode Detection**
1. OpenCV Barcode detector (from OpenCV-Contrib 4.5.4 package)
2. Run detection on item, store detections.
3. Display decoded barcodes with their count. (Draw Blue B-Box around them)
4. Draw Yellow B-Box around Undecoded barcodes.
5. Draw Red B-Box around item with no detected barcode.
