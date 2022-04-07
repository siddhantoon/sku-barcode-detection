# sku-barcode-detection
Barcode detection on grocery items

Steps to Run:

1. Download Weights file from [here](https://drive.google.com/file/d/12c1y4Rav5_Qi1VnHc8lMH8MriKjsUvQ9/view?usp=sharing).
2. Run main.py --images path/to/image/folder --weight path/to/weight.onnx
3. To save results use argument --save True

Requirements: OpenCV-Contrib version 4.5.4

Input Image             |  Predicted Result
:-------------------------:|:-------------------------:
<img src="/assets/IMG_20220303_173611.jpg?raw=true" alt="drawing" width="400" height="300"/>  |  <img src="/assets/sample_out.jpg?raw=true" alt="drawing" width="400" height="300"/></div>

Prediction Print

<img src="/assets/res.JPG?raw=true" alt="drawing" width="300" height="60"/>

[Link to Result images](https://drive.google.com/drive/folders/1RJzk1vweJnjmk7jmZ_ovHw6IQKaY296k?usp=sharing)
<hr>

### Methodology

**Object Detection**
1. [Yolov5](https://github.com/ultralytics/yolov5) small Model trained on [SKU-110K](https://paperswithcode.com/dataset/sku110k) for 10 epochs.
2. Exported to ONNX for inference.
3. OpenCV DNN to run Onnx model.
4. Run Non Max Suppression for multiple BBoxes.(NMS not working properly right now)
5. Send the detected boxes(groery items) for barcode detection.

**Barcode Detection**
1. OpenCV Barcode detector (from OpenCV-Contrib 4.5.4 package)
2. Run detection on item, store detections.
3. Display decoded barcodes with their count. (Draw Blue B-Box around them)
4. Draw Yellow B-Box around Undecoded barcodes.
5. Draw Red B-Box around item with no detected barcode.
