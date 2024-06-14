    This project used InSAR and convolutional neural network (CNN) to detect  active landslides over wide areas. It first 
performs InSAR analysis to produce a surface displacement velocity map of the target region and then employs an improved Faster 
RCNN based on attended ResNet-34 and Feature Pyramid Networks (FPN) to detect active landslides from the velocity map. Taking 
the Guizhou province in southwest China as a case study, we create a landslide image dataset includeing 462 images acquired by
the Sentinel-1 result and 145 images acquired by the PALSAR-2 result, and the dataset are manually labeled in VOC format using 
the LabelImg tool.


Configuration environment:
Python3.6/3.7/3.8
Pytorch1.7.1
pycocotools(Linux:pip install pycocotools; Windows:pip install pycocotools-windows)

File structure
  ├── RGB_Color_map: the RGB color mapping scheme to map InSAR-derived displacement rates
  ├── backbone: extract features from the displacement rates map
  ├── network_files: Faster R-CNN model（including Fast R-CNN and RPN module）
  ├── my_dataset.py: read the landslide image dataset
  ├── predict_all.py: predictive testing using trained weights
  ├── validation.py: obtain COCO metrics with trained weight data

Pre-training weights download address:
https://pan.baidu.com/s/1TlwJ0WtT7auUHiU5R-E06Q
password:  kw3v 

landslide image dataset download address:
https://pan.baidu.com/s/1TBbk-WWlJyGl7oEUaGwzbg 
password：2o2j 

Paper citation:
"Jiehua Cai; Lu Zhang; Jie Dong; Jinchen Guo; Yian Wang; Mingsheng Liao; Automatic identification of active landslides over wide areas from 
time-series InSAR measurements using Faster RCNN, International Journal of Applied Earth Observation and Geoinformation, 2023, 124: 103516"



