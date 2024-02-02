# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:34:28 2024

@author: akif
"""
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi import AutoDetectionModel

# yolov7 model path
yolov7_model_path = 'best.pt'

# image path
image_path =  "2.jpg"

# Detection
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov7pip', # or 'yolov7hub'
    model_path=yolov7_model_path,
    confidence_threshold=0.5,
    device='cpu', # or 'cuda:0'
)

result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height = 1080,
    slice_width = 1080,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
)

result.export_visuals(export_dir="sonuclar/")


