import os
import subprocess
import uuid
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import torch
import matplotlib.pyplot as plt
import statsmodels.api as sm
import ultralytics
from ultralytics import YOLO


class CVService:
    def __init__(self):
        pass

    def extract_image(self, file):
        try:
            output = {}
            data = np.load(file)
            colorData = data["color"]

            for row in colorData:
                row = row[::-1]
            colorData = colorData[::-1]
            
            depthData = data["depth"]
            for row in depthData:
                row = row[::-1]
            depthData = depthData[::-1]
            
            im = Image.fromarray(colorData)
            guid = str(uuid.uuid4())
            os.makedirs("./tmp/" + guid, exist_ok=True)
            im.save("./tmp/" + guid + "/" + file.filename + ".jpg")

            output["file_location"] = "./tmp/" + \
                guid + "/" + file.filename + ".jpg"
            output["depth_array"] = depthData
            output["directory"] = "./tmp/" + guid
            return output
        except Exception as err:
            return str(err)

    def process_image(self, image_path, fileName):
        yolo = YOLO("best.pt")
        guid = str(uuid.uuid4())
        yolo.predict(image_path, save_txt=True, max_det=1, line_width=1,
                     save=True, project="img", name=guid, conf=0.1)
        directory = "./img/" + guid 
        return directory
    
    def calculate_width(self, pixel_width, distance):
        fov_horizontal_deg = 69  # Horizontal field of view of the camera in degrees
        horizontal_resolution = 640  # Horizontal resolution of the camera
        
        # Calculate real-world width per pixel (assuming linear relationship)
        real_width_per_pixel = distance * np.tan(np.radians(fov_horizontal_deg / 2)) / (horizontal_resolution)

        # Calculate real-world width of the object
        real_width = pixel_width * real_width_per_pixel

        return real_width

    def analyze_data(self, depthData, processed_image, filename, polygon=False):
        annotation = pd.read_csv(
            processed_image + "/labels/" + filename + ".txt", sep=" ", header=None)
        
        # Get the pixels at which the annotation starts and ends
        annotation = annotation.to_numpy()[0, 1:].reshape(-1, 2)
        if polygon:
            minW = round(min(annotation[:, 0]) * len(depthData))
            maxW = round(max(annotation[:, 0]) * len(depthData))
            minH = round(min(annotation[:, 1]) * len(depthData[0]))
            maxH = round(max(annotation[:, 1]) * len(depthData[0]))
        else:
            centerX = annotation[0, 0] * len(depthData[0])
            centerY = annotation[0, 1] * len(depthData)
            width = annotation[1, 0] * len(depthData[0])
            height = annotation[1, 1] * len(depthData)
            minW = round(centerX - (width / 2))
            maxW = round(centerX + (width / 2))
            minH = round(centerY - (height / 2))
            maxH = round(centerY + (height / 2))

        # Remove all zeros
        depthData = depthData.astype('float')
        depthData[depthData == 0] = np.nan
        
        mean_depth, width = (np.nanmean(depthData[minH:maxH, minW:maxW]), width)
        real_width = self.calculate_width(width, mean_depth)

        return real_width
