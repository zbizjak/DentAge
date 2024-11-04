#    _____  ______ _   _ _______       _____ ______ 
#   |  __ \|  ____| \ | |__   __|/\   / ____|  ____|
#   | |  | | |__  |  \| |  | |  /  \ | |  __| |__   
#   | |  | |  __| | . ` |  | | / /\ \| | |_ |  __|  
#   | |__| | |____| |\  |  | |/ ____ \ |__| | |____ 
#   |_____/|______|_| \_|  |_/_/    \_\_____|______|
#                                                   
#                                                   

from fastai.vision.all import *
from fastai.vision.models.xresnet import *
from fastai.distributed import *
from accelerate.utils import write_basic_config
from accelerate import notebook_launcher
import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')

class DentalAgePredictor:
    def __init__(self, predict_folder, model_path):
        # Initialize the predictor with the folder containing the image and the model path
        self.predict_folder = predict_folder
        self.model_path = model_path
        self.predict_path = Path(predict_folder)
        self.predict_files = get_image_files(self.predict_path)
        
        # Ensure there is exactly one image file in the predict folder
        if len(self.predict_files) != 1:
            raise ValueError("There should be exactly one file in the predict folder")
        
        self.predict_file = self.predict_files[0]
        print(f'Predicting for file: {self.predict_file}')
        
        # Create dataloaders and load the model
        self.dls = self._create_dataloaders()
        self.learn = self._load_model()
    
    def _create_dataloaders(self):
        # Create dataloaders for the prediction
        def label_func(*args):
            return 0
    
        dtblk = DataBlock(blocks=(ImageBlock, RegressionBlock), get_items=get_image_files, get_y=label_func, item_tfms=Resize(1024))
        return dtblk.dataloaders(self.predict_folder, bs=1, num_workers=0)
    
    def _load_model(self):
        # Load the pre-trained model
        learn = vision_learner(self.dls, resnet34, y_range=(5, 97), loss_func=L1LossFlat(), metrics=mae)
        learn.load(self.model_path)
        return learn
    
    def predict(self):
        # Make a prediction on the image
        prediction = self.learn.predict(self.predict_file)[2].numpy()[0]
        return np.round(prediction)

# Example usage
if __name__ == '__main__':
    predict_folder = '/app/predict'
    model_path = '/app/best_model/best'
    real_age = 45

    # Create an instance of the predictor and make a prediction
    predictor = DentalAgePredictor(predict_folder, model_path)
    predicted_age = predictor.predict()
    print(f'Real age: {real_age}, Predicted age: {predicted_age}')
