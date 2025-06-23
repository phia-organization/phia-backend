import os
import cv2
import uuid
import json
import joblib
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rembg import remove

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

from lib import IS_PRODUCTION
class PredictionClass:
    def __init__(self):
      
        self.dir_identifier = None
        self.filename = None
        self.extension = None
        
        if not IS_PRODUCTION:
            print("🔧 Running in development mode. Creating output directories.")
            os.makedirs(f'outputs/{self.dir_identifier}', exist_ok=True)
        
        try:
          model_path = f'models/knn_model.pkl'
          self.knn = joblib.load(model_path)
          print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            raise Exception(f"❌ Failed to load model: {e}")
        
        try:
          scaler_path = f'models/scaler.pkl'
          self.scaler = joblib.load(scaler_path)
          print(f"✅ Scaler loaded from {scaler_path}")
        except Exception as e:
            raise Exception(f"❌ Failed to load scaler: {e}")
          
    def set_file_info(self, filename: str, extension: str, dir_identifier: str) -> None:
        """
        Sets the file information for the prediction class.

        Args:
            filename (str): The name of the file without extension.
            extension (str): The file extension.
            dir_identifier (str): Directory identifier for saving outputs.
        """
        self.filename = filename
        self.extension = extension
        self.dir_identifier = dir_identifier
        
        os.makedirs(f'outputs/{self.dir_identifier}', exist_ok=True)
    
    def plot_images(
        self,
        step: str,
        original: np.ndarray,
        processed: np.ndarray,
        title1: str = 'Original',
        title2: str = 'Processed'
    ) -> None:
        """
        Plots two images side by side for comparison.

        Args:
            step (str): Step identifier for the image.
            original (np.ndarray): The original image to be displayed.
            processed (np.ndarray): The processed image to be displayed.
            title1 (str, optional): Title for the original image. Defaults to 'Original'.
            title2 (str, optional): Title for the processed image. Defaults to 'Processed'.
        """
        if IS_PRODUCTION:
            return

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axs[0].set_title(title1)
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        axs[1].set_title(title2)
        axs[1].axis('off')

        plt.tight_layout()
        
        cv2.imwrite(f'outputs/{self.dir_identifier}/{self.filename}-{step}.{self.extension}', processed)
        plt.savefig(f'outputs/{self.dir_identifier}/plot-{self.filename}-{step}.{self.extension}')
        # plt.show()

    def remove_background(
      self,
      strip: np.ndarray,
    ) -> np.ndarray:
        """
        Removes the background from the strip image using rembg.

        Args:
            strip (np.ndarray): The input image with a background.

        Returns:
            np.ndarray: The image with the background removed.
        """
        strip_without_bg = remove(strip)
        
        self.plot_images(
            step='1-remove-bg',
            original=strip,
            processed=strip_without_bg,
            title1='Original Strip',
            title2='Strip without Background'
        )
        return strip_without_bg
      
    def rotate_vertically(self, strip: np.ndarray) -> np.ndarray:
        """
        Aligns the strip vertically by detecting its orientation.

        Args:
            strip (np.ndarray): Input image of the strip.

        Returns:
            np.ndarray: The rotated image of the strip aligned vertically.
        """
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Nenhum contorno encontrado.")

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        width, height = rect[1]

        if width > height:
            corrected_angle = angle - 90
        else:
            corrected_angle = angle

        if corrected_angle < -45:
            corrected_angle += 90

        (h, w) = strip.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, corrected_angle, 1.0)

        rotated = cv2.warpAffine(
            strip,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        _, binary_rotated = cv2.threshold(gray_rotated, 10, 255, cv2.THRESH_BINARY)
        top_half = binary_rotated[:h // 2, :]
        bottom_half = binary_rotated[h // 2:, :]

        top_mass = cv2.countNonZero(top_half)
        bottom_mass = cv2.countNonZero(bottom_half)

        if top_mass > bottom_mass:
            rotated = cv2.flip(rotated, 0)

        self.plot_images(
            step='2-rotate-vertically',
            original=strip,
            processed=rotated,
            title1='Strip without Background',
            title2='Strip Rotated Vertically'
        )

        return rotated

    def crop(    
        self,
        strip: np.ndarray,
    ) -> np.ndarray:
        """
        Crops the strip strip to remove any excess background.

        Args:
            strip (np.ndarray): The strip image of the strip.

        Returns:
            np.ndarray: The cropped image of the strip.
        """
        gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        _, binary_strip = cv2.threshold(gray_strip, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        cropped = strip[y:y+h, x:x+w]

        self.plot_images(
            step='3-crop',
            original=strip,
            processed=cropped,
            title1='Rotated Strip',
            title2='Cropped Strip'
        )

        return cropped
      
    def white_balance_gray_world(
        self,
        strip: np.ndarray,
    ) -> np.ndarray:
        """
        Applies white balance to the strip image.

        Args:
            strip (np.ndarray): The cropped image of the strip.

        Returns:
            np.ndarray: The white balanced image of the strip.
        """
        mean_rgb = strip.mean(axis=(0, 1))
        mean_gray = mean_rgb.mean()
        
        gain = mean_gray / mean_rgb
        balanced = strip.astype(np.float32) * gain
        
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        
        self.plot_images(
            step='4-white-balance',
            original=strip,
            processed=balanced,
            title1='Cropped Strip',
            title2='White Balanced Strip'
        )
        
        return balanced

    def extract_rgbs(
        self,
        strip: np.ndarray,
        y_measures_cm: dict[str, float],
    ) -> pd.DataFrame:
        """
        Extracts RGB values from the strip image at specified y-coordinates to a data frame, with Q1, Q2, Q3 and Q4 indexed by y-coordinates in cm.

        Args:
            strip (np.ndarray): The white balanced image of the strip.
            y_measures_cm (dict[str, float]): Dictionary with y-coordinates in cm.

        Returns:
            pd.DataFrame: DataFrame containing RGB values and their corresponding y-coordinates.
        """
            
        strip = cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)
        
        height, width, _ = strip.shape
        center_x = width // 2

        df_data = {}
        for key in ['Q1', 'Q2', 'Q3', 'Q4']:
            proportion = y_measures_cm[key] / y_measures_cm['total']
            y_pos = int(proportion * height)
            y_pos = min(max(y_pos, 0), height - 1)
            rgb = strip[y_pos, center_x]
            df_data[key] = rgb[:3]

        df = pd.DataFrame(df_data, index=['R', 'G', 'B']).T

        plt.imshow(strip)
        for key in ['Q1', 'Q2', 'Q3', 'Q4']:
            proportion = y_measures_cm[key] / y_measures_cm['total']
            y_pos = int(proportion * height)
            plt.plot(center_x, y_pos, 'ro')
            plt.text(center_x + 50, y_pos, key, color='red', fontsize=12, verticalalignment='center')

        plt.axis('off')
        plt.savefig(f'outputs/{self.dir_identifier}/{self.filename}-5-extract-rgbs.{self.extension}', bbox_inches='tight', pad_inches=0)
        # plt.show()

        df.to_csv(f'outputs/{self.dir_identifier}/{self.filename}-5-extract-rgbs.csv', index_label='Color')
        
        return df

    def predict_ph(self, rgb_df: pd.DataFrame) -> float:
        """
        Receives a DataFrame rgb_df with index ['Q1', 'Q2', 'Q3', 'Q4'] and columns ['R', 'G', 'B'].
        Loads the kNN model and scaler from the specified paths.
        Predicts the pH value based on the RGB values and returns it.
        
        Args:
            rgb_df (pd.DataFrame): DataFrame with RGB values indexed by Q1, Q2, Q3, and Q4.
            model_path (str): Path to the saved kNN model.
            scaler_path (str): Path to the saved scaler.
        Returns:
            float: Predicted pH value.
        """

        features_flat = []
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            rgb = rgb_df.loc[q].values
            features_flat.extend(rgb.tolist())

        features_array = np.array(features_flat).reshape(1, -1)

        columns = [
            'R1', 'G1', 'B1',
            'R2', 'G2', 'B2',
            'R3', 'G3', 'B3',
            'R4', 'G4', 'B4'
        ]
        features_df = pd.DataFrame([features_flat], columns=columns)
        normalized_features = self.scaler.transform(features_df)

        ph_predict = self.knn.predict(normalized_features)

        with open(f'outputs/{self.dir_identifier}/{self.filename}-6-predicted-ph.txt', 'w') as f:
          f.write(f"Predicted pH: {ph_predict[0]}\n")
          
        return ph_predict[0]






      
