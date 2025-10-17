import os
import cv2
import gdown
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rembg import remove

from lib import IS_PRODUCTION, MODEL_ID


class PredictionClass:
    def __init__(self):
        self.dir_identifier = None
        self.filename = None
        self.extension = None

        if not IS_PRODUCTION:
            print("      🔧 Running in development mode. Creating output directories.")
            os.makedirs("outputs/", exist_ok=True)

        model_path = "models/RandomForestClassifier.sav"

        if not os.path.exists(model_path):
            print("      ⬇️ Model file not found. Downloading from Google Drive...")
            try:
                gdown.download(id=MODEL_ID, output=model_path, quiet=False)
                print(f"      ✅ Model downloaded to {model_path}")
            except Exception as e:
                raise Exception(
                    f"      ❌ Failed to download from Google Drive: {str(e)}"
                )

        try:
            self.model = joblib.load(model_path)
            print(f"      ✅ Model loaded from {model_path}")
        except Exception as e:
            raise Exception(f"      ❌ Failed to load model: {e}")

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

        if not IS_PRODUCTION:
            os.makedirs(f"outputs/{self.dir_identifier}", exist_ok=True)

    def plot_images(
        self,
        step: str,
        original: np.ndarray,
        processed: np.ndarray,
        title1: str = "Original",
        title2: str = "Processed",
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
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        axs[1].set_title(title2)
        axs[1].axis("off")

        plt.tight_layout()

        cv2.imwrite(
            f"outputs/{self.dir_identifier}/{self.filename}-{step}.{self.extension}",
            processed,
        )
        plt.savefig(
            f"outputs/{self.dir_identifier}/plot-{self.filename}-{step}.{self.extension}"
        )
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
            step="1-remove-bg",
            original=strip,
            processed=strip_without_bg,
            title1="Original Strip",
            title2="Strip without Background",
        )
        return strip_without_bg

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

        contours, _ = cv2.findContours(
            binary_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        cropped = strip[y : y + h, x : x + w]

        self.plot_images(
            step="3-crop",
            original=strip,
            processed=cropped,
            title1="Rotated Strip",
            title2="Cropped Strip",
        )

        return cropped

    def extract_median_rgbs(self, strip: np.ndarray) -> list[np.ndarray]:
        if strip.shape[-1] == 4:
            strip = strip[:, :, :3]

        bottom_half = strip[strip.shape[0] // 2 :, :, :]
        upper_half = strip[: strip.shape[0] // 2, :, :]

        mask = (bottom_half >= (50, 50, 50)).all(axis=-1)
        # avg_white = np.mean(bottom_half[mask], axis=0)
        std = np.std(bottom_half[mask], axis=0)
        tolerated_distance = 0.3 * std

        threshold_dark = 50
        color_y1 = upper_half.shape[0] - 1
        while color_y1 >= 0:
            row = upper_half[color_y1, :, :]
            mask = (row > threshold_dark).all(axis=1)
            if mask.any():
                median_pixel = np.median(row[mask], axis=0)
                pixel_center = row[row.shape[0] // 2]
                if not (
                    np.abs(pixel_center - median_pixel) <= tolerated_distance
                ).all():
                    break
            color_y1 -= 1

        color_y0 = 0
        while color_y0 < upper_half.shape[0]:
            row = upper_half[color_y0, :, :]
            mask = (row > threshold_dark).all(axis=1)
            if mask.any():
                median_pixel = np.median(row[mask], axis=0)
                pixel_center = row[row.shape[0] // 2]
                if not (
                    np.abs(pixel_center - median_pixel) <= tolerated_distance
                ).all():
                    break
            color_y0 += 1
        color_y0 = max(0, color_y0 - 1)

        cropped = upper_half[color_y0 : (color_y1 + 20), :, :]

        self.plot_images(
            step="4-extract-rgbs",
            original=cv2.cvtColor(strip, cv2.COLOR_BGR2RGB),
            processed=cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB),
            title1="Cropped Strip",
            title2="Area for RGB Extraction",
        )

        q0 = cropped[: cropped.shape[0] // 4, :, :]
        q1 = cropped[cropped.shape[0] // 4 : cropped.shape[0] // 2, :, :]
        q2 = cropped[cropped.shape[0] // 2 : 3 * cropped.shape[0] // 4, :, :]
        q3 = cropped[3 * cropped.shape[0] // 4 :, :, :]
        quarters = [q0, q1, q2, q3]
        medians = []
        threshold_dark = 5
        for q in quarters:
            mask = (q > threshold_dark).all(axis=-1)
            if mask.any():
                valid_pixels = q[mask]
                median = np.median(valid_pixels, axis=(0))
            else:
                median = np.array([0, 0, 0])
            medians.append(median)

        return medians

    def index_to_interval(self, index: int) -> str | None:
        mapping = {-1: "2.50 - 4.00", 0: "4.00 - 5.50", 1: "5.50 - 7.00"}
        return mapping.get(index)

    def predict_ph_interval(self, median_rgbs: list[np.ndarray]) -> str | None:
        """
        Receives a DataFrame rgb_df with index ['Q1', 'Q2', 'Q3', 'Q4'] and columns ['R', 'G', 'B'].
        Loads the model model and scaler from the specified paths.
        Predicts the pH value based on the RGB values and returns it.

        Args:
            median_rgbs: list[np.ndarray]: List of RGB values for each quadrant.
        Returns:
            float: Predicted pH interval.
        """
        print(median_rgbs)
        rgb_df = pd.DataFrame(
            median_rgbs, index=["Q1", "Q2", "Q3", "Q4"], columns=["R", "G", "B"]
        )

        features_flat = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            rgb = rgb_df.loc[q].values
            features_flat.extend(rgb.tolist())

        columns = [
            "Q0_R",
            "Q0_G",
            "Q0_B",
            "Q1_R",
            "Q1_G",
            "Q1_B",
            "Q2_R",
            "Q2_G",
            "Q2_B",
            "Q3_R",
            "Q3_G",
            "Q3_B",
        ]
        features_df = pd.DataFrame([features_flat], columns=columns)

        ph_interval_predict = self.model.predict(features_df)

        if not IS_PRODUCTION:
            print(
                f"      🧪 Predicted pH interval: {self.index_to_interval(ph_interval_predict[0])}"
            )
            with open(
                f"outputs/{self.dir_identifier}/{self.filename}-6-predicted-ph.txt", "w"
            ) as f:
                f.write(
                    f"Predicted pH interval: {self.index_to_interval(ph_interval_predict[0])}\n"
                )

        return self.index_to_interval(ph_interval_predict[0])
