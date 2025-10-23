import os
import numpy as np
import cv2
import argparse


def check_prediction(prediction_path):
    files = os.listdir(prediction_path)
    for file in files:
        file_path = os.path.join(prediction_path, file)
        img = cv2.imread(file_path, 0)
        print(np.unique(img))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Check prediction files")
    parser.add_argument("prediction_path", type=str, help="Path to the prediction files")
    args = parser.parse_args()

    if not os.path.exists(args.prediction_path):
        raise FileNotFoundError(f"The specified path does not exist: {args.prediction_path}")

    check_prediction(args.prediction_path)

    print("All prediction files checked successfully.")
    