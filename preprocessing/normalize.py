import os
import cv2 as cv


def normalize():
    frames = os.listdir(os.path.join(os.getcwd(), 'silhouettes'))
    for frame in frames:
        img = cv.imread(os.path.join(os.getcwd(), 'silhouettes', frame))
        top_rows = img[:28, :]
        end_rows = img[-28:, :]
        start_cols = img[:, :28]
        last_cols = img[:, -28:]
        if 255 not in top_rows and 255 not in end_rows and 255 not in start_cols and 255 not in last_cols:
            cv.imwrite(
                f'{os.path.join(os.getcwd(),"normalize")}/{frame}', img)
