import os
import cv2 as cv


def normalize():
    frames = os.listdir(os.path.join(os.getcwd(), 'silhouettes'))
    for frame in frames:
        img = cv.imread(os.path.join(os.getcwd(), 'silhouettes', frame))
        top_rows = img[:25, :]
        end_rows = img[-25:, :]
        start_cols = img[:, :10]
        last_cols = img[:, -35:]
        if 255 in img:
            if (255 not in top_rows or 255 in img[:, 10:36]) and 255 not in end_rows and 255 not in start_cols and (255 not in last_cols or 255 in img[:, 11:-36]):
                cv.imwrite(
                    f'{os.path.join(os.getcwd(),"normalize")}/{frame}', img)
