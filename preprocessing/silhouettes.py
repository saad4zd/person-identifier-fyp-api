import os
import cv2 as cv


def gen_silhouettes():
    frames = os.listdir(os.path.join(os.getcwd(), 'frames'))
    for frame in frames:
        img = cv.imread(
            os.path.join(os.getcwd(), 'frames', frame)
        )
        bkgrd = cv.imread(
            os.path.join(os.getcwd(), "bkgrd", "5.png")
        )
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = cv.GaussianBlur(img_gray, (3, 3), 0)
        bkgrd_gray = cv.cvtColor(bkgrd, cv.COLOR_BGR2GRAY)
        bkgrd_gray = cv.GaussianBlur(bkgrd_gray, (3, 3), 0)
        isolated_person = cv.absdiff(img_gray, bkgrd_gray)
        _, binary_img = cv.threshold(
            isolated_person, 50, 255, cv.THRESH_BINARY
        )
        binary_img = cv.dilate(binary_img, None, iterations=2)
        cv.imwrite(
            f"{os.path.join(os.getcwd(),'silhouettes')}/{frame}",
            binary_img,
        )
