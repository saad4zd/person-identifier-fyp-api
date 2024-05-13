import os
import shutil
import cv2 as cv
from typing import List
from preprocessing.gei import gei_
from prediction.predict import predict
from fastapi import FastAPI, File, UploadFile
from preprocessing.normalize import normalize
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from preprocessing.silhouettes import gen_silhouettes
from fastapi.responses import HTMLResponse, JSONResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse(url="/docs")


@app.post("/upload/")
async def upload_file(files: List[UploadFile] = File(...)):
    # Making folders for Preprocessing Tasks
    temp_dir = os.path.join(BASE_DIR, "temp")
    frames_dir = os.path.join(BASE_DIR, "frames")
    bkgrd_dir = os.path.join(BASE_DIR, "bkgrd")
    silhouettes_dir = os.path.join(BASE_DIR, "silhouettes")
    normalize_dir = os.path.join(BASE_DIR, "normalize")
    gei_dir = os.path.join(BASE_DIR, "gei")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(bkgrd_dir, exist_ok=True)
    os.makedirs(silhouettes_dir, exist_ok=True)
    os.makedirs(normalize_dir, exist_ok=True)
    os.makedirs(gei_dir, exist_ok=True)


    for file in files:
        # Saving Video Files
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Generate frames from Videos
        capture_video = cv.VideoCapture(file_path)
        i = 1
        while True:
            isTrue, frame = capture_video.read()
            if not isTrue:
                print("Error: Could not read frame.")
                break
            if frame.shape[0] > 0 and frame.shape[1] > 0:
                if 'bkgrd' in file.filename:
                    cv.imwrite(f"bkgrd/{i}.png", frame)
                else:
                    cv.imwrite(f"frames/{i}.png", frame)
                i += 1
            else:
                print("Error: Invalid frame size.")
                break

        # Release the Video
        capture_video.release()

    # silhouettes extraction
    gen_silhouettes()

    # normalize silhouettes extraction
    normalize()

    # Gait Energy Image
    if gei_() == False:
        return JSONResponse(content={"message": f"Video frames aren't correct."}, status_code=400)

    # Prediction
    class_, confidence = predict()

    # Deleting the folders
    shutil.rmtree(temp_dir)
    shutil.rmtree(frames_dir)
    shutil.rmtree(bkgrd_dir)
    shutil.rmtree(silhouettes_dir)
    shutil.rmtree(normalize_dir)
    shutil.rmtree(gei_dir)

    # Sending Response Back
    return JSONResponse(content={"prediction": f"Person Name: {class_} with {round(confidence*100,2)}% Confidence"}, status_code=200)
