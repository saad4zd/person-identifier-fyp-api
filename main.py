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


@app.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse(url="/docs")


@app.post("/upload/")
async def upload_file(files: List[UploadFile] = File(...)):
    # Making folders for Preprocessing Tasks
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'frames'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'bkgrd'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'silhouettes'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'normalize'), exist_ok=True)
    os.makedirs(os.path.join(os.getcwd(), 'gei'), exist_ok=True)

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
    shutil.rmtree(os.path.join(os.getcwd(), temp_dir))
    shutil.rmtree(os.path.join(os.getcwd(), 'frames'))
    shutil.rmtree(os.path.join(os.getcwd(), 'bkgrd'))
    shutil.rmtree(os.path.join(os.getcwd(), 'silhouettes'))
    shutil.rmtree(os.path.join(os.getcwd(), 'normalize'))
    shutil.rmtree(os.path.join(os.getcwd(), 'gei'))

    # Sending Response Back
    return JSONResponse(content={"prediction": f"Person Name: {class_} with {round(confidence*100,2)}% Confidence"}, status_code=200)
