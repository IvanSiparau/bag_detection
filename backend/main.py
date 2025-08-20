import os
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil

from cv_model.yolo_pipline import YOLOBagDetector

app = FastAPI(title="Bag Detection App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../fronted"))

DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(FRONTEND_DIR):
    raise RuntimeError(f"Frontend directory not found: {FRONTEND_DIR}")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "cv_model", "best.pt")
print(f"Looking for model at: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")

detector = YOLOBagDetector(yolo_weights_path=MODEL_PATH)


@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    file_id = str(uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp4")

    with open(input_path, "wb") as f:
        f.write(await file.read())

    result_path = detector.predict(input_path, n_frames=2, conf=0.5)

    shutil.move(result_path, output_path)

    return JSONResponse({
        "message": "Видео обработано",
        "download_url": f"/download/{file_id}"
    })


@app.get("/download/{file_id}")
async def download_video(file_id: str):
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp4")
    if not os.path.exists(output_path):
        return JSONResponse({"error": "Видео не найдено или еще не обработано"}, status_code=404)
    return FileResponse(output_path, media_type="video/mp4", filename="result.mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
