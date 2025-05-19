import cv2
import kornia as K
import numpy as np
import torch
import uuid
import os
from kornia.contrib import FaceDetector, FaceDetectorResult
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, UploadFile, File, Response

app = FastAPI()


device = torch.device("cpu")  # or 'cuda:0' if GPU available
dtype = torch.float32

k: int = 21
s: float = 35.0

face_detector = FaceDetector().to(device=device, dtype=dtype)


def apply_blur_face(img: torch.Tensor, img_vis: np.ndarray, x1, y1, x2, y2):
    roi = img[..., y1:y2, x1:x2]
    roi = K.filters.gaussian_blur2d(roi, (k, k), (s, s))
    img_vis[y1:y2, x1:x2] = K.tensor_to_image(roi)


def blur_faces_from_file(file_bytes: bytes) -> bytes:
    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # Load image using Kornia
        img = K.io.load_image(tmp_path, K.io.ImageLoadType.RGB8, device=device)[None, ...].to(dtype=dtype)
        img_vis = K.tensor_to_image(img.byte())

        with torch.no_grad():
            dets = face_detector(img)
        dets = [FaceDetectorResult(o) for o in dets]

        for b in dets:
            for score, tl, br in zip(b.score.tolist(), b.top_left.int().tolist(), b.bottom_right.int().tolist()):
                if score < 0.7:
                    continue
                x1, y1 = tl
                x2, y2 = br
                apply_blur_face(img, img_vis, x1, y1, x2, y2)

        # Save result image to memory
        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr)
        return buffer.tobytes()

    finally:
        os.remove(tmp_path)

@app.post("/blur-face/")
async def blur_face(file: UploadFile = File(...)):
    contents = await file.read()
    result_bytes = blur_faces_from_file(contents)

    return Response(content=result_bytes, media_type="image/jpeg")
