import numpy as np
from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("../save_models/2.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI server!"}

@app.get("/ping")
async def ping():
    return {"message": "Hello, I'M Marwan"}
def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
file: UploadFile=File(...)
):
    image=read_file_as_image( await file.read())
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predict_calss=CLASS_NAMES[np.argmax(predictions[0])]
    confedince=np.max([predictions[0]])
    return {
        'class':predict_calss,
        "confidence":float(confedince)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9000)