from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

model = None
class_names = ["Early Blight", "Late Blight", "Healthy"]

BUCKET_NAME = "codebase-tf"  # اسم الـ GCP bucket

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """تحميل النموذج من GCP Bucket"""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def predict(request):
    global model
    if model is None:
        download_blob(BUCKET_NAME, "models/potatos.h5", "/tmp/potatos.h5")
        model = tf.keras.models.load_model("/tmp/potatos.h5")

    image = request.files["file"]
    image = np.array(Image.open(image).convert("RGB").resize((256, 256)))
    image = image / 255  # تطبيع الصورة

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)
    print("Predictions:", predictions)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(round(100 * np.max(predictions[0]), 2))  # تحويل لـ float عادي

    return {"class": predicted_class, "confidence": confidence}
