import tensorflow as tf
import time, pathlib, os

DEFAULT_MODEL = pathlib.Path(__file__).parent.parent / "models" / "3D_MRI_Brain_tumor_segmentation.h5"
MODEL_PATH    = pathlib.Path(os.getenv("MODEL_PATH", DEFAULT_MODEL))

print(f"[*] Loading model from {MODEL_PATH} ...")
start = time.perf_counter()
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print(f"[*] Model loaded in {time.perf_counter()-start:.1f}s")
