import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_pair(flair_slice: np.ndarray, ce_slice: np.ndarray) -> np.ndarray:
    """
    يأخذ مقطعين (Flair + T1-CE) بالحجم الأصلي (HxW)
    ويُرجعهما بعد إعادة التحجيم إلى 128x128 وتطبيع 0-1.
    """
    flair_r = cv2.resize(flair_slice, (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)
    ce_r    = cv2.resize(ce_slice,    (IMG_SIZE, IMG_SIZE), cv2.INTER_AREA)

    pair = np.stack([flair_r, ce_r], axis=-1).astype("float32")
    pair /= (pair.max() + 1e-6)          # تطبيع بسيط 0-1
    return pair
