import os

class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "model", "best_float32.tflite")
    LABELS_PATH = os.path.join(BASE_DIR, "notebooks", "labels.txt")
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.25
    COLOR_MAP = {
        "helmet": (0, 255, 0),
        "head": (0, 0, 255),
        "person": (255, 0, 0)
    }

config = Config()
