import cv2
import numpy as np
from ultralytics import YOLO
from ..config import config

class SafetyHelmetDetector:
    def __init__(self, model_path=config.MODEL_PATH, labels_path=config.LABELS_PATH):
        self.model = YOLO(model_path)
        self.class_names = self._load_labels(labels_path)
        self.color_map = config.COLOR_MAP

    def _load_labels(self, labels_path):
        try:
            with open(labels_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Warning: '{labels_path}' not found.")
            return []

    def predict(self, image, conf=config.CONFIDENCE_THRESHOLD):
        """
        Runs YOLO inference and returns annotated image.
        Expects image in RGB format (numpy array).
        """
        # Convert RGB to BGR for OpenCV processing
        img_to_draw = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        results = self.model.predict(source=image, conf=conf)
        result = results[0]

        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()

            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
            else:
                class_name = f"class_{class_id}"
                
            color = self.color_map.get(class_name, (255, 255, 255))

            x1, y1, x2, y2 = coords
            cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} {confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_to_draw, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(img_to_draw, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Convert back to RGB for output
        return cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
