import onnxruntime
import numpy as np
import cv2
import os

# --- Config ---
IMAGE_SIZE = 512
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'ghostnet_classifier.onnx')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
THRESHOLD_POSITIVE = 0.65


# --- Image Classifier Class ---
class ImageClassifier:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        self.session = onnxruntime.InferenceSession(model_path)
        print(f"Model loaded: {model_path}")

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def preprocess(self, np_img):
        img = cv2.resize(np_img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        return np.expand_dims(img, axis=0)  # (1, 3, H, W)

    def predict(self, np_img: np.ndarray):
        try:
            input_tensor = self.preprocess(np_img)
            ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
            ort_outputs = self.session.run(None, ort_inputs)
            logits = ort_outputs[0]
            probs = self.softmax(logits)
            confidence = probs[0][1]
            predicted_class_index = 1 if confidence > THRESHOLD_POSITIVE else 0
            return predicted_class_index, logits, confidence
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh numpy: {e}")
            return None, None, None

classifier = ImageClassifier(ONNX_MODEL_PATH)

# # --- Test ---
# if __name__ == "__main__":
#     image_path = "your_image.png"  # ← Thay bằng ảnh của bạn
#     if not os.path.exists(image_path):
#         print(f"Ảnh không tồn tại: {image_path}")
#         exit(1)

#     np_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
#     classifier = ImageClassifier(ONNX_MODEL_PATH)

#     class_to_idx = {0: "negative", 1: "positive"}
#     pred_idx, logits, confidence = classifier.predict(np_image)

#     if pred_idx is not None:
#         print(f"Prediction: {class_to_idx[pred_idx]} (Confidence: {confidence:.2%})")
#     else:
#         print("Không thể dự đoán.")
