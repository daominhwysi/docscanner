import cv2
import numpy as np

def remove_white_padding(image_np: np.ndarray) -> np.ndarray:
    """
    Loại bỏ padding trắng xung quanh nội dung chính của ảnh.
    Nhận và trả về ảnh dưới dạng NumPy array (BGR).
    
    Args:
        image_np (np.ndarray): Ảnh đầu vào dưới dạng NumPy (BGR).
        
    Returns:
        np.ndarray: Ảnh đã được crop bỏ padding trắng.
    """
    if image_np is None or image_np.size == 0:
        return image_np

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # 2. Invert the image: white → 0, text/dark → 255
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 3. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_np  # No content found

    # 4. Find bounding box around all contours
    all_contours = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_contours)

    # 5. Crop and return
    cropped = image_np[y:y+h, x:x+w]
    return cropped
