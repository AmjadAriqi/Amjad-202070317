import torch
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load a more accurate YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Use 'yolov5m', 'yolov5l', or 'yolov5x' for better accuracy

# Load the trained ripeness classification model
ripeness_model = load_model('best_tomato_model.keras')

def resize_image(image, target_width=800):
    """Resize image to target width while maintaining aspect ratio."""
    h, w = image.shape[:2]
    aspect_ratio = h / w
    target_height = int(target_width * aspect_ratio)
    return cv2.resize(image, (target_width, target_height))

def detect_and_classify_tomatoes(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Resize image to 800 pixels horizontally while maintaining aspect ratio
    img = resize_image(img, target_width=800)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform detection with YOLOv5
    results = model(img_rgb)
    detections = results.xyxy[0].numpy()

    # Process results
    for det in detections:  # detections per image
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.3:  # Lowered confidence threshold for better detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract tomato region
            tomato = img[y1:y2, x1:x2]
            if tomato.size == 0:  # Check for empty tomato region
                continue
            tomato = cv2.resize(tomato, (224, 224))
            tomato = tomato / 255.0
            tomato = np.expand_dims(tomato, axis=0)

            # Predict ripeness
            prediction = ripeness_model.predict(tomato)[0][0]
            label = "Ripe" if prediction > 0.5 else "Unripe"
            confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
            text = f'{label} ({confidence:.2f}%)'
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the result
    cv2.imshow("Tomato Detection and Ripeness", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
image_path = 'C:\\Users\\ADVANCED\\OneDrive\\Desktop\\Tomato Ripness Detection\\rr.jpg'
detect_and_classify_tomatoes(image_path)
