# ============================================================
# VISIONPORT - DEEP LEARNING OBJECT DETECTION
# Experiment: Project Orientation and Lifecycle Mapping
# Model: YOLOv8
# Source: uploaded Skill_1.ipynb (cell 1)
# ============================================================

# Run in Google Colab. The upload and download helpers below are
# provided by google.colab.files.
!pip install -q ultralytics

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from google.colab import files

print("=" * 60)
print("       VISIONPORT OBJECT DETECTION SYSTEM")
print("=" * 60)

print("\nUpload a real photograph...")
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

print("\nLoading YOLOv8 pretrained model...")
model = YOLO("yolov8n.pt")
print("Model loaded successfully!")

image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image could not be loaded!")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("\nImage Acquisition Completed")
print("Original Image Size:", image_rgb.shape)

print("\nPerforming Object Detection...")
results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    verbose=False
)

output_image = results[0].plot()
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

print("\nOBJECT DETECTION RESULTS")
print("-" * 50)
boxes = results[0].boxes

if boxes is not None and len(boxes) > 0:
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        object_name = model.names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        print(f"{i+1}. Object: {object_name}")
        print(f"   Confidence: {confidence*100:.2f}%")
        print(f"   Bounding Box: ({x1}, {y1}, {x2}, {y2})")
else:
    print("No objects detected.")

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Uploaded Image", fontsize=14)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title("YOLOv8 Object Detection - Bounding Boxes", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()

output_path = "VisionPort_Detection_Output.jpg"
cv2.imwrite(
    output_path,
    cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
)

print("\n" + "=" * 60)
print("           VISIONPORT SYSTEM REPORT")
print("=" * 60)
print("Image Acquisition       : SUCCESS")
print("Image Preprocessing     : SUCCESS")
print("YOLOv8 Model Loading    : SUCCESS")
print("Object Detection        : SUCCESS")
print("Bounding Box Generation: SUCCESS")
print("Output Visualization    : SUCCESS")
print("Total Objects Detected :", len(boxes))
print("=" * 60)
print("VISIONPORT PIPELINE COMPLETED!")

files.download(output_path)
