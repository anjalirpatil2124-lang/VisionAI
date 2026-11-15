from ultralytics import YOLO

print("Loading YOLO model...")

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully!")

# Perform object detection
results = model.predict(
    source="test.jpg",
    save=True,
    conf=0.25
)

# Display detected objects
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        print(
            f"Object: {object_name} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

print("Detection completed!")