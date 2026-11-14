import cv2
from ultralytics import YOLO
import os

# Load YOLO model
print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully!")


def detect_image(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read image!")

    # Perform object detection
    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    # Store detected objects
    detected_objects = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            object_name = model.names[class_id]

            detected_objects.append({
                "name": object_name,
                "confidence": round(confidence * 100, 2)
            })

    # Draw bounding boxes
    annotated_image = results[0].plot()

    # Create output folder
    os.makedirs("outputs", exist_ok=True)

    # Save result
    output_path = "outputs/detected_image.jpg"

    cv2.imwrite(output_path, annotated_image)

    return annotated_image, detected_objects


# Display detection results
if __name__ == "__main__":

    image, objects = detect_image("test.jpg")

    print("\nDETECTION RESULTS")

    print("Total objects detected:", len(objects))

    for obj in objects:

        print(
            obj["name"],
            "-",
            obj["confidence"],
            "%"
        )

    print("\nResult saved in outputs/detected_image.jpg")

    cv2.imshow("Object Detection", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()