import cv2
import os
from collections import Counter

from detection import model
from database import save_detection


def detect_video(input_path, output_path, user_id=None):

    # Open uploaded video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video!")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 25

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise ValueError("Unable to create output video!")

    object_counts = Counter()
    frame_count = 0
    confidence_records = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, conf=0.25, verbose=False)
            result = results[0]

            for box in result.boxes:
                class_id = int(box.cls[0])
                object_name = model.names[class_id]
                confidence = float(box.conf[0])
                object_counts[object_name] += 1
                confidence_records.append({
                    "name": object_name,
                    "confidence": round(confidence * 100, 2)
                })

            writer.write(result.plot())
            frame_count += 1

    finally:
        cap.release()
        writer.release()

    # Store this video session under the signed-in user's account.
    save_detection("Video", confidence_records, user_id=user_id)

    return {
        "frames_processed": frame_count,
        "object_counts": dict(object_counts),
        "output_path": output_path
    }


if __name__ == "__main__":
    print("Video detection module is ready!")
