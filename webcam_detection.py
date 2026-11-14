import cv2
import av
import threading

from streamlit_webrtc import VideoProcessorBase
from detection import model


class WebcamProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.object_counts = {}
        self.total_detections = 0
        self.confidence_records = []

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        results = model.predict(
            source=image,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        counts = {}

        frame_records = []

        for box in result.boxes:

            class_id = int(box.cls[0])

            name = model.names[class_id]

            confidence = float(box.conf[0]) * 100

            counts[name] = counts.get(name, 0) + 1

            frame_records.append({
                "name": name,
                "confidence": round(confidence, 2)
            })

        with self.lock:

            self.object_counts = counts

            self.total_detections += len(frame_records)

            self.confidence_records.extend(frame_records)

        annotated_image = result.plot()

        total_objects = sum(counts.values())

        cv2.putText(
            annotated_image,
            f"Objects: {total_objects}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(
            annotated_image,
            format="bgr24"
        )

    def get_results(self):

        with self.lock:

            return list(self.confidence_records)