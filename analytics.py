import streamlit as st
from datetime import datetime
from collections import Counter


def initialize_analytics():

    if "detection_history" not in st.session_state:
        st.session_state.detection_history = []


def record_detection(source, objects):

    initialize_analytics()

    object_counts = Counter(
        obj["name"] for obj in objects
    )

    confidence_values = [
        obj["confidence"] for obj in objects
    ]

    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values else 0
    )

    record = {
        "date": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "source": source,

        "total_objects": len(objects),

        "average_confidence": round(
            average_confidence, 2
        ),

        "object_counts": dict(object_counts)
    }

    st.session_state.detection_history.append(record)


def get_analytics():

    initialize_analytics()

    history = st.session_state.detection_history

    total_objects = sum(
        item["total_objects"] for item in history
    )

    total_images = sum(
        1 for item in history
        if item["source"] == "Image"
    )

    total_videos = sum(
        1 for item in history
        if item["source"] == "Video"
    )

    object_summary = Counter()

    for item in history:

        object_summary.update(
            item["object_counts"]
        )

    return {
        "total_objects": total_objects,
        "total_images": total_images,
        "total_videos": total_videos,
        "object_summary": dict(object_summary),
        "history": history
    }