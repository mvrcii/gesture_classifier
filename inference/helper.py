import math

import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

from config import INFERENCE_MIN_WRIST_VISIBILITY_THRESHOLD, INFERENCE_MIN_SHOULDER_DISTANCE_TO_ARM_LENGTH_RATIO


def find_frame_index_ms_ago(timestamps, millis):
    """ Returns the index of the first timestamp (searching backwards from the newest, which is longer than
    WINDOW_SIZE ago, returns -1 if no timestamp with a large enough delta can be found"""
    i = len(timestamps) - 1
    while i >= 0:
        # search backwards from the current frame, up to a frame that was 2 seconds ago
        i = i - 1
        dif = timestamps[-1] - timestamps[i]
        if dif > millis:
            return i
    return -1


def get_label_and_probability_for_sample(network, gesture_mapping, sample):
    result = network.predict(np.array([sample]), apply_feature_scaling=True)
    detected_label = gesture_mapping[np.argmax(result[0])]
    probability = np.amax(result[0])
    return detected_label, probability


def invalid_landmarks(pose_landmarks):
    if not pose_landmarks:
        return True

    # TODO: Optimize access
    left_shoulder = pose_landmarks.landmark[PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[PoseLandmark.RIGHT_SHOULDER]
    left_elbow = pose_landmarks.landmark[PoseLandmark.LEFT_ELBOW]
    right_elbow = pose_landmarks.landmark[PoseLandmark.RIGHT_ELBOW]
    left_wrist = pose_landmarks.landmark[PoseLandmark.LEFT_WRIST]
    right_wrist = pose_landmarks.landmark[PoseLandmark.RIGHT_WRIST]

    if left_wrist.visibility < INFERENCE_MIN_WRIST_VISIBILITY_THRESHOLD \
            or right_wrist.visibility < INFERENCE_MIN_WRIST_VISIBILITY_THRESHOLD:
        # print("Blocking: Poor wrist visibility")
        return True

    if right_shoulder.x > left_shoulder.x:
        # print("Blocking: Subject facing backwards")
        return True

    # Determining elbow with the best visibility and chose shoulder accordingly
    shoulder, elbow = (left_shoulder, left_elbow) \
        if left_elbow.visibility > right_elbow.visibility \
        else (right_shoulder, right_elbow)

    # Calculations for
    length_upper_arm = math.sqrt(math.pow(shoulder.x - elbow.x, 2) +
                                 math.pow(shoulder.y - elbow.y, 2))
    dist_shoulders = math.sqrt(math.pow(left_shoulder.x - right_shoulder.x, 2) +
                               math.pow(left_shoulder.y - right_shoulder.y, 2))

    if dist_shoulders < length_upper_arm * INFERENCE_MIN_SHOULDER_DISTANCE_TO_ARM_LENGTH_RATIO:
        # print("Blocking: Subject facing sidewards")
        return True
    return False
