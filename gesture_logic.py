# gesture_logic.py
"""
Gesture logic module – contains functions to analyze and classify hand gestures
based on MediaPipe hand landmarks.
"""

import math

# Define landmark indices for readability
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0


def distance(point1, point2):
    """
    Calculate Euclidean distance between two landmarks.
    Each point is a named tuple with x, y, and z attributes.
    """
    return math.sqrt(
        (point2.x - point1.x) ** 2 +
        (point2.y - point1.y) ** 2 +
        (point2.z - point1.z) ** 2
    )


def is_fist(hand_landmarks):
    """
    Detects if the hand is in a fist position (all fingertips close to wrist).
    Returns True if the hand is closed (fist), False otherwise.
    """
    wrist = hand_landmarks.landmark[WRIST]
    closed_fingers = 0

    for tip_id in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        tip = hand_landmarks.landmark[tip_id]
        if distance(wrist, tip) < 0.1:
            closed_fingers += 1

    return closed_fingers >= 4


def is_open_palm(hand_landmarks):
    """
    Detects if the hand is open (all fingertips far from wrist).
    Returns True if the hand is fully open, False otherwise.
    """
    wrist = hand_landmarks.landmark[WRIST]
    extended_fingers = 0

    for tip_id in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        tip = hand_landmarks.landmark[tip_id]
        if distance(wrist, tip) > 0.2:
            extended_fingers += 1

    return extended_fingers >= 4


def detect_gesture(hand_landmarks):
    """
    Wrapper function to detect predefined gestures.
    Returns the name of the gesture as a string.
    """
    if is_fist(hand_landmarks):
        return "Fist"
    elif is_open_palm(hand_landmarks):
        return "Open Palm"
    else:
        return "Unknown"