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


def distance(p1, p2):
    """
    Calculates 3D Euclidean distance between two landmarks.
    """
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2) ** 0.5


def is_fist(hand_landmarks):
    """
    Detects if most fingers are curled (fist gesture).
    Compares each fingertip to its base joint to estimate curling.
    """
    # Mapping: (tip, base)
    finger_joints = [
        (THUMB_TIP, 2),    # Thumb
        (INDEX_TIP, 5),    # Index
        (MIDDLE_TIP, 9),   # Middle
        (RING_TIP, 13),    # Ring
        (PINKY_TIP, 17)    # Pinky
    ]

    curled_fingers = 0
    for tip_idx, base_idx in finger_joints:
        tip = hand_landmarks.landmark[tip_idx]
        base = hand_landmarks.landmark[base_idx]
        if distance(tip, base) < 0.05:  # threshold for "curled"
            curled_fingers += 1

    return curled_fingers >= 4


def is_open_palm(hand_landmarks):
    """
    Detects if the hand is open (fingers extended).
    Compares fingertips to wrist to estimate extension.
    """
    wrist = hand_landmarks.landmark[WRIST]
    extended_fingers = 0
    for tip_idx in [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        tip = hand_landmarks.landmark[tip_idx]
        if distance(tip, wrist) > 0.2:
            extended_fingers += 1

    return extended_fingers >= 4


def detect_gesture(hand_landmarks):
    """
    Detects the current gesture and returns a string label.
    """
    if is_fist(hand_landmarks):
        return "Fist"
    elif is_open_palm(hand_landmarks):
        return "Open Palm"
    else:
        return "Unknown"