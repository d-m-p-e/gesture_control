# main.py
"""
Gesture Control – Webcam application with hand tracking
Goal: Initial detection and visualization of hand landmarks using MediaPipe and OpenCV
"""

import cv2
import mediapipe as mp
from gesture_logic import detect_gesture


# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the image from BGR (OpenCV format) to RGB (MediaPipe format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the original frame
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            
            gesture = detect_gesture(hand_landmarks)
            cv2.putText(frame, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    # Display the result in a window
    cv2.imshow("Gesture Control", frame)

    # Exit the loop when ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()