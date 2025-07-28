# face_3d_viewer.py

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Prepare matplotlib figure for 3D output
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    # Show original webcam feed
    cv2.imshow("Face Tracking", frame)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        # Get 3D coordinates
        x = [lm.x for lm in landmarks]
        y = [lm.y for lm in landmarks]
        z = [lm.z for lm in landmarks]

        # Clear previous plot
        ax.clear()
        ax.scatter(x, y, z, c='cyan', s=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-0.1, 0.1)
        ax.view_init(elev=10., azim=10)
        plt.draw()
        plt.pause(0.001)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.close()