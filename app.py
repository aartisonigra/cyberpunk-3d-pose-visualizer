import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.fft import rfft
from mpl_toolkits.mplot3d import Axes3D
import random


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()


duration = 0.5
sample_rate = 44100

def get_audio_amplitude():
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, blocking=True)
    fft = np.abs(rfft(audio[:, 0]))
    return np.max(fft)

score = 0
last_position = None

def random_color():
 
    return (random.random(), random.random(), random.random())

def draw_3d_landmarks(landmarks):
    global score, last_position
    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.axis('off')
    ax.set_title("Random Color Cyberpunk 3D Pose Clone")

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    zs = [lm.z for lm in landmarks]

    current_position = np.array([xs, ys, zs])
    if last_position is not None:
        movement = np.linalg.norm(current_position - last_position)
        score += movement * 100
    last_position = current_position

    color = random_color() 

    ax.scatter(xs, ys, zs, c=[color], s=80, edgecolors='white', linewidths=1.5, alpha=0.9)
    ax.text2D(0.05, 0.95, f"Score: {int(score)}", transform=ax.transAxes, color='cyan', fontsize=14)
    plt.pause(0.001)

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            draw_3d_landmarks(results.pose_landmarks.landmark)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        amplitude = get_audio_amplitude()
        if amplitude > 1500:
            print("Beat Detected!")

        cv2.imshow('Cyberpunk Mirror View', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    plt.close()
