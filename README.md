# Cyberpunk Mirror View

## Overview
This project performs real-time 3D human pose detection using a webcam feed along with audio beat detection from the microphone input.  
It uses MediaPipe to capture 3D pose landmarks, and displays them with random vibrant colors in a cyberpunk style using Matplotlib’s 3D scatter plot.  
Additionally, it analyzes audio input using FFT to detect beats based on amplitude thresholds.

## Features
- Real-time 3D pose tracking from webcam  
- Cyberpunk-style random color highlights on pose landmarks  
- Movement-based scoring system  
- Audio beat detection using microphone input  
- Live webcam feed with pose overlay using OpenCV  

## Requirements
- Python 3.x  
- OpenCV (`opencv-python`)  
- MediaPipe (`mediapipe`)  
- Matplotlib  
- NumPy  
- SoundDevice  
- SciPy  

## Installation
```bash
pip install opencv-python mediapipe matplotlib numpy sounddevice scipy

##Usage
python <your_script_name>.py
