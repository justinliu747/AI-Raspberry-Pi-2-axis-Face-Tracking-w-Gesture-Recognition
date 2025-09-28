# AI-Raspberry-Pi-2-axis-Face-Tracking-w-Gesture-Recognition
📹 Real-Time Face Tracking & Gesture Recognition Camera (Raspberry Pi 4)

A smart lecture-recording tool built on Raspberry Pi 4 that automatically tracks a professor’s movement and responds to hand gestures for controls like start/stop recording and marking key moments.

🔧 Tech stack

MediaPipe → real-time face & hand landmark detection

PyTorch → custom MLP trained on a 2k+ self-collected gesture dataset

OpenCV → frame handling, overlays, threaded video recording

Picamera2 / libcamera → dual-channel ISP for high-res display + low-res inference

⚡ Optimizations

Multithreaded pipeline for face detection, hand tracking & servo control

Efficient PiCamera2 capture + memory management

Threaded video writer for smooth, drop-free recording

📚 Key Learnings

Integrated computer vision + embedded ML + systems design

Collected & trained on a custom dataset, deployed on-device

Proved that with smart optimizations, a Raspberry Pi can run real-time AI workloads for education and automation

Video Demo: https://youtu.be/T8Q6L3_tIcw
