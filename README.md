Overview

This project implements a real-time fall detection system using human pose keypoints extracted with MediaPipe Pose and a 2-layer LSTM neural network built in PyTorch.

The model learns temporal movement patterns from pose sequences and classifies them into:
	•	Normal activity
	•	Fall event

The system supports both offline training from videos and real-time webcam inference.

⸻

Methodology

1. Pose Extraction
	•	MediaPipe Pose detects 33 body landmarks
	•	Each landmark contains (x, y, z) coordinates
	•	Total features per frame: 99

2. Sequence Modeling
	•	Frames are grouped into sequences of 30 consecutive frames
	•	Each sequence is passed into an LSTM model

3. Model Architecture
	•	Input size: 99
	•	LSTM layers: 2
	•	Hidden size: 128
	•	Dropout: 0.3
	•	Output: 2 classes (Normal, Fall)
	•	Loss: Weighted CrossEntropyLoss (improves fall recall)
	•	Optimizer: Adam (lr = 0.001)
