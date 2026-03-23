### 🔹 Real-Time Fall Detection using LSTM & MediaPipe

A real-time fall detection system built using human pose keypoints extracted with MediaPipe Pose and a 2-layer LSTM neural network implemented in PyTorch.  
The model learns temporal motion patterns from pose sequences and classifies them into **Normal activity** or **Fall event**.

Supports both offline training from video datasets and real-time webcam inference.

---

### ⚙️ Methodology

#### Pose Extraction
- MediaPipe Pose detects 33 body landmarks
- Each landmark contains (x, y, z) coordinates
- Total features per frame = 99

#### Sequence Modeling
- Frames grouped into sequences of 30 frames
- Each sequence passed to LSTM network
- Temporal motion used for fall detection

#### Model Architecture
- Input size: 99
- LSTM layers: 2
- Hidden size: 128
- Dropout: 0.3
- Output classes: 2 (Normal, Fall)
- Loss: Weighted CrossEntropyLoss (improves fall recall)
- Optimizer: Adam (lr = 0.001)

---

### 🧠 Features

- Pose-based fall detection (no object detection needed)
- Sequence learning using LSTM
- Class imbalance handling with weighted loss
- Real-time webcam inference
- Video dataset training support
- GPU / CPU auto detection

---

### 🛠 Tech Stack

- Python
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- scikit-learn

---

### ▶️ Usage

Train model

```bash
python main.py
