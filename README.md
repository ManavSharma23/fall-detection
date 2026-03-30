# Fall Detection System

A real-time fall detection system built using human pose keypoints extracted with MediaPipe Pose and a 2-layer LSTM neural network implemented in PyTorch. The model learns temporal motion patterns from pose sequences and classifies them into **Normal activity** or **Fall event**.

## 🚀 Features
- **Real-time Detection**: Live feed analysis from webcam with visual overlays.
- **Pose-based Analysis**: Leverages MediaPipe's 33 skeletal landmarks (no complex object detection needed).
- **Sequence Modeling**: Uses an LSTM architecture to capture temporal patterns of a fall.
- **Training Pipeline**: End-to-end script for data processing, training, and evaluation.
- **Class Balance**: Handles data imbalance with weighted CrossEntropyLoss.

## 🛠️ Tech Stack
- **Python**: Core programming language.
- **PyTorch**: Deep learning framework for the LSTM model.
- **MediaPipe**: Real-time pose estimation.
- **OpenCV**: Video processing and visualization.
- **NumPy & Scikit-learn**: Data processing and metrics.

## 📋 Methodology

### Pose Extraction
- MediaPipe Pose detects 33 body landmarks.
- Each landmark contains (x, y, z) coordinates.
- Total features per frame = 99.

### Sequence Modeling
- Frames are grouped into sequences of **30 frames**.
- Each sequence is passed to the LSTM network.
- Temporal motion patterns are used for fall classification.

### Model Architecture
- **Input size**: 99 features.
- **LSTM layers**: 2 layers with 128 hidden units each.
- **Dropout**: 0.3 for regularization.
- **Output classes**: 2 (Normal, Fall).
- **Optimizer**: Adam (learning rate = 0.001).

## 💻 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ManavSharma23/fall-detection.git
   cd fall-detection
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Project Structure
- `Main.py`: Primary script for training and inference.
- `fall_model.pth`: Pre-trained model weights.
- `dataset/`: Training data (subfolders: `normal`, `fall`).
- `requirements.txt`: Python package dependencies.

## 🕹️ Usage

Run the main script:
```bash
python Main.py
```
Options:
1. **Train Model**: Processes videos in `dataset/` and trains a new model.
2. **Run Real-time Demo**: Starts the webcam for live detection.

---

## 🔮 Future Work & Improvements

To make this system more robust and production-ready, the following enhancements are planned:

### 1. 🤖 Model Enhancements
- **Spatial-Temporal Graph Convolutional Networks (ST-GCN)**: Move from simple LSTM to GCNs to better capture the spatial relationships between skeletal joints.
- **Explainable AI (XAI)**: Integrate activation maps to highlight which specific joints (e.g., hip or shoulder) contributed most to the fall detection decision.
- **Broader Dataset**: Include diverse environments (low light, crowded rooms) and varied fall types (fainting, tripping, slipping).

### 2. ⚡ Performance & Deployment
- **Edge Compatibility**: Convert the model to **ONNX** or **TensorRT** for high-speed inference on devices like Raspberry Pi or NVIDIA Jetson.
- **Mobile Integration**: Develop a mobile app using Flutter or React Native that utilizes the camera for local fall monitoring.

### 3. 🛡️ Privacy & Ethics
- **Automated Anonymization**: Implement real-time blurring of faces or background elements to ensure user privacy, especially for residential use cases.
- **Local-First Processing**: Ensure all visual data stays on the device and only metadata (e.g., "Fall Detected") is sent to the cloud.

### 4. 🚨 Alert & Notification System
- **Cloud Integration**: Push alerts to a cloud backend (Firebase/AWS) when a fall is detected.
- **Emergency Notifications**: Integrate with **Twilio API** to automatically send SMS or call emergency contacts.

### 5. 👥 Multi-Person Support
- **Tracking Algorithms**: Integrate ByteTrack or SORT to maintain individual identities and detect falls in multi-person scenarios.

### 6. ⌚ Multi-Modal Fusion
- **Sensor Integration**: Combine camera data with **IMU sensor data** (accelerometer/gyroscope) from smartwatches to reduce false positives.

### 7. 🌐 User Dashboard
- **Web UI**: A centralized dashboard using Streamlit or React to monitor multiple camera feeds and view history logs.
