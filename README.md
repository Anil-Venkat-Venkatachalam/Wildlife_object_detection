# Bounding Box Object Detection & Classification using YOLOv8

This project implements wildlife object detection using the YOLO (You Only Look Once) framework. It leverages the Ultralytics YOLOv8 model to train, validate, and test an object detection pipeline on a custom wildlife dataset in YOLO format.
The goal of this project is to accurately detect and classify different wildlife species in images for research, conservation, and monitoring purposes.

---

## 📌 Features
- Preprocessing of YOLO dataset paths for Kaggle environment
- Training YOLOv8 model on custom wildlife dataset
- Inference on test images with bounding box visualization
- Saving trained weights for deployment (animal_detector.pt)

---

## ⚙️ Installation

Clone the repository and install the dependencies:

git clone https://github.com/Anil-Venkat-Venkatachalam/wildlife-object-detection.git
cd wildlife-object-detection

pip install ultralytics --upgrade
pip install matplotlib opencv-python pyyaml torch

---

## 📂 Dataset
The dataset is formatted in YOLO format and contains:
- train/ → Training images and labels
- valid/ → Validation images and labels
- test/ → Test images and labels
- In this project, the dataset is mounted from Kaggle Datasets. You can adjust the dataset paths in the code as needed:
- original_yaml_path = '/kaggle/input/object-detection-wildlife-dataset-yolo-format/final_data/data_wl.yaml'

---

## 🚀 Usage
### 1. Fix Dataset Paths
import yaml

with open(original_yaml_path, 'r') as f:
    data_dict = yaml.safe_load(f)

base_path = '/kaggle/input/object-detection-wildlife-dataset-yolo-format/final_data'
data_dict['train'] = f'{base_path}/train/images'
data_dict['val'] = f'{base_path}/valid/images'
data_dict['test'] = f'{base_path}/test/images'

### 2. Train Model
from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8s.pt')

model.train(
    data=fixed_yaml_path,
    epochs=5,
    imgsz=640,
    batch=16,
    device=0 if device == 'cuda' else 'cpu'
)

### 3. Run Predictions
results = model.predict(
    source=f'{base_path}/test/images',
    conf=0.25,
    save=True
)

### 4. Visualize Results

Bounding boxes are drawn using OpenCV and displayed with Matplotlib.

cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv.putText(img, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

### 5. Save Trained Model
model.save('animal_detector.pt')

---

## 📊 Results
The trained model outputs bounding boxes with class labels and confidence scores.
Example detections are saved automatically in runs/detect/predict/.

---

## 🛠️ Technologies Used
- Python
- Ultralytics YOLOv8
- PyTorch
- OpenCV
- Matplotlib

---

## 📌 Future Work
- Fine-tuning with larger datasets
- Support for video stream inference

---
