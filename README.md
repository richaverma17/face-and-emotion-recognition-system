# Face and Emotion Recognition System

This project consists of two parts:

1. **Face Recognition** – Binary classification of whether a face belongs to a specific user.
2. **Emotion Recognition** – Multiclass classification of the emotion expressed in the face.


## Task Overview

### Part 1: Face Recognition (Binary Classification)
- Create 2 classes Class 1 and class 0 and capture images
  - Class 1: Your face
  - Class 0: Not your face
- 3 model variants:
  - VGGFace (finetuned)
  - ResNet18 (trained from scratch)
  - ResNet18 (pretrained on ImageNet)

### Part 2: Emotion Recognition (Multiclass Classification)
- Create At least 3 classes/folders: e.g., Happy, Sad, angry and capture images according to the emotion
- Uses the same models as above, adapted for multiclass output

## Dataset Collection

- **Your Face Images:** Captured in varied conditions:
  - Lighting: Bright, Dim
  - Backgrounds: Plain, Cluttered
  - Occlusions: Hand, objects
  - Expressions: Happy, Sad, Angry, Neutral, etc.

- **Other Faces:** Collected from friends or internet (for binary classification)

- Augmentations: Random flips, rotations, brightness jitter


## Environment Requirements

- Python 3.x
- PyTorch
- torchvision
- wandb
- OpenCV

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Face Recognition (Binary Classification)

```bash
python face_train.py --model vggface --epochs 50
```

Available `--model` options: `vggface`, `resnet18_scratch`, `resnet18_pretrained`

### 2. Emotion Recognition (Multiclass Classification)

```bash
python emotion_train.py --model resnet18_pretrained --classes happy sad angry
```


## Demo: Face Unlock + Emotion Response

Simulates an unlock screen with prediction overlay:

```bash
python video_demo.py
```

Output: Shows “Unlocked” if face is detected, emotion label + optional reaction for expression.


## Evaluation Metrics

For both tasks:
- Accuracy
- Precision, Recall, F1-score (for each class)
- Confusion Matrix (Emotion classification)
- Training/Validation loss and accuracy curves (logged to wandb)
