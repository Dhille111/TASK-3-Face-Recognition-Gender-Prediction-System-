# Face Recognition + Gender Prediction System

This project combines:

- OpenCV Haar Cascade for face detection
- MobileNetV2 transfer learning for face recognition
- OpenCV DNN for gender prediction
- `gender-guesser` plus custom rules for name-based gender hints

## Current structure

```text
TASK 3/
├── capture_faces.py
├── train_model.py
├── recognize_faces.py
├── gender_predictor.py
├── label_map.json
├── requirements.txt
├── data/
│   └── registered_faces/
│       ├── dhille/
│       └── meeravali/
└── models/
    ├── face_model.keras
    ├── gender_net.caffemodel
    └── gender_deploy.prototxt
```

## Rename update

The registered person name has been changed from `kanoj` to `dhille`.

If you already have images under `data/registered_faces/kanoj/`, rename the folder to `data/registered_faces/dhille/` before retraining.

## Setup

```bash
pip install -r requirements.txt
```

## Capture faces

```bash
python capture_faces.py --name Dhille --count 100
```

## Train the model

```bash
python train_model.py
```

This script:

- scans `data/registered_faces/`
- splits the data into train/validation/test sets
- trains a MobileNetV2 classifier
- saves `models/face_model.keras`
- regenerates `label_map.json`

## Recognize faces

```bash
python recognize_faces.py
```

The recognition window shows the predicted name, confidence, and any available gender hint.

## Gender prediction files

The OpenCV gender model files are expected at:

- `models/gender_net.caffemodel`
- `models/gender_deploy.prototxt`

If those files are missing, the code still runs and falls back to name-based gender hints.
## output-images
<img width="1920" height="1011" alt="image" src="https://github.com/user-attachments/assets/2f60da1e-8803-4796-9a2f-17a2d1f05875" />



## Notes

This repository is suitable as a beginner project or college demo, but the recognition model is only as good as the number and quality of registered faces.
