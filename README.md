# Automated Wafer Defect Classification & Spatial Detection
This project implements an end-to-end machine learning inference pipeline for analyzing semiconductor wafer defects. It combines CNN-based defect classification with conditional YOLO-based spatial localization, reflecting how defect inspection systems are designed and deployed in real environments. The focus of this project is not only model accuracy, but production-style inference, routing logic, and deployability.

# Overview
1. Classifies the defect type using a convolutional neural network
2. Conditionally applies YOLO-based object detection where spatial localization is meaningful
3. Returns structured outputs suitable for APIs, visualization, or downstream automation

# Model Architecture
Input is grayscale wafer defect map image:
- 0 = No defect / Background
- 1 = Defect

Output is one of nine defect classes:
- none
- Loc
- Edge-Loc
- Center
- Edge-Ring
- Scratch
- Random
- Near-full
- Donut

The CNN performs wafer-level defect detection and determines whether spatial localization is required. reducing unnecessary computation.

Spatial Detection w/YOLO executed only for meaningful defect classes:
- Loc
- Edge-Loc
- Center
- Scratch
- Donut

# Model Benchmarking
In addition to the final architecture, this project evaluated alternative architectures to establish performance baselines. MobileNetv2 was benchmarked as an industry-standard lightweight CNN frequently used in production computer vision systems. Results showed that the custom-designed CNN outperformed MobileNetv2, achieving higher classification accuracy and faster inference latency across all defect classes. A RandomForest model was also benchmarked, providing superior inference latency, but poorly identifying spatially-related classes (Scratch, Edge-Loc, Loc, Edge-Ring).

# Inference Pipeline

Image → Preprocessing → CNN → (conditional) YOLO → Results

Core inference logic is implemented in src/wafer/pipeline.py

# Repository Structure
```
wafer-defects/
├── artifacts/
│   ├── cnn/
│   │   ├── cnn_model.keras
│   │   └── classes.json
│   └── yolo/
│       └── best.pt
│
├── docs/
│   └── An Analysis of Defect Patterns for Automated Wafer Defect Classification.pdf
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── CNN Modeling.ipynb
│   ├── MobileNetV2 Modeling.ipynb
│   ├── YOLO Modeling.ipynb
│   └── Additional modeling notebooks
│
├── src/
│   └── wafer/
│       ├── preprocess.py
│       ├── pipeline.py
│       └── __init__.py
│
├── scripts/
│   └── test_inference.py
│
├── service/
│   └── app.py
│
├── tests/
│   └── data/
│       └── Sample wafer images for inference testing
│ 
├── outputs/
│   └── annotated/
│       └── Sample JSON and annotated images from tests/data/ images
│
├── .gitignore
├── requirements.txt
├── .dockerignore
├── Dockerfile
└── README.md
```

# Inference Guide

This section explains how to perform inference through FastAPI from the Docker image.

1. Clone the repo

```bash
git clone https://github.com/k-indane/wafer-defects/
cd wafer-defects
```

2. Build the Docker image

```bash
docker build -t wafer-defects:latest .
```

3. Run the container

```bash
docker run --rm -p 8000:8000 wafer-defects:latest
```

- Interactive API Endpoint: http://localhost:8000/docs

#### JSON Inference 
1. Expand POST /pipeline endpoint
2. Click 'Try it out'
3. Upload a wafer map image from tests/data/
5. Click Execute

You will recieve a JSON output.

Example:
```json
{
  "ok": true,
  "model_versions": {
    "cnn": "1.0.0",
    "yolo": "1.0.0"
  },
  "cnn": {
    "pred_class": "Center",
    "top_idx": 8,
    "probs": {
      "none": 0.0,
      "Edge-Ring": 0.0,
      "Loc": 0.0,
      "Scratch": 0.0,
      "Near-full": 0.0,
      "Donut": 0.0,
      "Random": 0.0,
      "Edge-Loc": 0.0,
      "Center": 1.0
    }
  },
  "routing": {
    "should_run_yolo": true,
    "spatial_classes": [
      "Center",
      "Donut",
      "Edge-Loc",
      "Loc",
      "Scratch"
    ]
  },
  "yolo": {
    "ran": true,
    "detections": [
      {
        "cls_id": 0,
        "cls_name": "Center",
        "conf": 0.7932320237159729,
        "xyxy": [
          16.349590301513672,
          16.866743087768555,
          30.86593246459961,
          29.061891555786133
        ]
      }
    ]
  },
  "timing_ms": {
    "preprocess": 6.0002000027452596,
    "cnn": 104.23569999693427,
    "yolo": 53.602100000716746,
    "total": 163.83800000039628
  }
}
```

#### Annotated Image
1. Expand POST /pipeline/annotated endpoint
2. Click 'Try it out'
3. Upload a wafer map image from tests/data/
5. Click Execute

You will recieve a PNG image with annotated bounding boxes.

Example:

![Annotated Center Defect on Wafer](outputs/annotated/center_9355_annotated.png)

You can stop the service in the terminal with 'Ctrl + C'

### Run Inference using curl

#### JSON Inference

```bash
curl -X POST "http://localhost:8000/pipeline" -F "file=@tests/data/center_9355.png"
```

#### Annotated Image

```bash
curl -X POST "http://localhost:8000/pipeline" -F "file=@tests/data/center_9355.png" --output annotated.png
```

Image will save to disk.

# Detailed Technical Report
A comprehensive report accompanies this project and covers:
- Exploratory data analysis
- Preprocessing
- Feature creation
- RF/CNN/MobileNetv2/YOLO architecture and training
- Hyperparameter tuning
- Model evaluation
- Benchmarking
- Design tradeoffs
  
Located in docs/An Analysis of Defect Patterns for Automated Wafer Defect Classification.pdf

# Author
Kaustubh Indane

M.S. Data Science AI - Northwestern University

Background in semiconductor process integration

# License
This project is intended for educational and portfolio purposes only.