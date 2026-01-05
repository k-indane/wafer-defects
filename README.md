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
├── .gitignore
├── requirements.txt
└── README.md
```

# Inference Guide

This section explains how to:
1. Setup python environment
2. Launch FASTAPI inference service
3. Run inference via Swagger UI (or)
4. Run inference via curl (or)
5. Run inference without FastAPI service using batch script

## Create and activate environment from repo root (conda example)

conda create -n wafer-infer python=3.10 -y

conda activate wafer-infer

pip install -r requirements.txt

## Launch FastAPI Service

uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload

### Available endpoints
- Health check: http://localhost:8000/health
- Interactive API (Swagger UI): http://localhost:8000/docs

## Run Inference using Swagger (Recommended)
1. Expand POST /pipeline endpoint
2. Click 'Try it out'
3. Upload a wafer map image from tests/data/
5. Click Execute

You will recieve a JSON output.

## Annotated image via Swagger (Recommended)
1. Expand POST /pipeline/annotated endpoint
2. Click 'Try it out'
3. Upload a wafer map image from tests/data/
5. Click Execute

You will recieve a PNG image with annotated bounding boxes.

## Run Inference using curl

### JSON inference

curl -X POST "http://localhost:8000/pipeline" -F "file=@tests/data/center_9355.png"

### Annoated image output

curl -X POST "http://localhost:8000/pipeline" -F "file=@tests/data/center_9355.png" --output annotated.png

Image will save in current directory.

You can stop FastAPI server in the uvicorn terminal with 'Ctrl + C'

## Batch Annotation Script

The batch annotation script runs inference on all test images in tests/data/ and outputs to outputs/annotated/

Run the script in repo root

python scripts/annotate_test_images.py

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