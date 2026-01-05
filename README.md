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
├── dev_env.bat
├── .gitignore
└── README.md
```

# Running Inference
Activate environment with dev_ent.bat

Run inference on sample images: python scripts/test_inference.py

  Example Output:
  
  pred: Center | yolo: True | det: 1
  
  pred = CNN predicted class
  
  yolo = whether YOLO was executed
  
  det = number of detected bounding boxes

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

# Deployment Readiness
This project was structured with deployment in mind:
- Separation of preprocessing, inference, and service
- Conditional routing logic
- Production style JSON outputs
- FastAPI service scaffold
- Docker-ready structure

# Author
Kaustubh Indane
M.S. Data Science AI - Northwestern University

Background in semiconductor process integration

# License
This project is intended for educational and portfolio purposes only.
