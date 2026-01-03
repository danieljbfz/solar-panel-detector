# Solar Panel Detector

Learning computer vision for object detection in aerial imagery.

## Project Structure

```
solar-panel-detector/
├── README.md
├── requirements.txt
├── data/
│   └── sample_images/              # Place training and test images here
├── scripts/
│   ├── 01_image_basics.py          # Images as arrays, color channels, edges
│   ├── 02_classical_detection.py   # SIFT, ORB, template matching
│   ├── 03_yolo_inference.py        # Running pretrained models
│   ├── 04_train_detector.py        # Training custom detectors (GPU required)
│   └── 05_batch_pipeline.py        # Batch processing with geospatial outputs
└── outputs/                        # Saved figures and results
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Sample Data

Download solar panel images from the [Solar Panel Detection Dataset](https://www.kaggle.com/datasets/pythonafroz/solar-panel-detection) on Kaggle and place them in `data/sample_images/`.

Alternatively, use any aerial imagery containing solar panels.

## Running Scripts

```bash
cd scripts
python 01_image_basics.py
```

Each script saves outputs to the `outputs/` directory.

## Tech Stack

Python, NumPy, OpenCV, Matplotlib, Ultralytics YOLOv8, GDAL