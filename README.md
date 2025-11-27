# CIFAR-10 Image Classification — Full ML Pipeline

## Demo & Live App

* **Video Demo:**  https://www.loom.com/share/50a8b48bcc5f4e65ac0f19da583a0606
* **Live App:** https://mlopsummative-production.up.railway.app/


# Overview

This project demonstrates a fully operational MLOps pipeline for image classification using CIFAR-10.
It handles training, preprocessing, prediction, dockerized deployment, retraining, monitoring, and load testing.

### Highlights

* 87.39% test accuracy
* Real-time predictions (<100ms)
* Automatic image preprocessing & resizing
* Click-to-retrain with transfer learning
* Fully dockerized + scalable
* Interactive web UI + REST API
* Load-tested for up to 100 users


# What the System Does

### **Image Classification**

Classifies images into:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

### **Web Application**

* Upload an image → preview preprocessing → get prediction + confidence
* Visual dashboards for metrics & model interpretation
* Retrain the model with new data

### **Retraining Workflow**

Upload new training images → trigger retraining → model updates automatically.

### **Production Deployment**

* Docker + Docker Compose
* Gunicorn backend
* Scalable replicas
* Load tested with Locust



#  Dataset (CIFAR-10)

| Property     | Value                    |
| ------------ | ------------------------ |
| Total Images | 60,000                   |
| Train        | 50,000                   |
| Test         | 10,000                   |
| Size         | 32×32 RGB                |
| Classes      | 10                       |
| Balanced     | Yes                      |
| Download     | Automatic via TensorFlow |

Dataset source: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)



# Getting Started

## Prerequisites

* Python 3.9+
* pip
* Git
* 4GB+ RAM recommended


## Install & Setup

### Clone repo

git clone https://github.com/cyiza22/ml-pipeline-project.git
cd ml-pipeline-project


### Create & activate virtual environment

python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux


### Install dependencies

pip install --upgrade pip
pip install -r requirements.txt


## Train the Model

Open the notebook:

notebook/ml_pipeline.ipynb

Click **Run All**.

This will:

* download CIFAR-10
* train the CNN
* save model as `models/cifar10_classifier.h5`
* generate metrics & plots


## Run the Web App

python app.py

App runs at:
**[http://localhost:5000](http://localhost:5000)**


# Using the Web App

### Prediction Page

1. Upload image
2. See preview + resized 32×32 version
3. Click **Predict**
4. View predicted class + confidence chart

### Visualizations Page

* Accuracy, loss, precision, recall, F1
* Class distribution
* System metrics: uptime, latency, traffic

### Retrain Page

* Upload new images (any number)
* Click **Start Retraining**
* Model updates automatically

---

# API Endpoints

### Health Check

GET /health

### Predict

POST /predict


### Upload Training Images

POST /upload_training_data


### Retrain Model

POST /retrain


### Metrics

GET /metrics


# Model Performance

### Overall Metrics

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 87.39% |
| Precision | 87.56% |
| Recall    | 87.39% |
| F1        | 87.25% |

### Per-Class Accuracy

| Class      | Accuracy |
| ---------- | -------- |
| Airplane   | 89.5%    |
| Automobile | 93.2%    |
| Bird       | 82.4%    |
| Cat        | 78.9%    |
| Deer       | 89.8%    |
| Dog        | 85.2%    |
| Frog       | 91.7%    |
| Horse      | 90.3%    |
| Ship       | 92.1%    |
| Truck      | 90.8%    |

Run validation:

python test_model_accuracy.py



# Model Limitations

### Works Best With

* small images (32×32–256×256)
* centered objects
* simple backgrounds

### Struggles With

* high-resolution images
* multiple objects
* weird lighting or angles

Reason: CIFAR-10 resolution is low → downsampling causes information loss.


# Load Testing Summary

### Single Container

| Users | RPS  | Latency | Fail % |
| ----- | ---- | ------- | ------ |
| 10    | 8.2  | 45ms    | 0      |
| 50    | 32.1 | 156ms   | 0.2%   |
| 100   | 48.5 | 487ms   | 2.1%   |

### Three Containers

| Users | RPS  | Latency | Fail % |
| ----- | ---- | ------- | ------ |
| 10    | 9.8  | 32ms    | 0      |
| 50    | 47.6 | 76ms    | 0      |
| 100   | 92.4 | 198ms   | 0      |

Three containers = **90% higher throughput**, **60% lower latency**, **0 failures**.




# Project Structure

ML_Pipeline_Project/
│
├── app.py

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

├── README.md

├── notebook/

│   └── ml_pipeline.ipynb
    
│
├── src/

│   ├── preprocessing.py

│   ├── model.py

│   └── prediction.py
│
├── templates/

│   └── index.html
│
├── models/

│   └── best_model.h5

    ├── cifar10_classifier.h5

    ├── model_metadata.pkl

    ├── retrained_model_20251127_203535.h5

    └── training_history.pkl
│
├── data/

│   ├── train/

│   ├── test/

│   └── uploads/
│
├── scripts/

│   ├── deploy.sh

│   ├── run_load_test.sh

│   └── test_api.sh
│
└── tests/

    ├── test_preprocessing.py

    ├── test_model.py

    └── test_api.py


# Resources

* CIFAR-10 Dataset
* TensorFlow Docs
* Flask Docs
