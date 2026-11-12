# 🌾 RiceGuard – AI-Based Rice Plant Disease Detection

An AI-powered web application that detects common rice leaf diseases from uploaded images using a deep learning image classification model.

RiceGuard allows users to upload a rice leaf image and receive an AI-generated prediction, confidence score, disease information, and recommended management practices.

The project uses a MobileNetV2-based deep learning model trained on a dataset of six rice leaf categories. The trained model is exported to ONNX format and integrated into a Flask backend for local inference.

---

# 1. Project Overview

Rice is one of the world's most important food crops. Diseases affecting rice plants can reduce crop quality and production if they are not identified and managed in time.

RiceGuard aims to provide an accessible tool that helps users identify common rice leaf diseases through image classification.

Instead of manually examining every image, users can upload a rice leaf photograph and receive an AI-generated result within seconds.

## Main Objectives

- Detect common rice leaf diseases using deep learning.
- Provide an easy-to-use image upload interface.
- Display the predicted disease and model confidence score.
- Provide basic disease descriptions.
- Display suggested disease management and prevention practices.
- Run predictions locally without requiring an external AI API.

---

# 2. Supported Disease Classes

The model supports six categories:

| No. | Class | Description |
|---|---|---|
| 1 | Bacterial Leaf Blight | A bacterial disease that causes yellowing, drying, and lesions along leaf edges. |
| 2 | Brown Spot | A fungal disease that produces brown oval-shaped spots on rice leaves. |
| 3 | Healthy Rice Leaf | A rice leaf classified as healthy among the model's supported categories. |
| 4 | Leaf Blast | A fungal disease that causes characteristic spindle-shaped lesions. |
| 5 | Leaf scald | A fungal disease associated with scald-like lesions and leaf drying. |
| 6 | Sheath Blight | A fungal disease that commonly affects the lower leaf sheaths. |

**Important:** The model can only classify images into these six trained categories. It does not currently guarantee that an uploaded image is a rice leaf or reject every unsupported image.

---

# 3. Technology Stack

## Artificial Intelligence and Machine Learning

- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- ONNX
- ONNX Runtime
- Google Colab

## Backend

- Python 3.12
- Flask
- Flask-CORS
- Pillow

## Frontend

- HTML5
- CSS3
- JavaScript
- Responsive web design

## Development Tools

- Visual Studio Code
- Git
- GitHub
- Google Colab

---

# 4. System Architecture

RiceGuard follows a simple client-server architecture.

```text
              USER
                |
                |
         Opens Web Application
                |
                |
          FRONTEND
        HTML / CSS / JS
                |
                |
        Uploads Leaf Image
                |
                |
          FLASK BACKEND
          Python / Flask
                |
                |
        Image Preprocessing
          Resize to 224x224
          Convert to RGB
                |
                |
          ONNX RUNTIME
                |
                |
        MobileNetV2 Model
                |
                |
       Six-Class Prediction
                |
                |
       Result Processing
                |
                |
       FRONTEND RESULTS
       - Predicted Disease
       - Confidence Score
       - Plant Status
       - Disease Description
       - Management Advice
```

## How the Application Works

1. The user opens the RiceGuard web application.
2. The user selects a rice leaf image.
3. JavaScript previews the selected image.
4. The frontend sends the image to the Flask API.
5. Flask reads and preprocesses the image.
6. The processed image is passed to the ONNX Runtime model.
7. The model generates scores for six classes.
8. The backend selects the class with the highest score.
9. Disease information and management recommendations are retrieved.
10. The frontend displays the prediction results.

---

# 5. Deep Learning Model

## Model Architecture

The project uses MobileNetV2, a convolutional neural network architecture designed for efficient image classification.

The model uses ImageNet pretrained weights and is fine-tuned for the six rice leaf categories.

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 |
| Input Image Size | 224 × 224 |
| Input Channels | 3 (RGB) |
| Number of Classes | 6 |
| Output | Six-class prediction scores |
| Model Format | ONNX |
| Inference Engine | ONNX Runtime |
| Execution Provider | CPU |

### Training Techniques

- Transfer learning using pretrained MobileNetV2.
- Image resizing.
- RGB image processing.
- Random horizontal flipping.
- Random rotation.
- Random zoom.
- Fine-tuning of selected model layers.
- Batch normalization layers frozen during fine-tuning.

### Training Dataset

The project uses a rice leaf disease image dataset containing 3,829 images.

| Disease Class | Images |
|---|---:|
| Bacterial Leaf Blight | 636 |
| Brown Spot | 646 |
| Healthy Rice Leaf | 653 |
| Leaf Blast | 634 |
| Leaf scald | 628 |
| Sheath Blight | 632 |
| **Total** | **3,829** |

### Dataset Split

The dataset was divided using stratified splitting to maintain class proportions.

| Dataset | Images |
|---|---:|
| Training | 2,680 |
| Validation | 574 |
| Testing | 575 |
| **Total** | **3,829** |

### Training Performance

The best validation accuracy achieved during training was approximately:

**85.89%**

This is validation accuracy, not test accuracy. The final held-out test evaluation is a separate step.

---

# 6. Project Directory Structure

The main project structure is:

```text
Rice-Disease-Detection/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── uploads/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/
│   ├── rice_disease_model.onnx
│   ├── rice_disease_model.keras
│   └── class_names.json
│
├── training/
│
├── README.md
└── .gitignore
```

### Folder Explanation

**backend/**

Contains the Flask application, API endpoints, dependencies, and upload-related resources.

**frontend/**

Contains the user interface built with HTML, CSS, and JavaScript.

**models/**

Contains the trained deep learning model and class-name mapping.

**training/**

Reserved for model training notebooks, scripts, and related training resources.

**README.md**

Contains project documentation and setup instructions.

**.gitignore**

Prevents unnecessary files from being uploaded to GitHub.

---

# 7. Installation and Setup Guide

Follow these instructions to run RiceGuard on a Windows computer.

## Requirements

Install the following software:

- Python 3.12 (recommended)
- Visual Studio Code
- Git
- A modern web browser

The project uses CPU-based ONNX Runtime inference, so a dedicated NVIDIA GPU is not required.

## Step 1: Download the Project

Clone the GitHub repository:

```powershell
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate into the project:

```powershell
cd Rice-Disease-Detection
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the actual repository URL.

Alternatively, download the project ZIP file from GitHub and extract it.

## Step 2: Create a Virtual Environment

From the project root directory, execute:

```powershell
python -m venv backend/venv
```

Activate the virtual environment:

```powershell
.\backend\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\backend\venv\Scripts\Activate.ps1
```

Verify Python:

```powershell
python --version
```

Python 3.12 is recommended for this project.

## Step 3: Install Dependencies

Make sure the virtual environment is activated.

Install the required packages:

```powershell
python -m pip install --upgrade pip
```

```powershell
python -m pip install -r backend/requirements.txt
```

The main dependencies include:

| Package | Purpose |
|---|---|
| Flask | Backend web framework |
| Flask-CORS | Cross-origin request support |
| Pillow | Image loading and processing |
| NumPy | Numerical operations |
| ONNX Runtime | Running the trained model |

## Step 4: Verify the Model Files

Make sure the following files exist:

```text
models/
    rice_disease_model.onnx
    class_names.json
```

The ONNX model is required for predictions.

The `class_names.json` file contains the mapping between the model's output positions and the six disease classes.

The Keras model is retained as a training/model-development artifact and is not required for normal ONNX inference.

## Step 5: Run the Application

From the project root directory, with the virtual environment activated, execute:

```powershell
python -u backend/app.py
```

The Flask server should start on:

```text
http://127.0.0.1:5000
```

Open the following address in your browser:

**http://127.0.0.1:5000/**

RiceGuard should now be accessible.

---

# 8. API Documentation

The backend exposes the following endpoints.

## 8.1 Home Endpoint

**URL:**

```http
GET /
```

**Purpose:**

Serves the RiceGuard frontend.

## 8.2 Health Check Endpoint

**URL:**

```http
GET /api/health
```

**Purpose:**

Checks whether the Flask backend is running and the model has been loaded.

**Example response:**

```json
{
  "status": "success",
  "message": "Rice Disease Detection API is running.",
  "model_loaded": true,
  "classes": [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight"
  ]
}
```

## 8.3 Disease Prediction Endpoint

**URL:**

```http
POST /api/predict
```

**Content-Type:**

```text
multipart/form-data
```

**Required field:**

```text
image
```

The endpoint accepts an uploaded image and returns the predicted disease and related information.

**Example response:**

```json
{
  "status": "success",
  "prediction": "Leaf Blast",
  "confidence": 84.93,
  "plant_status": "Potentially Infected",
  "description": "Leaf Blast is a fungal disease that causes spindle-shaped lesions on rice leaves.",
  "management": [
    "Use resistant rice varieties.",
    "Avoid excessive nitrogen fertilizer.",
    "Monitor crops regularly.",
    "Maintain appropriate field water management.",
    "Consult an agricultural expert about disease control."
  ],
  "disclaimer": "AI-generated prediction for guidance only."
}
```

The example illustrates the response structure. Actual predictions and scores depend on the uploaded image.

## API Error Handling

The backend handles common errors, including:

- No image uploaded.
- Empty filename.
- Invalid or unsupported image.
- Image exceeding the 8 MB upload limit.
- Internal prediction errors.

---

# 9. Image Preprocessing and Prediction

Before inference, the uploaded image passes through these steps:

1. Open the uploaded image.
2. Convert it to RGB format.
3. Resize it to 224 × 224 pixels.
4. Convert it into a NumPy float32 array.
5. Add the batch dimension.
6. Pass the array to ONNX Runtime.

**Important preprocessing detail:**

The trained ONNX model already contains its own Rescaling layer.

Therefore, the Flask backend must not normalize the image a second time.

The input should remain in the 0–255 pixel range as float32 values.

---

# 10. Model Files and Deployment

RiceGuard uses ONNX Runtime to perform predictions.

This makes it possible to run the trained model without installing TensorFlow on the deployment computer.

Required runtime files:

```text
models/
    rice_disease_model.onnx
    class_names.json
```

The Keras model is useful for further training and model development but is not necessary for running the Flask inference application.

The project does not require a database or an external AI API for its current functionality.

---

# 11. Current Features

- AI-based rice leaf disease classification.
- Six supported disease categories.
- Image upload functionality.
- Image preview before analysis.
- Drag-and-drop image upload.
- Prediction through Flask API.
- ONNX Runtime inference.
- Confidence score display.
- Healthy and potentially infected status.
- Disease descriptions.
- Disease management recommendations.
- Image size validation.
- Error handling.
- Responsive frontend interface.
- Local CPU-based inference.

---

# 12. Current Limitations and Future Improvements

This project is a working prototype and can be improved further.

Planned improvements include:

1. Improve classification accuracy.
2. Evaluate the model using a confusion matrix.
3. Generate precision, recall, and F1-score reports.
4. Investigate commonly confused disease classes.
5. Add better handling of unsupported and non-rice images.
6. Improve confidence interpretation and prediction reliability.
7. Test the application with more real-world rice leaf images.
8. Explore deployment options for wider accessibility.

---

# 13. Important Usage Disclaimer

RiceGuard provides AI-generated predictions for educational and informational purposes.

The displayed confidence score is the model's highest class score. It is not a calibrated probability or a guarantee that the prediction is correct.

The application is not a replacement for professional agricultural diagnosis.

Users should consult qualified agricultural experts before making important crop treatment decisions.

---

# 14. Developer Information

**Project:** RiceGuard – AI-Based Rice Plant Disease Detection

**Category:** Artificial Intelligence / Machine Learning / Web Development

**Model:** MobileNetV2

**Backend:** Python / Flask

**Frontend:** HTML / CSS / JavaScript

**Inference:** ONNX Runtime

---

# 15. License and Dataset Information

Add an appropriate project license before distributing the software publicly.

The dataset is used for model development. Its original licensing and redistribution terms should be checked separately before redistributing the dataset or using it commercially.

---

**RiceGuard – Supporting smarter rice crop monitoring through artificial intelligence.**