import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from PIL import Image, UnidentifiedImageError

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "rice_disease_model.onnx"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
FRONTEND_DIR = BASE_DIR / "frontend"

IMAGE_SIZE = (224, 224)
MAX_UPLOAD_SIZE = 8 * 1024 * 1024  # 8 MB


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="/static"
)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==========================================
# DISEASE INFORMATION
# ==========================================

DISEASE_INFO = {

    "Bacterial Leaf Blight": {
        "description": (
            "Bacterial Leaf Blight is a bacterial disease "
            "that causes yellowing, drying and lesions along "
            "rice leaf edges."
        ),
        "management": [
            "Use resistant rice varieties.",
            "Maintain proper field drainage.",
            "Avoid excessive nitrogen fertilizer.",
            "Remove infected plant material where appropriate.",
            "Monitor crops regularly for disease symptoms."
        ]
    },

    "Brown Spot": {
        "description": (
            "Brown Spot is a fungal disease that produces "
            "brown oval-shaped spots on rice leaves."
        ),
        "management": [
            "Use healthy and disease-free seeds.",
            "Maintain balanced crop nutrition.",
            "Improve field sanitation.",
            "Remove infected crop residues where appropriate.",
            "Consult an agricultural expert about suitable fungicides."
        ]
    },

    "Healthy Rice Leaf": {
        "description": (
            "The model did not identify symptoms associated "
            "with the six diseases included in its training classes."
        ),
        "management": [
            "Continue regular crop monitoring.",
            "Maintain balanced irrigation and fertilization.",
            "Use preventive crop management practices."
        ]
    },

    "Leaf Blast": {
        "description": (
            "Leaf Blast is a fungal disease that causes "
            "spindle-shaped lesions on rice leaves."
        ),
        "management": [
            "Use resistant rice varieties.",
            "Avoid excessive nitrogen fertilizer.",
            "Monitor crops regularly.",
            "Maintain appropriate field water management.",
            "Consult an agricultural expert about disease control."
        ]
    },

    "Leaf scald": {
        "description": (
            "Leaf Scald is a fungal disease that causes "
            "scald-like lesions and drying on rice leaves."
        ),
        "management": [
            "Use healthy seeds and resistant varieties.",
            "Maintain good field sanitation.",
            "Avoid excessive nitrogen fertilizer.",
            "Monitor leaves for disease development."
        ]
    },

    "Sheath Blight": {
        "description": (
            "Sheath Blight is a fungal disease that commonly "
            "affects the lower leaf sheaths of rice plants."
        ),
        "management": [
            "Avoid excessive crop density.",
            "Maintain appropriate irrigation and drainage.",
            "Remove infected crop residues where appropriate.",
            "Monitor lower plant sheaths regularly.",
            "Consult an agricultural expert about disease management."
        ]
    }
}


# ==========================================
# LOAD MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

logger.info("ONNX model loaded successfully.")
logger.info("Model input: %s", session.get_inputs()[0].shape)
logger.info("Model output: %s", session.get_outputs()[0].shape)
logger.info("Disease classes: %s", class_names)

if len(class_names) != 6:
    raise ValueError("Expected exactly six disease classes.")


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ==========================================
# HEALTH CHECK API
# ==========================================

@app.route("/api/health", methods=["GET"])
def health_check():

    return jsonify({
        "status": "success",
        "message": "Rice Disease Detection API is running.",
        "model_loaded": session is not None,
        "classes": class_names
    }), 200


# ==========================================
# IMAGE PREDICTION API
# ==========================================

@app.route("/api/predict", methods=["POST"])
def predict():

    # Check whether an image was uploaded
    if "image" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No image uploaded. Please select an image."
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Please select an image."
        }), 400

    try:

        # Open and convert uploaded image
        image = Image.open(file.stream).convert("RGB")

        # Resize image to match model input
        image = image.resize(IMAGE_SIZE)

        # Convert image into NumPy array
        image_array = np.array(
            image,
            dtype=np.float32
        )

        # IMPORTANT:
        # The ONNX model already contains the Rescaling layer.
        # Therefore, do NOT normalize the image here.

        # Add batch dimension
        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Run ONNX inference
        predictions = session.run(
            [output_name],
            {input_name: image_array}
        )[0][0]

        # Get predicted class
        predicted_index = int(np.argmax(predictions))

        predicted_class = class_names[predicted_index]

        # Get prediction score
        confidence = float(predictions[predicted_index]) * 100

        # Get disease information
        disease_info = DISEASE_INFO.get(
            predicted_class,
            {
                "description": "No additional information available.",
                "management": []
            }
        )

        # Determine plant status
        if predicted_class == "Healthy Rice Leaf":
            plant_status = "Healthy"
        else:
            plant_status = "Potentially Infected"

        # Log prediction
        logger.info(
            "Prediction: %s | Score: %.2f%%",
            predicted_class,
            confidence
        )

        # Return response
        return jsonify({

            "status": "success",

            "prediction": predicted_class,

            "confidence": round(confidence, 2),

            "plant_status": plant_status,

            "description": disease_info["description"],

            "management": disease_info["management"],

            "disclaimer": (
                "AI-generated prediction for guidance only. "
                "The confidence score is not a calibrated probability "
                "or confirmed agricultural diagnosis. "
                "Consult an agricultural expert for confirmation."
            )

        }), 200

    except (UnidentifiedImageError, OSError, ValueError):

        return jsonify({
            "status": "error",
            "message": "Invalid or unsupported image. Please upload a valid image."
        }), 400

    except Exception:

        logger.exception("Prediction error occurred.")

        return jsonify({
            "status": "error",
            "message": "An internal error occurred during prediction."
        }), 500


# ==========================================
# FILE SIZE ERROR HANDLER
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({
        "status": "error",
        "message": "Image size exceeds the maximum allowed limit of 8 MB."
    }), 413


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )