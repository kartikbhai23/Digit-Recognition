# ==========================================================
# Import Required Libraries
# ==========================================================

import base64                     # Used to decode Base64 image received from frontend
import io                         # Allows image bytes to be treated like a file
import os                         # Used to check if model file exists

import torch                      # PyTorch main library
import torch.nn.functional as F   # Contains activation functions like Softmax

from PIL import Image, ImageOps   # PIL is used for image processing

from torchvision import transforms  # Image preprocessing utilities

from fastapi import FastAPI, HTTPException  # FastAPI framework

# Middleware to allow frontend (React/HTML) to communicate with backend
from fastapi.middleware.cors import CORSMiddleware

# Used for request body validation
from pydantic import BaseModel

# Import your custom trained model architecture
from model import MNISTModel


# ==========================================================
# Create FastAPI Application
# ==========================================================

app = FastAPI(

    # API title shown in Swagger UI
    title="MNIST Live Digit Recognition API",

    # Description shown in Swagger documentation
    description="PyTorch + FastAPI powered backend for real-time handwritten digit recognition",

    # API version
    version="1.0.0"
)


# ==========================================================
# Enable CORS
# ==========================================================
# Without this, browser blocks requests from another port
# Example:
# Frontend -> localhost:5500
# Backend  -> localhost:8000

app.add_middleware(
    CORSMiddleware,

    # Allow every frontend
    allow_origins=["*"],

    # Allow cookies if needed
    allow_credentials=True,

    # Allow all HTTP methods
    allow_methods=["*"],

    # Allow all headers
    allow_headers=["*"],
)


# ==========================================================
# Request Body Schema
# ==========================================================
# Frontend sends:
#
# {
#    "image":"base64 string..."
# }

class ImageData(BaseModel):
    image: str


# ==========================================================
# Device Selection
# ==========================================================

# If GPU is available use CUDA
# otherwise CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# Create Model
# ==========================================================

# Create same architecture used during training
model = MNISTModel(
    hidden1=406,
    hidden2=256
)


# ==========================================================
# Load Trained Weights
# ==========================================================

MODEL_PATH = "best_model.pth"

# Check whether model exists
if os.path.exists(MODEL_PATH):

    try:

        # Load saved weights
        model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=device
            )
        )

        print(f"[INFO] Loaded model weights from '{MODEL_PATH}' onto {device}")

    except Exception as e:

        print(f"[ERROR] Error loading '{MODEL_PATH}': {e}")

else:

    print(f"[WARNING] Model file '{MODEL_PATH}' not found in current directory.")


# ==========================================================
# Move model to CPU/GPU
# ==========================================================

model.to(device)

# Switch model to inference mode
# Disables dropout and batch normalization updates
model.eval()


# ==========================================================
# Image Preprocessing Pipeline
# ==========================================================

# Exactly same preprocessing used while training

transform = transforms.Compose([

    # Resize image to MNIST size
    transforms.Resize((28, 28)),

    # Convert PIL Image → Tensor
    transforms.ToTensor(),

    # Normalize image
    # (Image - Mean) / Std
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )

])


# ==========================================================
# Image Preprocessing Function
# ==========================================================

def preprocess_base64_image(base64_str: str) -> torch.Tensor:
    """
    Converts Base64 canvas image into
    normalized tensor suitable for the model.
    """

    # Frontend usually sends:
    # data:image/png;base64,xxxxxxxx

    # Remove the header part
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    # Decode Base64 into bytes
    image_bytes = base64.b64decode(base64_str)

    # Read bytes as image
    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("L")       # Convert to grayscale

    # Canvas:
    # White background
    # Black digit
    #
    # MNIST:
    # Black background
    # White digit
    #
    # Therefore invert colors

   
    image = ImageOps.invert(image)

    # Apply resize + tensor + normalization
    tensor = transform(image)

    # Add batch dimension
    #
    # Before:
    # (1,28,28)
    #
    # After:
    # (1,1,28,28)

    tensor = tensor.unsqueeze(0)

    # Move tensor to CPU/GPU
    return tensor.to(device)


# ==========================================================
# Root Endpoint
# ==========================================================

@app.get("/")
def api_root():

    # Health check endpoint

    return {

        "status": "online",

        "framework": "PyTorch + FastAPI",

        "device": str(device),

        "message": "MNIST Live Digit Recognition API is active."
    }


# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.post("/predict")
async def predict_digit(data: ImageData = None):
    """
    Accepts Base64 image payload:
    { "image": "data:image/png;base64,..." }
    """
    if not data or not data.image:
        raise HTTPException(
            status_code=400,
            detail="No image provided. Request body must contain JSON with 'image' key."
        )

    try:
        # Convert image into tensor
        image_tensor = preprocess_base64_image(data.image)

        # Disable gradient calculation
        with torch.no_grad():
            # Forward pass
            outputs = model(image_tensor)

            # Convert logits to probabilities
            probabilities = F.softmax(outputs, dim=1).squeeze(0)

            # Get highest probability
            confidence, prediction = torch.max(probabilities, dim=0)

            # Convert tensor -> Python list
            prob_list = [float(p) for p in probabilities.cpu().numpy()]

        pred_digit = int(prediction.item())
        conf_val = float(confidence.item())
        conf_pct = f"{conf_val * 100:.1f}%"
        all_preds = {str(i): f"{prob_list[i] * 100:.1f}%" for i in range(10)}

        # Return prediction to frontend with full format support
        return {
            "prediction": pred_digit,
            "digit": pred_digit,
            "confidence": conf_val,
            "confidence_pct": conf_pct,
            "probabilities": prob_list,
            "all_predictions": all_preds
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )