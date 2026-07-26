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
from model import CNN


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
model = CNN(
    conv1_filters=64,
    conv2_filters=32,
    fc_units=256,
    dropout_rate=0.3
)


# ==========================================================
# Load Trained Weights
# ==========================================================

MODEL_PATH = "Models/FinalModel_98.94.pth"

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
    Converts Base64 canvas image into normalized tensor for PyTorch MNIST model.
    Uses if-else brightness auto-detection and 20x20 bounding-box centering.
    """
    import numpy as np

    # Remove data URL header if present
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    # Decode Base64 into PIL Image (Grayscale)
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    # If-else brightness check for background color:
    img_np = np.array(image)
    if img_np.mean() > 128:
        # High brightness means light/white canvas -> Invert to black background + white digit
        image = ImageOps.invert(image)
        img_np = np.array(image)
    else:
        # Dark canvas -> Keep original black background + white digit
        pass

    # Bounding box crop & 20x20 centering inside 28x28 black matrix
    nonzero_coords = np.argwhere(img_np > 30)
    if len(nonzero_coords) > 0:
        min_y, min_x = nonzero_coords.min(axis=0)
        max_y, max_x = nonzero_coords.max(axis=0)

        pad = 10
        min_y = max(0, min_y - pad)
        min_x = max(0, min_x - pad)
        max_y = min(img_np.shape[0], max_y + pad)
        max_x = min(img_np.shape[1], max_x + pad)

        cropped = image.crop((min_x, min_y, max_x, max_y))
        w, h = cropped.size
        if w > h:
            new_w = 20
            new_h = max(1, int(20 * (h / w)))
        else:
            new_h = 20
            new_w = max(1, int(20 * (w / h)))

        resized_digit = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        final_28x28 = Image.new("L", (28, 28), 0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        final_28x28.paste(resized_digit, (paste_x, paste_y))
        image = final_28x28
    else:
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

    # Apply PyTorch transforms (ToTensor, Normalize 0.1307, 0.3081)
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)

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