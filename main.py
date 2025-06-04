from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import io
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.cm as cm
from model import get_model
import traceback

app = FastAPI()

# Allow all CORS (adjust if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model()
model.load_state_dict(torch.load("checkpoints/checkpoints/best_model_fold1.pth", map_location=device))
model.to(device)
model.eval()

# Grad-CAM setup
target_layers = [model.model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def annotate(image: Image.Image, text: str, position=(5, 5)):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, fill=(255, 255, 255), font=font)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")
        resized_image = image.resize((224, 224))
        input_tensor = transform(resized_image).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

        # Grad-CAM
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())

        # Apply threshold mask
        threshold = 0.3
        mask = grayscale_cam > threshold

        # Heatmap (jet colormap)
        heatmap = cm.jet(grayscale_cam)[..., :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        # Original RGB image
        original_rgb = np.array(resized_image.convert("RGB"))

        # Apply mask: blend only ROI
        overlaid = original_rgb.copy()
        overlaid[mask] = (0.5 * original_rgb[mask] + 0.5 * heatmap[mask]).astype(np.uint8)

        # Create annotated images
        img_pred_label = "Pneumonia" if pred_class == 1 else "Normal"
        img_confidence = f"{probs[pred_class] * 100:.2f}%"

        img_left = annotate(Image.fromarray(original_rgb), f"Pred: {img_pred_label}")
        img_right = annotate(Image.fromarray(overlaid), f"Confidence: {img_confidence}")

        # Combine with gap
        gap = 10
        combined = Image.new("RGB", (224 * 2 + gap, 224), color=(255, 255, 255))
        combined.paste(img_left, (0, 0))
        combined.paste(img_right, (224 + gap, 0))

        # Encode as PNG
        buf = io.BytesIO()
        combined.save(buf, format="PNG")
        encoded_img = buf.getvalue().hex()

        return JSONResponse(content={
            "pred_class": img_pred_label,
            "confidence_normal": float(probs[0]),
            "confidence_pneumonia": float(probs[1]),
            "image": encoded_img
        })

    except Exception as e:
        print("🔥 Exception occurred:")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
