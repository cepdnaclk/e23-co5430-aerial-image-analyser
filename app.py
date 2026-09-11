# 1. Install required packages
!pip install -q segmentation-models-pytorch albumentations gradio

import os
import glob
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import gradio as gr

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 2. Load U-Net Architecture (ResNet-34)
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(device)

# Load weights if available in Kaggle inputs or current directory
pth_files = glob.glob("/kaggle/input/**/*.pth", recursive=True) + glob.glob("*.pth")
if pth_files:
    print(f"--> Loading trained weights: {pth_files[0]}")
    model.load_state_dict(torch.load(pth_files[0], map_location=device))

model.eval()

# 3. Segmentation Logic
def segment_aerial_image(input_img, threshold=0.5):
    if input_img is None:
        return None, None, "Please upload an image."

    if len(input_img.shape) == 2:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2RGB)
    elif input_img.shape[2] == 4:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGBA2RGB)

    orig_h, orig_w = input_img.shape[:2]
    new_h = (orig_h // 32) * 32 if orig_h >= 32 else 32
    new_w = (orig_w // 32) * 32 if orig_w >= 32 else 32
    resized_img = cv2.resize(input_img, (new_w, new_h))

    transform = ToTensorV2()
    tensor_img = transform(image=resized_img)['image'].float() / 255.0
    tensor_img = tensor_img.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        prob_map = torch.sigmoid(logits).cpu().squeeze().numpy()

    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
    binary_mask_resized = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    overlay = input_img.copy()
    red_mask = np.zeros_like(input_img)
    red_mask[:, :] = [255, 0, 0] # Highlight building structures in RED
    
    building_pixels = binary_mask_resized > 0
    overlay[building_pixels] = cv2.addWeighted(
        input_img[building_pixels], 0.5, red_mask[building_pixels], 0.5, 0
    )

    coverage = (np.sum(binary_mask_resized > 0) / (orig_h * orig_w)) * 100
    status_text = f"### **Building Footprint Coverage:** `{coverage:.2f}%` of total area"

    return binary_mask_resized, overlay, status_text

# 4. Create & Launch UI
with gr.Blocks(title="Aerial Image Infrastructure Analyser") as demo:
    gr.Markdown("# 🛸 Aerial & Satellite Image Analyser\n### Infrastructure & Building Segmentation using U-Net")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Aerial Image")
            threshold_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Confidence Threshold")
            submit_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            output_overlay = gr.Image(label="Detected Buildings Overlay")
            output_mask = gr.Image(label="Binary Footprint Mask")
            output_info = gr.Markdown()

    submit_btn.click(segment_aerial_image, inputs=[input_image, threshold_slider], outputs=[output_mask, output_overlay, output_info])

# Launches both inside Kaggle output AND creates a shareable web link
demo.launch(share=True, inline=True)
