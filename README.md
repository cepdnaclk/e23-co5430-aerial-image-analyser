# e23-co5430-aerial-image-analyser

This project aims to detect and segment infrastructure, such as buildings and roads, from aerial and satellite images using image processing techniques. 

It takes a standard RGB aerial image and outputs a **binary mask** (where white represents infrastructure and black represents the background). The project uses PyTorch and the **Inria Aerial Image Labeling Dataset**.

## Project Structure & Pipeline

The project is structured into four main Jupyter Notebooks:

### 1. Data Preparation and EDA (`01-data-preparation-and-eda.ipynb`)
Feeding massive 5000x5000 pixel images directly into a neural network would result in GPU Out-of-Memory errors. This notebook sets up a robust data pipeline:
* **Train/Validation/Test Split:** Splits the 36 image tiles per city into Train (1-25), Validation (26-30), and Test (31-36).
* **Patch Extraction:** Uses a custom PyTorch `Dataset` to dynamically slice the massive images into smaller, manageable patches (e.g., 256x256 or 512x512 pixels) on the fly.
* **Exploratory Data Analysis (EDA):** Visualizes the patches and ground truth masks to ensure perfect alignment before training.

### 2. Baseline Model (`02-baseline-model.ipynb`)
To establish a minimum performance bar, this notebook implements a `SimpleBaselineCNN`.
* A lightweight Convolutional Neural Network with a basic encoder/decoder and no complex skip-connections.
* Trained for just a few epochs to serve as a lower-bound comparison metric for the advanced models.

### 3. U-Net Training (`03-unet-training.ipynb`)
This notebook contains the core machine learning logic.
* **Data Augmentation:** Uses `albumentations` for dynamic data augmentation (flips, rotations, color shifts) to prevent overfitting and improve robustness under various lighting conditions.
* **Architecture & Transfer Learning:** Implements the **U-Net** architecture using a pre-trained `resnet34` encoder (ImageNet weights) via `segmentation_models_pytorch`. This provides a massive head start in recognizing basic shapes and textures.
* **Loss Function:** Optimized using a custom compound loss combining **Binary Cross Entropy (BCE)** (for pixel-level classification) and **Dice Loss** (to ensure the overall shape matches).
* **Model Checkpointing:** Monitors validation Intersection over Union (IoU) and saves the best performing weights.

### 4. Final Evaluation & Demo (`04-final-evaluation-and-demo.ipynb`)
Evaluates the best trained U-Net model on the unseen Test Set (Tiles 31-36).
* **Metrics Used:** Computes **Intersection over Union (IoU)** and the **Dice Coefficient** to quantify the model's accuracy.
* **Visual Demo & Failure Analysis:** Generates side-by-side plots (Input | Ground Truth | Prediction) to showcase performance and explicitly analyzes failure cases (e.g., buildings obscured by trees or shadows).
